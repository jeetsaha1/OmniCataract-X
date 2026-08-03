"""
OmniCataract-X - FastAPI Backend for Render Deployment
"""
import os
import io
import base64
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_GRADCAM_IMPORT_ERROR = None
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as exc:
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None
    show_cam_on_image = None
    _GRADCAM_IMPORT_ERROR = exc

# Load modules
def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

p4 = load_module("04_optimization_distillation_export.py", "phase4_module")

# Configuration
DISCLAIMER_TEXT = (
    "This is a research prototype. It is NOT a substitute for professional "
    "ophthalmological diagnosis. Severity is a statistically-derived proxy, "
    "not a clinically validated LOCS III grade."
)

MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 100

# Initialize FastAPI app
app = FastAPI(
    title="OmniCataract-X API",
    description="Cataract Screening Research Prototype",
    version="3.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX detector
ONNX_PATH = "models/omnicataract_student_int8.onnx"
THRESHOLDS_PATH = "models/severity_summary.json"
STUDENT_PATH = "models/student_distilled.pt"

print(f"Loading model from {ONNX_PATH}...")
detector = p4.CataractDetectorONNX(ONNX_PATH, severity_thresholds_path=THRESHOLDS_PATH)
print("✅ Model loaded successfully!")

student_model = None
if os.path.exists(STUDENT_PATH):
    try:
        phase2 = load_module("02_core_detection_model.py", "phase2_module")
        device = "cpu"
        student_model = p4.build_student_model(phase2, backbone_name=p4.STUDENT_BACKBONE, pretrained=False)
        student_model.load_state_dict(torch.load(STUDENT_PATH, map_location=device, weights_only=False))
        student_model.to(device).eval()
        print("✅ Student model loaded successfully for Grad-CAM")
    except Exception as exc:
        student_model = None
        print(f"[warn] Grad-CAM student model unavailable: {exc}")

class PredictionResponse(BaseModel):
    cataract_detected: bool
    cataract_confidence: float
    quality_score: float
    quality_status: str
    severity_score: float
    severity_grade: str
    message: str
    heatmap_base64: Optional[str] = None


def _image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SingleLogitWrapper(nn.Module):
    def __init__(self, detector_module: nn.Module):
        super().__init__()
        self.detector = detector_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.detector(x)["detection_logits"]
        return logits.unsqueeze(1)


def get_gradcam_target_layer(model: nn.Module):
    return model.backbone.blocks[6]


def generate_heatmap(image: Image.Image) -> Optional[str]:
    if student_model is None:
        return None

    resized = image.convert("RGB").resize((224, 224))
    rgb_float = np.array(resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb_float - mean) / std
    input_tensor = torch.tensor(normalized.transpose(2, 0, 1)).unsqueeze(0).float()

    try:
        device = next(student_model.parameters()).device
        input_tensor = input_tensor.to(device)
    except StopIteration:
        pass

    wrapped = SingleLogitWrapper(student_model)
    target_layer = get_gradcam_target_layer(student_model)

    if GradCAMPlusPlus is not None and show_cam_on_image is not None and ClassifierOutputTarget is not None:
        with GradCAMPlusPlus(model=wrapped, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]
        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=0.6)
        return _image_to_base64(Image.fromarray(overlay))

    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        student_model.zero_grad(set_to_none=True)
        wrapped(input_tensor)[:, 0].sum().backward()

        activation = activations.get("value")
        gradient = gradients.get("value")
        if activation is None or gradient is None:
            return None

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activation).sum(dim=1).relu().squeeze(0)
        cam = cam - cam.min()
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        cam_np = cam.detach().cpu().numpy()
        cam_np = np.array(Image.fromarray((cam_np * 255).astype(np.uint8)).resize((224, 224))) / 255.0
        heatmap = np.zeros_like(rgb_float)
        heatmap[..., 0] = cam_np
        heatmap[..., 1] = cam_np * 0.35
        overlay = np.clip(0.6 * rgb_float + 0.4 * heatmap, 0.0, 1.0)
        return _image_to_base64(Image.fromarray((overlay * 255).astype(np.uint8)))
    finally:
        forward_handle.remove()
        backward_handle.remove()

def validate_uploaded_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="File is not a recognizable image format.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Corrupted or unreadable image file: {e}")
    
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    width, height = image.size
    
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(status_code=422, detail=f"Image too small ({width}x{height}).")
    
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(status_code=422, detail=f"Image too large ({width}x{height}).")
    
    return image

@app.get("/")
async def root():
    return {
        "message": "OmniCataract-X API",
        "status": "running",
        "disclaimer": DISCLAIMER_TEXT
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes)
    result = detector.predict(image)
    
    if result["quality_status"] == "poor":
        message = "Image quality appears too low for a reliable assessment. Please retake the photo."
    elif not result["cataract_detected"]:
        message = "No cataract detected. This is a screening result, not a diagnosis."
    else:
        message = f"Cataract detected with {result['severity_grade'].lower()} severity. Please consult an ophthalmologist."
    
    result["message"] = message
    result["heatmap_base64"] = generate_heatmap(image)
    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)