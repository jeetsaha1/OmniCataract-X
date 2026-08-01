"""
OmniCataract-X - FastAPI Backend for Render Deployment
"""
import os
import io
import json
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

QUALITY_GATE_THRESHOLD = 0.7
MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 100

# Initialize FastAPI app
app = FastAPI(
    title="OmniCataract-X API",
    description="Cataract Screening Research Prototype",
    version="3.0"
)

# CORS - Update this with your actual frontend URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production!
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the ONNX detector
ONNX_PATH = "models/omnicataract_student_int8.onnx"
THRESHOLDS_PATH = "models/severity_summary.json"

print(f"Loading model from {ONNX_PATH}...")
detector = p4.CataractDetectorONNX(ONNX_PATH, severity_thresholds_path=THRESHOLDS_PATH)
print("✅ Model loaded successfully!")

class PredictionResponse(BaseModel):
    cataract_detected: bool
    cataract_confidence: float
    quality_score: float
    quality_status: str
    severity_score: float
    severity_grade: str
    message: str

def validate_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Validate and load uploaded image"""
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
    """
    Predict cataract from uploaded fundus image
    """
    file_bytes = await file.read()
    image = validate_uploaded_image(file_bytes)
    
    # Run inference
    result = detector.predict(image)
    
    # Build message
    if result["quality_status"] == "poor":
        message = "Image quality appears too low for a reliable assessment. Please retake the photo with better lighting and focus."
    elif not result["cataract_detected"]:
        message = "No cataract detected. This is a screening result, not a diagnosis."
    else:
        message = (f"Cataract detected with {result['severity_grade'].lower()} severity "
                  f"(proxy score, not a clinical grade). Please consult an ophthalmologist.")
    
    result["message"] = message
    return result

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)