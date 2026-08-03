"""
05_application.py
===================
OmniCataract-X v3.0 — Phase 5: Application & UI (final phase).

Run this AFTER Phase 4's exit milestone has passed (a working, benchmarked
.onnx file must exist, plus the PyTorch student checkpoint for Grad-CAM++).

WHAT THIS FILE COVERS (maps to the plan's Task 5.1 - 5.4):
  5.1  FastAPI backend: a single /predict endpoint wrapping Phase 4's
       CataractDetectorONNX, with real error handling for corrupted files,
       non-image uploads, and degenerate image sizes.
  5.2  Gradio frontend: upload -> calls the backend -> displays the
       binary result, a quality-gate warning, a severity gauge, and a
       Grad-CAM++ heatmap, with a mandatory disclaimer.
  5.3  Grad-CAM++ integration against the ORIGINAL PyTorch student model
       (not the ONNX export — pytorch_grad_cam needs real gradients,
       which the ONNX graph doesn't expose), with a simple cache so the
       same uploaded image isn't re-processed twice.
  5.4  End-to-end tests: real image round-trips through the full stack,
       plus explicit edge-case tests (corrupted file, non-image file,
       extreme image dimensions) verified to fail SAFELY (a clean 4xx
       response) rather than crashing the server.

WHY THE SAFETY-FIRST ORDER IN THE /predict RESPONSE MATTERS:
The backend always computes and returns quality_status, but it is the
FRONTEND (Section 5.2 below) that decides not to show a diagnosis when
quality is poor. This mirrors the plan's app-flow design explicitly: a
user should never receive a false-confidence result on an unusable image.

EXIT MILESTONE (this is the project's final exit milestone):
  End-to-end flow works: upload -> quality check -> detection -> severity
  -> heatmap -> disclaimer, tested on both real and edge-case images, with
  no manual intervention required.
"""

import io
import csv
import time
import hashlib
import argparse
import threading
import importlib.util
from pathlib import Path
from typing import Optional, Tuple
import os
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel
import uvicorn

import gradio as gr
import requests

_GRADCAM_IMPORT_ERROR = None
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception as exc:  # environment-dependent import guard
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None
    show_cam_on_image = None
    _GRADCAM_IMPORT_ERROR = exc


# ==========================================================================
# CONFIG
# ==========================================================================
# DRIVE_ROOT = "/content/drive/MyDrive/OmniCataract-X"
# DEFAULT_ONNX_PATH = f"{DRIVE_ROOT}/exports/cataract_detector_int8.onnx"
# DEFAULT_THRESHOLDS_PATH = f"{DRIVE_ROOT}/results/phase3/severity_summary.json"
# DEFAULT_PHASE4_CHECKPOINT = f"{DRIVE_ROOT}/checkpoints/phase4/student_best.pt"
# RESULTS_DIR = f"{DRIVE_ROOT}/results/phase5"


# Automatically detect the folder where this script is located
DRIVE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Point to the local 'models' folder where you downloaded your files
DEFAULT_ONNX_PATH = os.path.join(DRIVE_ROOT, "models", "omnicataract_student_int8.onnx")
DEFAULT_THRESHOLDS_PATH = os.path.join(DRIVE_ROOT, "models", "severity_summary.json")
DEFAULT_PHASE4_CHECKPOINT = os.path.join(DRIVE_ROOT, "models", "student_distilled.pt")

RESULTS_DIR = os.path.join(DRIVE_ROOT, "results", "phase5")

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

QUALITY_GATE_THRESHOLD = 0.7   # WHY 0.7, not 0.5: this is a screening tool —
                                # erring toward "ask for a retake" is safer
                                # than risking a diagnosis on a poor image.
                                # Documented as a calibratable constant, not
                                # a permanently fixed value — see Phase 2/3's
                                # note on deriving thresholds from real
                                # validation ROC curves before shipping.

MAX_IMAGE_DIMENSION = 4096
MIN_IMAGE_DIMENSION = 100

DISCLAIMER_TEXT = (
    "This is a research prototype. It is NOT a substitute for professional "
    "ophthalmological diagnosis. Severity is a statistically-derived proxy, "
    "not a clinically validated LOCS III grade. If you have concerns about "
    "your eyes or vision, please consult an eye care professional."
)


# ==========================================================================
# UTILITY — load numbered-file modules (same pattern as Phases 3 and 4)
# ==========================================================================
def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# TASK 5.1 — FASTAPI BACKEND
# ==========================================================================
class PredictionResponse(BaseModel):
    """
    WHY a Pydantic model rather than a plain dict: FastAPI uses this to
    auto-generate response validation AND API documentation (visible at
    /docs) — free correctness checking and free docs from one declaration.
    """
    cataract_detected: bool
    cataract_confidence: float
    quality_score: float
    quality_status: str
    severity_score: float
    severity_grade: str
    message: str


def build_message(result: dict) -> str:
    """WHY this is a separate, testable function rather than inlined in the
    endpoint: the exact wording is something you'll want to review, tweak,
    and unit-test independently of the HTTP plumbing around it."""
    if result["quality_status"] == "poor":
        return ("Image quality appears too low for a reliable assessment. "
                "Please retake the photo with better lighting and focus.")
    if not result["cataract_detected"]:
        return "No cataract detected. This is a screening result, not a diagnosis."
    return (f"Cataract detected with {result['severity_grade'].lower()} severity "
            f"(proxy score, not a clinical grade). Please consult an ophthalmologist.")


def validate_uploaded_image(file_bytes: bytes) -> Image.Image:
    """
    WHY this validation is a separate function called explicitly, rather
    than relying on PIL to just fail wherever it fails: this is exactly
    the edge-case surface the plan's Task 5.4 asks to be tested
    (corrupted files, non-image files, extreme dimensions) — having one
    function that raises a clear, specific HTTPException for each failure
    mode makes the endpoint itself simple and makes each failure mode
    independently testable.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()  # WHY verify() first: catches truncated/corrupted
                         # files without fully decoding them, cheaply.
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="File is not a recognizable image format.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Corrupted or unreadable image file: {e}")

    # WHY re-open after verify(): PIL's verify() leaves the image object in
    # a state that can't be used for further processing — re-opening fresh
    # is the documented, correct pattern here, not a redundant step.
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    width, height = image.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Image too small ({width}x{height}). Minimum dimension is {MIN_IMAGE_DIMENSION}px."
        )
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=422,
            detail=f"Image too large ({width}x{height}). Maximum dimension is {MAX_IMAGE_DIMENSION}px."
        )

    return image


def create_app(onnx_path: str = DEFAULT_ONNX_PATH,
               thresholds_path: str = DEFAULT_THRESHOLDS_PATH) -> FastAPI:
    """
    WHY the ONNX session is loaded HERE, once, and closed over by the
    endpoint (rather than loaded inside the /predict function): loading an
    ONNX Runtime session has real overhead. Doing it once at app-creation
    time — not per-request — is what makes the API fast enough to be
    useful; loading it per-request would silently reintroduce most of the
    latency Phase 4's distillation and quantization work was for.
    """
    phase4 = load_module("04_optimization_distillation_export.py", "phase4_module")
    detector = phase4.CataractDetectorONNX(onnx_path, severity_thresholds_path=thresholds_path)

    app = FastAPI(
        title="OmniCataract-X API",
        description="Research prototype cataract screening API. " + DISCLAIMER_TEXT,
    )

    # WHY CORS is wide-open (allow_origins=["*"]) here: this is a local
    # research-prototype demo where the Gradio frontend and this backend
    # both run on localhost during development. A real deployment should
    # restrict this to the actual frontend's origin — flagged here so it
    # isn't silently carried into a production config unexamined.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        file_bytes = await file.read()
        image = validate_uploaded_image(file_bytes)

        result = detector.predict(image)
        result["message"] = build_message(result)
        return result

    return app


def run_backend(onnx_path: str = DEFAULT_ONNX_PATH, thresholds_path: str = DEFAULT_THRESHOLDS_PATH,
                 host: str = BACKEND_HOST, port: int = BACKEND_PORT) -> None:
    """WHY this is a blocking call: intended for standalone/production use
    (e.g. `python 05_application.py --backend`), where the process's whole
    job is to run the server. For Colab notebook use where you want the
    backend running WHILE ALSO doing other things in the same session, use
    run_backend_in_thread() instead."""
    app = create_app(onnx_path, thresholds_path)
    uvicorn.run(app, host=host, port=port)


def run_backend_in_thread(onnx_path: str = DEFAULT_ONNX_PATH,
                           thresholds_path: str = DEFAULT_THRESHOLDS_PATH,
                           host: str = BACKEND_HOST, port: int = BACKEND_PORT) -> threading.Thread:
    """
    WHY this exists specifically for Colab: a single Colab cell/session is
    one process. Running FastAPI in a background daemon thread lets a
    LATER cell in the same notebook launch the Gradio frontend and have it
    successfully call `http://127.0.0.1:8000/predict` — both "processes"
    living in the same Colab runtime without needing separate terminals.
    """
    app = create_app(onnx_path, thresholds_path)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # WHY a short wait-and-poll loop rather than a fixed sleep: uvicorn
    # startup time is not perfectly predictable; polling /health is the
    # correct way to know the server is actually ready before returning
    # control to the caller (who is likely about to make a request).
    for _ in range(50):
        try:
            r = requests.get(f"http://{host}:{port}/health", timeout=0.2)
            if r.status_code == 200:
                print(f"[ok] Backend ready at http://{host}:{port}")
                break
        except requests.exceptions.ConnectionError:
            time.sleep(0.1)
    else:
        print("[warn] Backend did not respond to /health within the timeout. "
              "It may still be starting — check manually before relying on it.")

    return thread


# ==========================================================================
# TASK 5.3 — GRAD-CAM++ INTEGRATION (runs against PyTorch, not ONNX)
# ==========================================================================
class SingleLogitWrapper(nn.Module):
    """
    WHY this wrapper exists: pytorch_grad_cam's target classes
    (ClassifierOutputTarget) expect a model whose output can be indexed
    like (batch, num_classes) — our detection head outputs a raw (batch,)
    tensor. Reshaping to (batch, 1) and targeting index 0 is the standard,
    minimal adaptation for a single-logit binary head.
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.detector(x)["detection_logits"]
        return logits.unsqueeze(1)  # (B,) -> (B, 1)


_gradcam_cache: dict = {}  # WHY module-level, not a class: simplest possible
                            # cache for a single-process demo app; see
                            # generate_gradcam_overlay's docstring for the
                            # cache-key rationale.


def _image_hash(pil_image: Image.Image) -> str:
    """WHY hash the raw bytes rather than e.g. the filename: the same
    image re-uploaded under a different filename (or the same filename
    re-uploaded with different content) must be treated correctly either
    way — content-based hashing is the only approach that's actually
    correct for a cache key here."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


def get_gradcam_target_layer(student_model: nn.Module):
    """
    WHY blocks[6] specifically: this is the SAME hook point verified and
    used for feature-map distillation in Phase 4 (see
    get_student_hook_modules) — the last MobileNetV3 stage before the
    final classifier head, at 7x7 spatial resolution. Reusing an
    already-verified structural fact rather than re-guessing it here.
    """
    return student_model.backbone.blocks[6]


def generate_gradcam_overlay(student_model: nn.Module, pil_image: Image.Image,
                              img_size: int = 224, alpha: float = 0.4,
                              use_cache: bool = True) -> Image.Image:
    """
    WHY this operates on the ORIGINAL PyTorch model, never the ONNX export:
    Grad-CAM needs real backward-pass gradients through named layers, which
    an ONNX Runtime session does not expose. This is a deliberate,
    documented split — the ONNX model is what's fast for the actual
    prediction; the PyTorch model is kept around specifically for this
    explainability step.
    """
    cache_key = _image_hash(pil_image)
    if use_cache and cache_key in _gradcam_cache:
        return _gradcam_cache[cache_key]

    def _compose_overlay(rgb_float: np.ndarray, cam_2d: np.ndarray) -> Image.Image:
        cam_2d = np.clip(cam_2d, 0.0, 1.0)
        heatmap = np.zeros_like(rgb_float)
        heatmap[..., 0] = cam_2d
        heatmap[..., 1] = cam_2d * 0.35
        overlay = np.clip((1.0 - alpha) * rgb_float + alpha * heatmap, 0.0, 1.0)
        return Image.fromarray((overlay * 255).astype(np.uint8))

    def _prepare_input() -> Tuple[np.ndarray, torch.Tensor]:
        resized = pil_image.convert("RGB").resize((img_size, img_size))
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

        return rgb_float, input_tensor

    student_model.eval()
    rgb_float, input_tensor = _prepare_input()

    if GradCAMPlusPlus is None:
        target_layer = get_gradcam_target_layer(student_model)
        activations = {}
        gradients = {}

        def forward_hook(_module, _inputs, output):
            activations["value"] = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0].detach()

        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
        try:
            wrapped = SingleLogitWrapper(student_model)
            student_model.zero_grad(set_to_none=True)
            wrapped(input_tensor)[:, 0].sum().backward()

            activation = activations.get("value")
            gradient = gradients.get("value")
            if activation is None or gradient is None:
                raise RuntimeError("Grad-CAM fallback could not capture activations or gradients.")

            weights = gradient.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation).sum(dim=1).relu().squeeze(0)
            cam = cam - cam.min()
            cam_max = cam.max()
            if cam_max > 0:
                cam = cam / cam_max

            cam_image = Image.fromarray((cam.detach().cpu().numpy() * 255).astype(np.uint8))
            cam_image = cam_image.resize((img_size, img_size), Image.BILINEAR)
            overlay_image = _compose_overlay(rgb_float, np.array(cam_image).astype(np.float32) / 255.0)
        finally:
            forward_handle.remove()
            backward_handle.remove()
    else:
        wrapped = SingleLogitWrapper(student_model)
        target_layer = get_gradcam_target_layer(student_model)

        with GradCAMPlusPlus(model=wrapped, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0]

        overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True, image_weight=1 - alpha)
        overlay_image = Image.fromarray(overlay)

    if use_cache:
        _gradcam_cache[cache_key] = overlay_image
    return overlay_image


# ==========================================================================
# TASK 5.2 — GRADIO FRONTEND
# ==========================================================================
def severity_gauge_html(severity_grade: str, severity_score: float) -> str:
    """WHY a small hand-built HTML bar rather than a generic gr.Slider:
    a slider implies the user can change the value, which is misleading
    for a read-only result display — plain HTML avoids that confusion."""
    colors = {"Mild": "#0f9b8e", "Moderate": "#e2a72e", "Severe": "#d9534f"}
    color = colors.get(severity_grade, "#6b7280")
    pct = int(severity_score * 100)
    return f"""
    <div style="font-family: sans-serif;">
      <div style="font-weight:600; color:{color}; margin-bottom:4px;">
        Severity (proxy): {severity_grade} ({pct}%)
      </div>
      <div style="background:#eee; border-radius:6px; height:14px; width:100%;">
        <div style="background:{color}; width:{pct}%; height:14px; border-radius:6px;"></div>
      </div>
    </div>
    """


def call_backend_predict(pil_image: Image.Image, backend_url: str) -> dict:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    buf.seek(0)

    response = requests.post(f"{backend_url}/predict", files={"file": ("upload.png", buf, "image/png")})
    if response.status_code != 200:
        raise gr.Error(f"Backend error ({response.status_code}): {response.json().get('detail', 'unknown error')}")
    return response.json()


def format_result_for_display(result: dict, student_model: nn.Module = None,
                               uploaded_image: Image.Image = None) -> tuple:
    """
    WHY this is a standalone module-level function rather than a closure
    defined inside build_gradio_interface: this is where the plan's
    core safety requirement actually lives — a poor-quality image must
    show ONLY the retake warning, never a diagnosis alongside it. Logic
    this important needs to be directly unit-testable without needing to
    spin up or introspect a Gradio Blocks object at all. build_gradio_interface
    below is now a thin wrapper that just wires this function to the UI.

    Returns: (result_markdown, severity_html, heatmap_image_or_None, heatmap_visible)
    """
    if result["quality_status"] == "poor":
        return (f"⚠️ {result['message']}", "", None, False)

    confidence_pct = int(result["cataract_confidence"] * 100)
    binary_text = (
        f"**Cataract Detected** (confidence: {confidence_pct}%)"
        if result["cataract_detected"]
        else f"**No Cataract Detected** (confidence: {100 - confidence_pct}%)"
    )

    severity_html = ""
    heatmap = None
    if result["cataract_detected"] and student_model is not None and uploaded_image is not None:
        severity_html = severity_gauge_html(result["severity_grade"], result["severity_score"])
        heatmap = generate_gradcam_overlay(student_model, uploaded_image)

    return (binary_text, severity_html, heatmap, True)


def build_gradio_interface(backend_url: str = BACKEND_URL, student_model: nn.Module = None) -> gr.Blocks:
    """
    WHY student_model is an optional parameter rather than loaded inside
    this function: Grad-CAM needs the PyTorch model in memory, which the
    caller (a notebook cell, or main()) already has loaded from a
    checkpoint — passing it in avoids this function silently trying (and
    likely failing) to reload a checkpoint path it doesn't actually know.
    """

    def predict_and_display(uploaded_image: Image.Image):
        if uploaded_image is None:
            raise gr.Error("Please upload an image first.")

        result = call_backend_predict(uploaded_image, backend_url)
        text, severity_html, heatmap, visible = format_result_for_display(
            result, student_model, uploaded_image
        )
        return text, severity_html, heatmap, gr.update(visible=visible)

    with gr.Blocks(title="OmniCataract-X — Research Prototype") as demo:
        gr.Markdown("## OmniCataract-X — Cataract Screening Research Prototype")
        gr.Markdown(f"*{DISCLAIMER_TEXT}*")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload a fundus image")
                submit_btn = gr.Button("Analyze", variant="primary")
            with gr.Column():
                result_text = gr.Markdown(label="Result")
                severity_display = gr.HTML(label="Severity")
                heatmap_output = gr.Image(label="Grad-CAM++ Heatmap (where the model looked)", visible=True)

        submit_btn.click(
            fn=predict_and_display,
            inputs=[image_input],
            outputs=[result_text, severity_display, heatmap_output, heatmap_output],
        )

        gr.Markdown(f"---\n{DISCLAIMER_TEXT}")

    return demo


def launch_frontend(backend_url: str = BACKEND_URL, student_model: nn.Module = None, **launch_kwargs) -> None:
    demo = build_gradio_interface(backend_url, student_model)
    demo.launch(**launch_kwargs)


# ==========================================================================
# TASK 5.4 — END-TO-END TESTING
# ==========================================================================
def make_synthetic_test_image(size: tuple = (300, 300), seed: int = 0) -> Image.Image:
    """WHY synthetic: this function is used by both the smoke test (no
    real fundus images available) and can be swapped for real images in
    an actual E2E run — same call signature either way."""
    rng = np.random.RandomState(seed)
    arr = (rng.rand(*size, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_e2e_test(client, test_images: list, output_csv: str = None) -> list:
    """
    WHY this accepts a `client` parameter (either a real `requests`-backed
    session pointed at a live server, or a FastAPI `TestClient`) rather
    than hardcoding one: the smoke test uses TestClient (no real server
    needed, faster, fully reliable in any environment); a real E2E run
    against a genuinely running backend uses `requests` — this function's
    logic is identical either way since both expose a compatible `.post()`.
    """
    results = []

    for i, image in enumerate(test_images):
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        t0 = time.perf_counter()
        response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
        elapsed_ms = (time.perf_counter() - t0) * 1000

        row = {
            "image_index": i,
            "status_code": response.status_code,
            "elapsed_ms": round(elapsed_ms, 2),
        }
        if response.status_code == 200:
            row.update(response.json())
        else:
            row["error"] = response.json().get("detail", "unknown")

        results.append(row)
        print(f"[e2e] image {i}: status={row['status_code']} elapsed={row['elapsed_ms']:.1f}ms")

    if output_csv:
        import pandas as pd
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_csv, index=False)
        print(f"[ok] E2E results saved to {output_csv}")

    return results


def run_edge_case_tests(client) -> dict:
    """
    WHY every case here expects a 4xx status, never a 200 AND never an
    unhandled 500: the whole point of Task 5.4's edge-case tests is
    proving the API fails SAFELY on bad input — a crash (500) or a
    false-confidence 200 on garbage input would both be real production
    bugs, not just cosmetic ones.
    """
    outcomes = {}

    # 1. Corrupted file (valid-looking bytes that aren't a real image)
    corrupted_bytes = b"this is not a real image file, just random text pretending to be one"
    r = client.post("/predict", files={"file": ("corrupted.png", io.BytesIO(corrupted_bytes), "image/png")})
    outcomes["corrupted_file"] = {"status": r.status_code, "passed": 400 <= r.status_code < 500}

    # 2. Non-image file (a real file type, just not an image)
    text_bytes = b"hello world, this is a plain text file"
    r = client.post("/predict", files={"file": ("notes.txt", io.BytesIO(text_bytes), "text/plain")})
    outcomes["non_image_file"] = {"status": r.status_code, "passed": 400 <= r.status_code < 500}

    # 3. Extremely large image
    huge_image = make_synthetic_test_image(size=(5000, 5000))
    buf = io.BytesIO()
    huge_image.save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("huge.png", buf, "image/png")})
    outcomes["extremely_large_image"] = {"status": r.status_code, "passed": 400 <= r.status_code < 500}

    # 4. Extremely small image
    tiny_image = make_synthetic_test_image(size=(20, 20))
    buf = io.BytesIO()
    tiny_image.save(buf, format="PNG")
    buf.seek(0)
    r = client.post("/predict", files={"file": ("tiny.png", buf, "image/png")})
    outcomes["extremely_small_image"] = {"status": r.status_code, "passed": 400 <= r.status_code < 500}

    all_passed = all(o["passed"] for o in outcomes.values())
    for name, outcome in outcomes.items():
        status_word = "PASS" if outcome["passed"] else "FAIL"
        print(f"[{status_word}] Edge case '{name}': got HTTP {outcome['status']} "
              f"(expected 4xx, safe rejection)")

    return {"outcomes": outcomes, "all_passed": all_passed}


# ==========================================================================
# CLI ENTRY POINT
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description="OmniCataract-X Phase 5: Application")
    parser.add_argument("--smoke-test", action="store_true",
                         help="In-process test with TestClient + a tiny untrained model, no real server/checkpoints needed.")
    parser.add_argument("--backend", action="store_true", help="Run the FastAPI backend standalone (blocking).")
    parser.add_argument("--onnx-path", type=str, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--thresholds-path", type=str, default=DEFAULT_THRESHOLDS_PATH)
    args = parser.parse_args()

    if args.backend:
        run_backend(args.onnx_path, args.thresholds_path)
        return

    if args.smoke_test:
        print("=" * 70)
        print("PHASE 5 SMOKE TEST (in-process TestClient, tiny untrained model)")
        print("=" * 70)

        phase2 = load_module("02_core_detection_model.py", "phase2_module")
        phase3 = load_module("03_severity_engine.py", "phase3_module")
        phase4 = load_module("04_optimization_distillation_export.py", "phase4_module")

        print("\n--- Building a tiny untrained student model + exporting to ONNX ---")
        student = phase2.CataractDetectorDualHead(backbone_name="mobilenetv3_large_100", pretrained=False)
        severity_probe = phase3.SeverityProbe(input_dim=student.feature_dim)
        combined = phase4.CombinedExportModel(student, severity_probe)

        onnx_path = "/tmp/phase5_smoke_test.onnx"
        phase4.export_to_onnx(combined, onnx_path)

        print("\n--- Creating FastAPI app with the smoke-test ONNX model ---")
        app = create_app(onnx_path=onnx_path, thresholds_path="/nonexistent/path.json")
        client = TestClient(app)

        r = client.get("/health")
        assert r.status_code == 200
        print("[TEST PASS] /health endpoint responds correctly.")

        print("\n--- Running E2E test on synthetic images ---")
        test_images = [make_synthetic_test_image(seed=i) for i in range(5)]
        results = run_e2e_test(client, test_images, output_csv="/tmp/phase5_e2e_results.csv")
        assert all(r["status_code"] == 200 for r in results), "Some valid images were unexpectedly rejected"
        assert all("cataract_detected" in r for r in results), "Response schema missing expected fields"
        print("[TEST PASS] All valid synthetic images processed successfully with correct schema.")

        print("\n--- Running edge-case tests ---")
        edge_results = run_edge_case_tests(client)
        assert edge_results["all_passed"], "One or more edge cases failed to be safely rejected!"
        print("[TEST PASS] All edge cases safely rejected with 4xx, no crashes.")

        print("\n--- Testing Grad-CAM++ generation ---")
        fake_image = make_synthetic_test_image(seed=42)
        heatmap = generate_gradcam_overlay(student, fake_image)
        assert heatmap.size == (224, 224), f"Unexpected heatmap size: {heatmap.size}"
        print("[TEST PASS] Grad-CAM++ heatmap generated with correct dimensions.")

        # WHY test the cache explicitly: silently NOT hitting the cache
        # (e.g. due to a hashing bug) would just look like "it's a bit
        # slower," never like a visible failure — worth checking directly.
        cache_size_before = len(_gradcam_cache)
        _ = generate_gradcam_overlay(student, fake_image)  # same image again
        assert len(_gradcam_cache) == cache_size_before, "BUG: cache grew on a repeated identical image"
        print("[TEST PASS] Grad-CAM cache correctly reused for an identical repeated image.")

        print("\n--- Testing message builder ---")
        good_result = {"quality_status": "good", "cataract_detected": True, "severity_grade": "Moderate"}
        msg = build_message(good_result)
        assert "Moderate" in msg or "moderate" in msg
        poor_result = {"quality_status": "poor", "cataract_detected": False, "severity_grade": "Mild"}
        msg_poor = build_message(poor_result)
        assert "retake" in msg_poor.lower()
        print("[TEST PASS] Message builder produces correct, distinguishable messages.")

        print("\n[ok] PHASE 5 SMOKE TEST COMPLETE — all checks passed.")
        print("=" * 70)
    else:
        print("Usage:\n"
              "  --smoke-test    Run the full in-process test suite (no real server needed)\n"
              "  --backend       Run the FastAPI backend standalone (blocking)\n\n"
              "For interactive Colab use, call run_backend_in_thread() followed by "
              "launch_frontend() from separate notebook cells, e.g.:\n\n"
              "  from importlib import import_module\n"
              "  app5 = import_module('05_application')\n"
              "  app5.run_backend_in_thread(onnx_path='...', thresholds_path='...')\n"
              "  app5.launch_frontend(student_model=my_loaded_pytorch_student, share=True)\n")


if __name__ == "__main__":
    main()
