# import gradio as gr
# import requests
# import io
# from PIL import Image

# # --- IMPORTANT: Point this to your live Render URL ---
# BACKEND_URL = "https://omnicataract-x-api.onrender.com" 

# def predict_cataract(image):
#     if image is None:
#         return "Please upload a fundus image first."

#     # Convert image to bytes
#     buffer = io.BytesIO()
#     image.save(buffer, format="PNG")
#     buffer.seek(0)

#     # Send to your Render Backend
#     try:
#         response = requests.post(
#             f"{BACKEND_URL}/predict", 
#             files={"file": ("upload.png", buffer, "image/png")}
#         )
        
#         if response.status_code != 200:
#             return f"Server Error: {response.status_code}"
            
#         result = response.json()
        
#     except Exception as e:
#         return f"Connection Error: {e}"

#     # Format the result
#     if result["quality_status"] == "poor":
#         return f"⚠️ **Quality Warning:** {result['message']}"

#     status = "✅ **Cataract Detected**" if result["cataract_detected"] else "✅ **No Cataract Detected**"
#     confidence = result["cataract_confidence"] * 100
    
#     return f"""
#     ### {status}
#     - **Confidence:** {confidence:.1f}%
#     - **Quality Score:** {result['quality_score']*100:.1f}%
#     - **Severity Grade:** {result['severity_grade']}
    
#     **Analysis:** {result['message']}
#     """

# # Build the UI
# with gr.Blocks(title="OmniCataract-X") as demo:
#     gr.Markdown("# 🔬 OmniCataract-X")
#     gr.Markdown("### Cataract Screening Research Prototype")
    
#     with gr.Row():
#         with gr.Column():
#             img_input = gr.Image(type="pil", label="Upload Fundus Image")
#             btn = gr.Button("Analyze Image", variant="primary")
            
#         with gr.Column():
#             result_text = gr.Markdown(label="Diagnosis Result")
            
#     btn.click(fn=predict_cataract, inputs=[img_input], outputs=[result_text])
    
#     gr.Markdown("---\n⚠️ **Disclaimer:** This is a research prototype. Not a substitute for professional diagnosis.")

# if __name__ == "__main__":
#     demo.launch()




import importlib
import os
import socket
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
BACKEND_URL = os.environ.get("OMNICATARACT_BACKEND_URL", "https://omnicataract-x-api.onrender.com")
LOCAL_STUDENT_PATH = BASE_DIR / "models" / "student_distilled.pt"


def _find_free_port(start_port: int = 7860, end_port: int = 7900) -> int:
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"No free port found in range {start_port}-{end_port}")


def main() -> None:
    # Load the application module
    app5 = importlib.import_module("05_application")

    student = None
    if LOCAL_STUDENT_PATH.exists():
        try:
            # Load your PyTorch student model for Grad-CAM only if the local
            # environment has the required model dependencies.
            p2 = importlib.import_module("02_core_detection_model")
            p4 = importlib.import_module("04_optimization_distillation_export")

            device = "cpu"
            student = p4.build_student_model(p2, backbone_name=p4.STUDENT_BACKBONE, pretrained=False)
            student.load_state_dict(torch.load(LOCAL_STUDENT_PATH, map_location=device, weights_only=False))
            student.to(device).eval()
            print("✅ Student model loaded")
        except Exception as exc:
            student = None
            print(f"[warn] Grad-CAM model unavailable locally; launching frontend without heatmap support: {exc}")
    else:
        print(f"[warn] Missing {LOCAL_STUDENT_PATH}; launching frontend without heatmap support")

    share_frontend = os.environ.get("OMNICATARACT_FRONTEND_SHARE", "0") != "0"

    print(f"🚀 Launching Gradio frontend against {BACKEND_URL}...")
    if share_frontend:
        print("[info] Gradio share link is enabled. This is the frontend link you want, not the backend API URL.")

    server_port = int(os.environ.get("OMNICATARACT_FRONTEND_PORT", _find_free_port()))
    if server_port != 7860:
        print(f"[info] Port 7860 was busy; using {server_port} instead.")

    # Launch the frontend
    app5.launch_frontend(
        backend_url=BACKEND_URL,
        student_model=student,
        server_name="127.0.0.1",
        server_port=server_port,
        share=share_frontend,
    )


if __name__ == "__main__":
    main()