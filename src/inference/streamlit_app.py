"""
Streamlit Demo App: Compression-Aware Video Deepfake Detection

Upload a video → get Real/Fake/Uncertain prediction with confidence.

Usage:
    # Local (Mac):
    streamlit run src/inference/streamlit_app.py

    # Colab (with ngrok):
    # See README.md for Colab-specific instructions

Author: Simran Chaudhary
"""

import os
import sys
import tempfile
import time

import streamlit as st
import numpy as np
from PIL import Image

# Ensure project root is on path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="centered",
)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-real {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .result-fake {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .result-uncertain {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🔍 Video Deepfake Detector")
st.markdown(
    "**Compression-Aware Detection using Spatial + Frequency Features**\n\n"
    "Upload a video to analyze whether it contains deepfake manipulation."
)
st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ──────────────────────────────────────────────
# Sidebar Config
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    checkpoint_path = st.text_input(
        "Model Checkpoint Path",
        value="results/checkpoints/best_hybrid_c23_c40.pth",
        help="Path to the trained model checkpoint file."
    )

    mode = st.selectbox(
        "Detection Mode",
        options=["hybrid", "spatial", "frequency"],
        index=0,
        help="hybrid = spatial + frequency features (recommended)."
    )

    target_fps = st.slider("Sampling FPS", 1.0, 10.0, 5.0, 0.5,
                           help="Frames per second to sample from the video.")
    max_frames = st.slider("Max Frames", 10, 200, 100, 10,
                           help="Maximum number of frames to analyze.")

    st.divider()
    st.markdown("### 📊 Architecture")
    st.markdown("""
    - **Spatial**: EfficientNet-B0
    - **Frequency**: DWT (Haar) + CNN
    - **Fusion**: Concatenate → MLP
    - **Aggregation**: Mean frame probs
    """)

    st.divider()
    st.caption("MSc Research Project — 2026")


# ──────────────────────────────────────────────
# Main Area
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Supported formats: MP4, AVI, MOV, MKV. Max recommended: 2 minutes."
)

if uploaded_file is not None:
    # Show video preview
    st.video(uploaded_file)

    # Analyze button
    if st.button("🔬 Analyze Video", type="primary", use_container_width=True):

        # Validate checkpoint exists
        abs_ckpt = os.path.join(ROOT_DIR, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
        if not os.path.exists(abs_ckpt):
            st.error(
                f"❌ Checkpoint not found: `{abs_ckpt}`\n\n"
                "Please train a model first or provide the correct path."
            )
            st.stop()

        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # Import inference (lazy to avoid import errors if deps missing)
            from src.inference.video_inference import load_model, predict_video
            from src.utils.face_detection import create_detector
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

            with st.spinner("Loading model..."):
                model = load_model(abs_ckpt, mode=mode, device=device)
                face_detector = create_detector(device=device)

            with st.spinner("Analyzing video... This may take a moment."):
                t0 = time.time()
                result = predict_video(
                    tmp_path, model, mode=mode, device=device,
                    target_fps=target_fps, max_frames=max_frames,
                    face_detector=face_detector,
                )
                elapsed = time.time() - t0

            # ── Display Results ──
            st.divider()
            st.subheader("📋 Results")

            label = result["label"]
            score = result["score"]
            n_frames = result["num_frames_used"]

            # Main result card
            if label == "REAL":
                st.markdown(
                    f"<div class='result-real'>"
                    f"<h1>✅ REAL</h1>"
                    f"<p>Confidence: <strong>{(1 - score)*100:.1f}%</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif label == "FAKE":
                st.markdown(
                    f"<div class='result-fake'>"
                    f"<h1>❌ FAKE</h1>"
                    f"<p>Confidence: <strong>{score*100:.1f}%</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='result-uncertain'>"
                    f"<h1>⚠️ UNCERTAIN</h1>"
                    f"<p>Score: <strong>{score*100:.1f}%</strong></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fake Probability", f"{score:.4f}")
            with col2:
                st.metric("Frames Analyzed", n_frames)
            with col3:
                st.metric("Processing Time", f"{elapsed:.1f}s")

            # Sample frames
            if result.get("sample_frames"):
                st.divider()
                st.subheader("🖼️ Sample Frames")
                cols = st.columns(min(len(result["sample_frames"]), 5))
                for i, (frame_idx, face_rgb, prob) in enumerate(result["sample_frames"]):
                    with cols[i]:
                        st.image(face_rgb, caption=f"Frame {frame_idx}\nP(fake)={prob:.3f}",
                                 use_container_width=True)

            # Frame probability distribution
            if result.get("frame_probs") and len(result["frame_probs"]) > 1:
                st.divider()
                st.subheader("📈 Frame-level Probabilities")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(result["frame_probs"], color="#4361ee", linewidth=1.5)
                ax.axhline(y=0.5, color="#dc3545", linestyle="--", alpha=0.5, label="Threshold")
                ax.set_xlabel("Frame Index")
                ax.set_ylabel("P(Fake)")
                ax.set_ylim(-0.05, 1.05)
                ax.legend()
                ax.set_title("Per-Frame Fake Probability")
                st.pyplot(fig)
                plt.close(fig)

            if result.get("error"):
                st.warning(f"⚠️ {result['error']}")

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

else:
    # Placeholder when no video uploaded
    st.info("👆 Upload a video file to get started.")

    st.divider()
    st.subheader("ℹ️ How it works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1️⃣ Upload")
        st.markdown("Upload an MP4 video file (up to ~2 min recommended).")

    with col2:
        st.markdown("### 2️⃣ Analyze")
        st.markdown("The model samples frames, detects faces, and extracts spatial + frequency features.")

    with col3:
        st.markdown("### 3️⃣ Result")
        st.markdown("Get a Real/Fake/Uncertain label with confidence score.")
