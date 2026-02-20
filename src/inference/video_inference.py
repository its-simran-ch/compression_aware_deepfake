#!/usr/bin/env python3
"""
Video inference pipeline: Video file → Real/Fake/Uncertain prediction.

End-to-end pipeline:
1. Load model + checkpoint
2. Sample frames at fixed FPS
3. Detect and crop faces
4. For each face: compute spatial + DWT features → fusion → frame probability
5. Aggregate frame probabilities → video-level prediction

Usage:
    python src/inference/video_inference.py \\
        --video_path test_video.mp4 \\
        --checkpoint results/checkpoints/best_hybrid_c23_c40.pth \\
        --mode hybrid

Author: Simran Chaudhary
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.fusion_classifier import HybridDeepfakeClassifier
from src.utils.video_utils import sample_frames
from src.utils.face_detection import create_detector, detect_face
from src.utils.dwt_utils import face_to_dwt_tensor
from src.utils.metrics import aggregate_frame_to_video


# ImageNet normalization (same as training)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model(checkpoint_path: str, mode: str = "hybrid",
               device: str = "cpu") -> HybridDeepfakeClassifier:
    """Load trained model from checkpoint."""
    model = HybridDeepfakeClassifier(mode=mode, pretrained_spatial=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_video(
    video_path: str,
    model: HybridDeepfakeClassifier,
    mode: str = "hybrid",
    device: str = "cpu",
    target_fps: float = 5.0,
    max_frames: int = 100,
    face_detector: Optional[object] = None,
) -> Dict:
    """
    Run deepfake detection on a video file.

    Args:
        video_path: Path to the input video (.mp4).
        model: Loaded HybridDeepfakeClassifier.
        mode: 'spatial', 'frequency', or 'hybrid'.
        device: 'cpu' or 'cuda'.
        target_fps: Frame sampling rate.
        max_frames: Maximum frames to process.
        face_detector: Pre-initialized MTCNN detector (reuse for efficiency).

    Returns:
        Dict with keys:
        - label:           "REAL" | "FAKE" | "UNCERTAIN"
        - score:           float (mean probability, 0=real, 1=fake)
        - num_frames_used: int
        - frame_probs:     list[float] (per-frame probabilities)
        - sample_frames:   list of (frame_idx, face_rgb, prob) for visualization
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Initialize face detector if not provided
    if face_detector is None:
        face_detector = create_detector(device=device)

    # Step 1: Sample frames
    try:
        frames = sample_frames(video_path, target_fps=target_fps, max_frames=max_frames)
    except Exception as e:
        return {
            "label": "UNCERTAIN",
            "score": 0.5,
            "num_frames_used": 0,
            "error": f"Failed to read video: {e}",
            "frame_probs": [],
            "sample_frames": [],
        }

    if not frames:
        return {
            "label": "UNCERTAIN",
            "score": 0.5,
            "num_frames_used": 0,
            "error": "No frames could be extracted",
            "frame_probs": [],
            "sample_frames": [],
        }

    # Step 2: Detect faces and compute probabilities
    frame_probs = []
    sample_frames_list = []

    model.eval()
    with torch.no_grad():
        for frame_idx, frame_bgr in frames:
            # Detect face
            face_rgb = detect_face(frame_bgr, face_detector, output_size=224)
            if face_rgb is None:
                continue

            # Prepare RGB input
            pil_img = Image.fromarray(face_rgb)
            rgb_tensor = EVAL_TRANSFORM(pil_img).unsqueeze(0).to(device)

            # Prepare DWT input
            dwt_tensor = None
            if mode in ("frequency", "hybrid"):
                dwt_tensor = face_to_dwt_tensor(
                    face_rgb, target_size=112, wavelet="haar"
                ).unsqueeze(0).to(device)

            # Forward
            rgb_in = rgb_tensor if mode != "frequency" else None
            dwt_in = dwt_tensor if mode != "spatial" else None
            logit = model(rgb_input=rgb_in, dwt_input=dwt_in)
            prob = torch.sigmoid(logit).item()

            frame_probs.append(prob)

            # Keep sample frames for visualization (first 5)
            if len(sample_frames_list) < 5:
                sample_frames_list.append((frame_idx, face_rgb, prob))

    # Step 3: Aggregate
    result = aggregate_frame_to_video(frame_probs)
    result["frame_probs"] = frame_probs
    result["sample_frames"] = sample_frames_list

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run deepfake detection inference on a video.",
    )
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to input video file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint.")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["spatial", "frequency", "hybrid"])
    parser.add_argument("--target_fps", type=float, default=5.0)
    parser.add_argument("--max_frames", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, mode=args.mode, device=device)

    # Run inference
    t0 = time.time()
    result = predict_video(
        args.video_path, model, mode=args.mode, device=device,
        target_fps=args.target_fps, max_frames=args.max_frames,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"PREDICTION RESULT")
    print(f"{'='*50}")
    print(f"  Video:       {args.video_path}")
    print(f"  Label:       {result['label']}")
    print(f"  Confidence:  {result['score']:.4f} ({result['score']*100:.1f}%)")
    print(f"  Frames used: {result['num_frames_used']}")
    print(f"  Time:        {elapsed:.2f}s")
    if result.get("error"):
        print(f"  Error:       {result['error']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
