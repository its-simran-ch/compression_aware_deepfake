"""
Video utilities for frame sampling and metadata extraction.

Provides functions to:
- Get video metadata (FPS, frame count, duration) via OpenCV
- Sample frames at a fixed rate (default 5 fps, max 100 frames)

Author: Simran Chaudhary
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata using OpenCV.

    Returns:
        dict with keys: fps, frame_count, duration_sec, width, height
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["frame_count"] / max(info["fps"], 1e-6)
    cap.release()
    return info


def sample_frames(
    video_path: str,
    target_fps: float = 5.0,
    max_frames: int = 100,
) -> List[Tuple[int, np.ndarray]]:
    """
    Sample frames from a video at a fixed target FPS.

    Args:
        video_path: Path to the video file.
        target_fps: Desired sampling rate (frames per second).
        max_frames: Maximum number of frames to return.

    Returns:
        List of (frame_index, frame_bgr) tuples.
        Frames are in BGR format (OpenCV default).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0:
        native_fps = 30.0  # fallback

    # Compute sampling interval (in terms of frame indices)
    step = max(1, int(round(native_fps / target_fps)))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            frames.append((frame_idx, frame))
            if len(frames) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frames


def sample_frames_uniform(
    video_path: str,
    n_frames: int = 32,
) -> List[Tuple[int, np.ndarray]]:
    """
    Sample exactly n_frames uniformly distributed across the video.

    Useful when you want a fixed number of frames regardless of video length.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append((int(idx), frame))

    cap.release()
    return frames
