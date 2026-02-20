"""
Face detection and cropping using MTCNN (via facenet-pytorch).

Provides:
- detect_face(): single frame → cropped 224×224 face (or None)
- extract_faces_from_frames(): batch of frames → list of face crops

Author: Simran Chaudhary
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple

# MTCNN from facenet-pytorch (pip install facenet-pytorch)
from facenet_pytorch import MTCNN


def create_detector(device: str = "cpu") -> MTCNN:
    """
    Create an MTCNN face detector.

    Args:
        device: 'cpu' or 'cuda'

    Returns:
        Configured MTCNN detector instance.
    """
    detector = MTCNN(
        image_size=224,
        margin=40,               # extra margin around the face bounding box
        min_face_size=60,        # minimum face size in pixels
        thresholds=[0.6, 0.7, 0.7],  # detection thresholds
        factor=0.709,
        post_process=False,      # return raw PIL crops, not normalized tensors
        select_largest=True,     # pick the largest face when multiple detected
        keep_all=False,
        device=device,
    )
    return detector


def detect_face(
    frame_bgr: np.ndarray,
    detector: MTCNN,
    output_size: int = 224,
    margin: int = 40,
) -> Optional[np.ndarray]:
    """
    Detect and crop the largest face from a BGR frame.

    Args:
        frame_bgr: Input frame in BGR format (OpenCV convention).
        detector: MTCNN detector instance.
        output_size: Output face crop size (square).
        margin: Extra margin around the detected bounding box.

    Returns:
        RGB face crop as np.ndarray (output_size × output_size × 3),
        or None if no face detected.
    """
    # Convert BGR → RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Detect bounding boxes
    boxes, probs = detector.detect(pil_img)

    if boxes is None or len(boxes) == 0:
        return None

    # Take the first (largest) face
    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box

    # Add margin, clamping to image bounds
    h, w = frame_rgb.shape[:2]
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    face_crop = frame_rgb[y1:y2, x1:x2]

    if face_crop.size == 0:
        return None

    # Resize to output_size × output_size
    face_crop = cv2.resize(face_crop, (output_size, output_size),
                           interpolation=cv2.INTER_LINEAR)
    return face_crop


def extract_faces_from_frames(
    frames: List[Tuple[int, np.ndarray]],
    detector: MTCNN,
    output_size: int = 224,
) -> List[Tuple[int, np.ndarray]]:
    """
    Extract face crops from a list of (frame_idx, frame_bgr) tuples.

    Args:
        frames: List of (frame_index, bgr_frame) from video_utils.sample_frames.
        detector: MTCNN instance.
        output_size: Square crop size.

    Returns:
        List of (frame_index, face_crop_rgb) for frames where a face was found.
        Frames without detected faces are silently skipped.
    """
    results = []
    for frame_idx, frame_bgr in frames:
        face = detect_face(frame_bgr, detector, output_size)
        if face is not None:
            results.append((frame_idx, face))
    return results
