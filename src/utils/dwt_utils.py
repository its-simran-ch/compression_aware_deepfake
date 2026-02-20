"""
Discrete Wavelet Transform (DWT) utilities for frequency-domain feature extraction.

Uses PyWavelets to compute 2D DWT on face crops, producing a fixed-size
tensor suitable for the frequency branch CNN.

Wavelet: Haar (db1) — simple, fast, well-suited for detecting
compression artifacts and edge-level manipulations.

Author: Simran Chaudhary
"""

import numpy as np
import pywt
import torch
from typing import Tuple


def compute_dwt_2d(
    image: np.ndarray,
    wavelet: str = "haar",
    level: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute single-level 2D DWT on a grayscale image.

    Args:
        image: Grayscale image as 2D numpy array (H, W), values in [0, 255].
        wavelet: Wavelet family name (default: 'haar' = 'db1').
        level: Decomposition level (we use 1 for simplicity).

    Returns:
        Tuple of (cA, cH, cV, cD) — approximation, horizontal detail,
        vertical detail, diagonal detail. Each is shape (H/2, W/2).
    """
    # Normalize to [0, 1] for numerically stable wavelet coefficients
    image_norm = image.astype(np.float32) / 255.0

    coeffs = pywt.dwt2(image_norm, wavelet)
    cA, (cH, cV, cD) = coeffs

    return cA, cH, cV, cD


def face_to_dwt_tensor(
    face_rgb: np.ndarray,
    target_size: int = 112,
    wavelet: str = "haar",
) -> torch.Tensor:
    """
    Convert an RGB face crop to a 4-channel DWT tensor.

    Pipeline:
    1. Convert RGB → Grayscale
    2. Resize to (target_size*2, target_size*2) so DWT output is (target_size, target_size)
    3. Apply 2D DWT → 4 subbands (cA, cH, cV, cD)
    4. Stack into a 4-channel tensor of shape (4, target_size, target_size)

    Args:
        face_rgb: RGB face crop, shape (H, W, 3), values in [0, 255].
        target_size: Desired spatial size of each subband.
        wavelet: Wavelet name.

    Returns:
        Tensor of shape (4, target_size, target_size) — float32.
    """
    import cv2

    # Step 1: RGB → Grayscale
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)

    # Step 2: Resize so DWT output matches target_size
    # DWT halves the spatial dimensions, so input needs to be 2× target
    input_size = target_size * 2
    gray = cv2.resize(gray, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Step 3: 2D DWT
    cA, cH, cV, cD = compute_dwt_2d(gray, wavelet=wavelet)

    # Step 4: Stack into 4-channel tensor
    # Each subband should be approximately (target_size, target_size)
    subbands = [cA, cH, cV, cD]

    # Ensure exact size (DWT can sometimes be off by 1 pixel)
    resized = []
    for sb in subbands:
        if sb.shape[0] != target_size or sb.shape[1] != target_size:
            sb = cv2.resize(sb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        resized.append(sb)

    tensor = np.stack(resized, axis=0).astype(np.float32)  # (4, H, W)
    return torch.from_numpy(tensor)


def batch_faces_to_dwt(
    faces: np.ndarray,
    target_size: int = 112,
    wavelet: str = "haar",
) -> torch.Tensor:
    """
    Convert a batch of face crops to DWT tensors.

    Args:
        faces: Batch of RGB faces, shape (B, H, W, 3), uint8.
        target_size: Spatial size for each DWT subband.
        wavelet: Wavelet name.

    Returns:
        Tensor of shape (B, 4, target_size, target_size).
    """
    tensors = []
    for face in faces:
        t = face_to_dwt_tensor(face, target_size=target_size, wavelet=wavelet)
        tensors.append(t)
    return torch.stack(tensors, dim=0)
