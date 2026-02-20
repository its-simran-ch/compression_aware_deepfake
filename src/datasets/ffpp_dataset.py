"""
PyTorch Dataset for FaceForensics++ preprocessed face crops.

Consumes the CSV metadata index produced by extract_faces_ffpp.py.
Returns frames as (3, 224, 224) tensors with labels and DWT tensors.

Author: Simran Chaudhary
"""

import os
import csv
import numpy as np
from PIL import Image
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.dwt_utils import face_to_dwt_tensor


# ImageNet normalization (required for EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Training transforms with augmentations."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Eval/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FFPPFrameDataset(Dataset):
    """
    Frame-level dataset for FaceForensics++ face crops.

    CSV columns expected:
        video_id, split, label, compression, manipulation, frame_idx, frame_path

    Returns:
        dict with keys:
        - rgb:         (3, 224, 224) tensor (ImageNet normalized)
        - dwt:         (4, 112, 112) tensor (DWT subbands)
        - label:       int (0 = real, 1 = fake)
        - compression: str (e.g., 'c23')
        - video_id:    str
    """

    def __init__(
        self,
        metadata_csv: str,
        root_dir: str,
        split: str = "train",
        compressions: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        include_dwt: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            metadata_csv: Path to metadata.csv from extract_faces_ffpp.py.
            root_dir: Root directory of face crops (parent of the CSV).
            split: 'train', 'val', or 'test'.
            compressions: List of compressions to include (e.g., ['c23', 'c40']).
            transform: torchvision transforms for RGB images.
            include_dwt: Whether to compute and return DWT tensors.
            max_samples: Limit the dataset size (for quick testing).
        """
        self.root_dir = root_dir
        self.split = split
        self.include_dwt = include_dwt

        # Default transforms
        if transform is None:
            self.transform = get_train_transforms() if split == "train" else get_eval_transforms()
        else:
            self.transform = transform

        # Load metadata
        self.samples = []
        with open(metadata_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split
                if row["split"] != split:
                    continue

                # Filter by compression
                if compressions and row["compression"] not in compressions:
                    continue

                self.samples.append({
                    "frame_path": os.path.join(root_dir, row["frame_path"]),
                    "label": 0 if row["label"] == "real" else 1,
                    "compression": row["compression"],
                    "video_id": row["video_id"],
                    "manipulation": row.get("manipulation", "unknown"),
                })

        # Optionally limit
        if max_samples and len(self.samples) > max_samples:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        print(f"[FFPPFrameDataset] split={split}, compressions={compressions}, "
              f"samples={len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load image
        img = Image.open(sample["frame_path"]).convert("RGB")
        img_np = np.array(img)  # (H, W, 3) uint8

        # RGB tensor (with transforms)
        rgb_tensor = self.transform(img)

        result = {
            "rgb": rgb_tensor,
            "label": sample["label"],
            "compression": sample["compression"],
            "video_id": sample["video_id"],
        }

        # DWT tensor
        if self.include_dwt:
            dwt_tensor = face_to_dwt_tensor(img_np, target_size=112, wavelet="haar")
            result["dwt"] = dwt_tensor

        return result


class FFPPVideoDataset(Dataset):
    """
    Video-level dataset: groups frames by video_id for evaluation.

    Returns all frames for a given video, useful for video-level aggregation.
    """

    def __init__(
        self,
        metadata_csv: str,
        root_dir: str,
        split: str = "test",
        compressions: Optional[List[str]] = None,
        max_frames_per_video: int = 50,
    ):
        self.root_dir = root_dir
        self.transform = get_eval_transforms()
        self.max_frames_per_video = max_frames_per_video

        # Group frames by video_id
        videos = {}
        with open(metadata_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                if compressions and row["compression"] not in compressions:
                    continue

                vid = row["video_id"]
                if vid not in videos:
                    videos[vid] = {
                        "label": 0 if row["label"] == "real" else 1,
                        "compression": row["compression"],
                        "frames": [],
                    }
                videos[vid]["frames"].append(
                    os.path.join(root_dir, row["frame_path"])
                )

        self.videos = list(videos.items())
        print(f"[FFPPVideoDataset] split={split}, videos={len(self.videos)}")

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> dict:
        video_id, info = self.videos[idx]
        frame_paths = info["frames"][:self.max_frames_per_video]

        rgb_tensors = []
        dwt_tensors = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            img_np = np.array(img)
            rgb_tensors.append(self.transform(img))
            dwt_tensors.append(face_to_dwt_tensor(img_np, target_size=112))

        return {
            "video_id": video_id,
            "label": info["label"],
            "compression": info["compression"],
            "rgb": torch.stack(rgb_tensors),     # (N, 3, 224, 224)
            "dwt": torch.stack(dwt_tensors),     # (N, 4, 112, 112)
        }
