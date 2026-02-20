"""
PyTorch Dataset for Celeb-DF v2 preprocessed face crops.

Used for cross-dataset evaluation only (model trained on FF++).

Expected directory structure after face extraction:
    data/celeb_df_faces/
    ├── real/
    │   ├── {video_id}/
    │   │   ├── 000000.png
    │   │   └── ...
    └── fake/
        ├── {video_id}/
        │   ├── 000000.png
        │   └── ...

Or with a metadata CSV index similar to FF++.

Author: Simran Chaudhary
"""

import os
import csv
import glob
import numpy as np
from PIL import Image
from typing import Optional, List

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils.dwt_utils import face_to_dwt_tensor


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CelebDFFrameDataset(Dataset):
    """
    Frame-level dataset for Celeb-DF v2 face crops.

    Can load from either:
    1. A CSV index (same format as FF++ metadata.csv)
    2. A directory structure (real/ and fake/ folders)
    """

    def __init__(
        self,
        root_dir: str,
        metadata_csv: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        include_dwt: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            root_dir: Root directory of Celeb-DF face crops.
            metadata_csv: Optional CSV index (if available).
            transform: Image transforms.
            include_dwt: Whether to compute DWT tensors.
            max_samples: Limit dataset size.
        """
        self.root_dir = root_dir
        self.include_dwt = include_dwt
        self.transform = transform or get_eval_transforms()

        self.samples = []

        if metadata_csv and os.path.exists(metadata_csv):
            self._load_from_csv(metadata_csv)
        else:
            self._load_from_directory(root_dir)

        if max_samples and len(self.samples) > max_samples:
            import random
            random.seed(42)
            self.samples = random.sample(self.samples, max_samples)

        print(f"[CelebDFFrameDataset] samples={len(self.samples)}")

    def _load_from_csv(self, csv_path: str):
        """Load samples from CSV metadata."""
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append({
                    "frame_path": os.path.join(self.root_dir, row["frame_path"]),
                    "label": 0 if row["label"] == "real" else 1,
                    "video_id": row.get("video_id", "unknown"),
                })

    def _load_from_directory(self, root_dir: str):
        """Load samples by scanning real/ and fake/ subdirectories."""
        for label_name, label_int in [("real", 0), ("fake", 1)]:
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(label_dir):
                # Try alternative names
                for alt in ["Celeb-real", "YouTube-real"] if label_name == "real" else ["Celeb-synthesis"]:
                    alt_dir = os.path.join(root_dir, alt)
                    if os.path.isdir(alt_dir):
                        label_dir = alt_dir
                        break

            if not os.path.isdir(label_dir):
                print(f"  [WARN] Directory not found: {label_dir}")
                continue

            # Scan for images (could be in subdirectories per video)
            images = sorted(
                glob.glob(os.path.join(label_dir, "**", "*.png"), recursive=True) +
                glob.glob(os.path.join(label_dir, "**", "*.jpg"), recursive=True)
            )

            for img_path in images:
                # Extract video_id from parent folder
                vid_id = os.path.basename(os.path.dirname(img_path))
                self.samples.append({
                    "frame_path": img_path,
                    "label": label_int,
                    "video_id": vid_id,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        img = Image.open(sample["frame_path"]).convert("RGB")
        img_np = np.array(img)

        rgb_tensor = self.transform(img)

        result = {
            "rgb": rgb_tensor,
            "label": sample["label"],
            "video_id": sample["video_id"],
        }

        if self.include_dwt:
            dwt_tensor = face_to_dwt_tensor(img_np, target_size=112, wavelet="haar")
            result["dwt"] = dwt_tensor

        return result


class CelebDFVideoDataset(Dataset):
    """
    Video-level dataset for Celeb-DF v2.
    Groups frames by video_id for video-level evaluation.
    """

    def __init__(
        self,
        root_dir: str,
        metadata_csv: Optional[str] = None,
        max_frames_per_video: int = 50,
    ):
        self.root_dir = root_dir
        self.transform = get_eval_transforms()
        self.max_frames_per_video = max_frames_per_video

        # Load all frames first
        frame_ds = CelebDFFrameDataset(
            root_dir=root_dir,
            metadata_csv=metadata_csv,
            include_dwt=False,
        )

        # Group by video_id
        videos = {}
        for s in frame_ds.samples:
            vid = s["video_id"]
            if vid not in videos:
                videos[vid] = {"label": s["label"], "frames": []}
            videos[vid]["frames"].append(s["frame_path"])

        self.videos = list(videos.items())
        print(f"[CelebDFVideoDataset] videos={len(self.videos)}")

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
            "rgb": torch.stack(rgb_tensors),
            "dwt": torch.stack(dwt_tensors),
        }
