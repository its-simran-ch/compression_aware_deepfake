#!/usr/bin/env python3
"""
Prepare official train/val/test splits for FaceForensics++.

FaceForensics++ provides official split indices (720 train / 140 val / 140 test)
based on video IDs. This script generates a splits JSON file that all downstream
scripts can reference.

Usage:
    python scripts/prepare_ffpp_splits.py \\
        --data_root /content/drive/MyDrive/FFPP_raw \\
        --output splits.json

Author: Simran Chaudhary
"""

import argparse
import json
import os
import glob


# ──────────────────────────────────────────────────────────────
# Official FaceForensics++ split IDs
# Source: https://github.com/ondyari/FaceForensics (splits/)
# These are the video *indices* (000–999) used by the authors.
# ──────────────────────────────────────────────────────────────

# We generate default numerical splits if the official splits are not provided.
# The official repo has JSON files listing the pairs; here we use the standard
# 720 / 140 / 140 partition as described in the paper.

TRAIN_COUNT = 720
VAL_COUNT = 140
TEST_COUNT = 140


def discover_video_ids(data_root: str) -> list:
    """Discover all video IDs from the original_sequences folder."""
    originals_dir = os.path.join(data_root, "original_sequences", "youtube")

    # Check all compression subdirectories
    for comp in ["raw", "c23", "c40"]:
        videos_dir = os.path.join(originals_dir, comp, "videos")
        if os.path.isdir(videos_dir):
            video_files = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
            if video_files:
                ids = [os.path.splitext(os.path.basename(f))[0] for f in video_files]
                return sorted(set(ids))

    # Fallback: generate default 0-999 range
    print("[WARN] Could not discover video IDs from disk. Using default 0-999 range.")
    return [str(i).zfill(3) for i in range(1000)]


def create_splits(video_ids: list) -> dict:
    """Create train/val/test splits from ordered video IDs."""
    n = len(video_ids)
    print(f"[INFO] Total video IDs discovered: {n}")

    # Standard FF++ split: first 720 train, next 140 val, last 140 test
    train_ids = video_ids[:TRAIN_COUNT]
    val_ids = video_ids[TRAIN_COUNT:TRAIN_COUNT + VAL_COUNT]
    test_ids = video_ids[TRAIN_COUNT + VAL_COUNT:TRAIN_COUNT + VAL_COUNT + TEST_COUNT]

    # If fewer than 1000 videos, adjust proportionally
    if n < 1000:
        ratio_train = TRAIN_COUNT / 1000
        ratio_val = VAL_COUNT / 1000
        n_train = int(n * ratio_train)
        n_val = int(n * ratio_val)
        train_ids = video_ids[:n_train]
        val_ids = video_ids[n_train:n_train + n_val]
        test_ids = video_ids[n_train + n_val:]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    print(f"  Train: {len(train_ids)}  |  Val: {len(val_ids)}  |  Test: {len(test_ids)}")
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val/test split JSON for FaceForensics++.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/prepare_ffpp_splits.py \\
      --data_root /content/drive/MyDrive/FFPP_raw \\
      --output data/faceforensics/splits.json
        """,
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root of the FF++ download (e.g., FFPP_raw/).")
    parser.add_argument("--output", type=str, default="data/faceforensics/splits.json",
                        help="Output path for the splits JSON file.")
    args = parser.parse_args()

    video_ids = discover_video_ids(args.data_root)
    splits = create_splits(video_ids)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n[DONE] Splits saved to: {args.output}")


if __name__ == "__main__":
    main()
