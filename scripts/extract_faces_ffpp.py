#!/usr/bin/env python3
"""
Extract face crops from FaceForensics++ videos.

Iterates over FF++ video files, samples frames at a fixed FPS, detects and
crops the largest face per frame, and saves crops as PNGs with a CSV index.

**Supports resuming:** If the script is interrupted, re-run it with the same
arguments. It will skip videos whose output folders already contain .png files.

Output structure:
    data/ffpp_faces/{split}/{label}/{compression}/{video_id}/{frame_idx}.png

CSV index columns:
    video_id, split, label, compression, manipulation, frame_idx, frame_path

Usage (Colab / Kaggle — run on GPU for faster MTCNN):
    python scripts/extract_faces_ffpp.py \\
        --data_root /content/drive/MyDrive/FFPP_raw \\
        --output_dir /content/drive/MyDrive/ffpp_faces \\
        --splits_json data/faceforensics/splits.json \\
        --compressions c23 c40 \\
        --manipulations Deepfakes FaceSwap \\
        --target_fps 5 --max_frames 50

Author: Simran Chaudhary
"""

import argparse
import csv
import json
import os
import sys
import glob
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.video_utils import sample_frames
from src.utils.face_detection import create_detector, detect_face

import cv2


# ── FF++ directory layout mapping ──
COMPRESSION_MAP = {
    "c0": "raw",
    "raw": "raw",
    "c23": "c23",
    "c40": "c40",
}

MANIPULATION_TYPES = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def find_videos(data_root: str, dataset: str, compression: str) -> list:
    """
    Find all .mp4 video files for a given dataset and compression.

    Args:
        data_root: Root of the FF++ download (e.g., FFPP_raw/).
        dataset: 'original' or manipulation name.
        compression: 'raw', 'c23', or 'c40'.
    """
    comp_dir = COMPRESSION_MAP.get(compression, compression)

    if dataset == "original":
        base = os.path.join(data_root, "original_sequences", "youtube", comp_dir, "videos")
    else:
        base = os.path.join(data_root, "manipulated_sequences", dataset, comp_dir, "videos")

    if not os.path.isdir(base):
        print(f"  [WARN] Directory not found: {base}")
        return []

    videos = sorted(glob.glob(os.path.join(base, "*.mp4")))
    return videos


def get_video_id(video_path: str) -> str:
    """Extract the video ID from the filename (e.g., '000_003' or '000')."""
    return os.path.splitext(os.path.basename(video_path))[0]


def get_video_split(video_id: str, splits: dict) -> str:
    """
    Determine the split (train/val/test) for a video ID.

    For manipulated videos, the ID is like '000_003' — we check the source ID (first part).
    """
    # For manipulated videos: use source video ID (before underscore)
    source_id = video_id.split("_")[0]

    for split_name, ids in splits.items():
        if source_id in ids or video_id in ids:
            return split_name

    return "train"  # default fallback


def video_already_extracted(vid_out_dir: str) -> bool:
    """Check if a video has already been extracted (has .png files)."""
    if not os.path.isdir(vid_out_dir):
        return False
    pngs = glob.glob(os.path.join(vid_out_dir, "*.png"))
    return len(pngs) > 0


def collect_existing_entries(output_dir: str) -> list:
    """
    Walk the output directory and rebuild CSV entries from existing face crops.
    This is used when resuming to regenerate the metadata CSV.
    """
    entries = []
    # Walk: output_dir / split / label / compression / video_id / frame.png
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            for comp in ["c0", "c23", "c40"]:
                base = os.path.join(output_dir, split, label, comp)
                if not os.path.isdir(base):
                    continue
                for video_id in sorted(os.listdir(base)):
                    vid_dir = os.path.join(base, video_id)
                    if not os.path.isdir(vid_dir):
                        continue
                    for png_file in sorted(glob.glob(os.path.join(vid_dir, "*.png"))):
                        fname = os.path.basename(png_file)
                        frame_idx = int(os.path.splitext(fname)[0])
                        rel_path = os.path.relpath(png_file, output_dir)
                        # Determine manipulation type from label
                        manipulation = "original" if label == "real" else "unknown"
                        entries.append([
                            video_id, split, label, comp,
                            manipulation, frame_idx, rel_path,
                        ])
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Extract face crops from FaceForensics++ videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root of FF++ download (e.g., FFPP_raw/).")
    parser.add_argument("--output_dir", type=str, default="data/ffpp_faces",
                        help="Output directory for face crops.")
    parser.add_argument("--splits_json", type=str, default=None,
                        help="Path to splits.json from prepare_ffpp_splits.py.")
    parser.add_argument("--compressions", nargs="+", default=["c23", "c40"],
                        choices=["c0", "raw", "c23", "c40"],
                        help="Compression levels to process.")
    parser.add_argument("--manipulations", nargs="+", default=["Deepfakes", "FaceSwap"],
                        choices=MANIPULATION_TYPES,
                        help="Manipulation types to process.")
    parser.add_argument("--target_fps", type=float, default=5.0,
                        help="Sampling rate (frames/second).")
    parser.add_argument("--max_frames", type=int, default=50,
                        help="Max frames per video.")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Max videos per category (for quick testing).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for MTCNN: 'cpu' or 'cuda'.")
    args = parser.parse_args()

    # Load splits
    splits = {"train": [], "val": [], "test": []}
    if args.splits_json and os.path.exists(args.splits_json):
        with open(args.splits_json) as f:
            splits = json.load(f)
        print(f"[INFO] Loaded splits from {args.splits_json}")
    else:
        print("[INFO] No splits JSON provided — generating default splits")
        all_ids = [str(i).zfill(3) for i in range(1000)]
        splits = {
            "train": all_ids[:720],
            "val": all_ids[720:860],
            "test": all_ids[860:],
        }

    # Initialize face detector
    print(f"[INFO] Initializing MTCNN on {args.device}...")
    detector = create_detector(device=args.device)

    # Ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    total_faces = 0
    total_skipped = 0
    total_resumed = 0

    # Collect all CSV rows (both existing + new)
    all_csv_rows = []

    # Process each compression × dataset combination
    datasets_to_process = [("original", "real")] + \
                          [(m, "fake") for m in args.manipulations]

    for compression in args.compressions:
        comp_label = compression if compression != "raw" else "c0"

        for dataset_name, label in datasets_to_process:
            print(f"\n{'='*60}")
            print(f"Processing: {dataset_name} / {comp_label}")
            print(f"{'='*60}")

            videos = find_videos(args.data_root, dataset_name, compression)
            if args.max_videos:
                videos = videos[:args.max_videos]

            print(f"  Found {len(videos)} videos")

            manipulation = dataset_name if dataset_name != "original" else "original"
            label_str = "real" if label == "real" else "fake"

            for video_path in tqdm(videos, desc=f"{dataset_name}/{comp_label}"):
                video_id = get_video_id(video_path)
                split = get_video_split(video_id, splits)

                # Output directory for this video
                vid_out_dir = os.path.join(
                    args.output_dir, split, label_str, comp_label, video_id
                )

                # ── RESUME CHECK ──
                if video_already_extracted(vid_out_dir):
                    # Collect existing entries for CSV
                    existing_pngs = sorted(glob.glob(os.path.join(vid_out_dir, "*.png")))
                    for png_path in existing_pngs:
                        fname = os.path.basename(png_path)
                        frame_idx = int(os.path.splitext(fname)[0])
                        rel_path = os.path.relpath(png_path, args.output_dir)
                        all_csv_rows.append([
                            video_id, split, label_str, comp_label,
                            manipulation, frame_idx, rel_path,
                        ])
                        total_faces += 1
                    total_resumed += 1
                    continue

                # ── Extract faces ──
                try:
                    frames = sample_frames(video_path,
                                           target_fps=args.target_fps,
                                           max_frames=args.max_frames)
                except Exception as e:
                    print(f"  [ERROR] Failed to read {video_path}: {e}")
                    continue

                os.makedirs(vid_out_dir, exist_ok=True)

                for frame_idx, frame_bgr in frames:
                    face = detect_face(frame_bgr, detector)
                    if face is None:
                        total_skipped += 1
                        continue

                    # Save face crop as PNG
                    fname = f"{frame_idx:06d}.png"
                    fpath = os.path.join(vid_out_dir, fname)
                    # Convert RGB → BGR for OpenCV saving
                    cv2.imwrite(fpath, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))

                    # Relative path for CSV
                    rel_path = os.path.relpath(fpath, args.output_dir)
                    all_csv_rows.append([
                        video_id, split, label_str, comp_label,
                        manipulation, frame_idx, rel_path,
                    ])
                    total_faces += 1

    # Write CSV index (fresh write — includes both resumed + new entries)
    csv_path = os.path.join(args.output_dir, "metadata.csv")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["video_id", "split", "label", "compression",
                          "manipulation", "frame_idx", "frame_path"])
        writer.writerows(all_csv_rows)

    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Total face crops saved : {total_faces}")
    print(f"  Total frames skipped   : {total_skipped}")
    print(f"  Videos resumed (skip)  : {total_resumed}")
    print(f"  CSV index              : {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
