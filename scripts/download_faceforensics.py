#!/usr/bin/env python3
"""
Download FaceForensics++ dataset using the official download script.

You MUST have official access approved via the FaceForensics++ request form.
The official download script is: https://kaldir.vc.in.tum.de/faceforensics_download_v4.py

Since you already downloaded the data to Google Drive (FFPP_raw/), this script
is provided for reference and re-downloads if needed.

Usage (Colab):
    # 1. Download the official script
    !wget https://kaldir.vc.in.tum.de/faceforensics_download_v4.py -O download.py

    # 2. Run for each dataset/compression:
    !python download.py /content/drive/MyDrive/FFPP_raw -d original -c raw -t videos --server EU2
    !python download.py /content/drive/MyDrive/FFPP_raw -d original -c c23 -t videos --server EU2
    !python download.py /content/drive/MyDrive/FFPP_raw -d original -c c40 -t videos --server EU2
    !python download.py /content/drive/MyDrive/FFPP_raw -d Deepfakes -c raw -t videos --server EU2
    # ... repeat for Face2Face, FaceSwap, NeuralTextures at c0/c23/c40

Local (Mac) — only if you have storage:
    python scripts/download_faceforensics.py --output_dir data/faceforensics

Author: Simran Chaudhary
"""

import argparse
import os
import subprocess
import sys


OFFICIAL_SCRIPT_URL = "https://kaldir.vc.in.tum.de/faceforensics_download_v4.py"

DATASETS = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
COMPRESSIONS = ["raw", "c23", "c40"]


def download_official_script(dest_path: str = "faceforensics_download_v4.py") -> str:
    """Download the official FF++ download script if not already present."""
    if not os.path.exists(dest_path):
        print(f"[INFO] Downloading official FF++ script to {dest_path} ...")
        subprocess.run(["wget", "-q", OFFICIAL_SCRIPT_URL, "-O", dest_path], check=True)
    else:
        print(f"[INFO] Official script already exists at {dest_path}")
    return dest_path


def run_download(script_path: str, output_dir: str, dataset: str,
                 compression: str, server: str = "EU2"):
    """Run a single download command via the official script."""
    cmd = [
        sys.executable, script_path,
        output_dir,
        "-d", dataset,
        "-c", compression,
        "-t", "videos",
        "--server", server,
    ]
    print(f"\n[DOWNLOAD] {dataset} / {compression}")
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def verify_download(output_dir: str):
    """Quick integrity check: print folder structure and video counts."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Folder structure")
    print("=" * 60)
    for root, dirs, files in os.walk(output_dir):
        depth = root.replace(output_dir, "").count(os.sep)
        indent = "  " * depth
        mp4s = [f for f in files if f.endswith(".mp4")]
        if mp4s or depth <= 3:
            print(f"{indent}{os.path.basename(root)}/  ({len(mp4s)} videos)")


def main():
    parser = argparse.ArgumentParser(
        description="Download FaceForensics++ dataset (reference script).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything (Colab, with Drive mounted):
  python scripts/download_faceforensics.py --output_dir /content/drive/MyDrive/FFPP_raw --all

  # Download a specific subset:
  python scripts/download_faceforensics.py --output_dir data/faceforensics \\
      --datasets original Deepfakes --compressions c23 c40

  # Just verify what's already downloaded:
  python scripts/download_faceforensics.py --output_dir /content/drive/MyDrive/FFPP_raw --verify_only
        """,
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory to store FaceForensics++ data.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS,
                        help="Which datasets to download.")
    parser.add_argument("--compressions", nargs="+", default=COMPRESSIONS,
                        choices=COMPRESSIONS,
                        help="Which compression levels to download.")
    parser.add_argument("--server", type=str, default="EU2",
                        help="Download server (default: EU2).")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets at all compressions.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing download, do not download.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.verify_only:
        verify_download(args.output_dir)
        return

    # Download the official script first
    script_path = download_official_script()

    datasets = DATASETS if args.all else args.datasets
    compressions = COMPRESSIONS if args.all else args.compressions

    for ds in datasets:
        for comp in compressions:
            run_download(script_path, args.output_dir, ds, comp, args.server)

    verify_download(args.output_dir)
    print("\n[DONE] FaceForensics++ download complete!")


if __name__ == "__main__":
    main()
