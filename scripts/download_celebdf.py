#!/usr/bin/env python3
"""
Download Celeb-DF (v2) dataset.

Since Celeb-DF v2 is typically distributed as a zip file via Google Drive,
this script helps with extraction and verification.

The user already has the zip downloaded. This script extracts it and
organizes it into the expected structure.

Usage (Colab):
    python scripts/download_celebdf.py \\
        --zip_path /content/drive/MyDrive/Celeb-DF-v2.zip \\
        --output_dir /content/drive/MyDrive/Celeb-DF-v2

Local (Mac):
    python scripts/download_celebdf.py \\
        --zip_path ~/Downloads/Celeb-DF-v2.zip \\
        --output_dir data/celeb_df

Author: Simran Chaudhary
"""

import argparse
import os
import zipfile
import glob


def extract_zip(zip_path: str, output_dir: str):
    """Extract the Celeb-DF zip file."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Zip file not found at: {zip_path}\n"
            "Please provide the correct path to your Celeb-DF-v2.zip file."
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Extracting {zip_path} -> {output_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    print("[INFO] Extraction complete.")


def verify_celebdf(output_dir: str):
    """Verify Celeb-DF directory structure and print statistics."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Celeb-DF v2 Structure")
    print("=" * 60)

    # Expected subfolders
    expected = {
        "Celeb-real": "Real celebrity videos",
        "Celeb-synthesis": "Synthesized (deepfake) celebrity videos",
        "YouTube-real": "Real YouTube videos",
    }

    total_videos = 0
    for folder_name, description in expected.items():
        # Search both in output_dir directly and one level down
        folder = os.path.join(output_dir, folder_name)
        if not os.path.isdir(folder):
            # Try searching one level deeper (in case zip has a wrapper folder)
            matches = glob.glob(os.path.join(output_dir, "*", folder_name))
            if matches:
                folder = matches[0]

        if os.path.isdir(folder):
            videos = [f for f in os.listdir(folder)
                       if f.endswith((".mp4", ".avi", ".mkv"))]
            print(f"  ✓ {folder_name}: {len(videos)} videos — {description}")
            total_videos += len(videos)
        else:
            print(f"  ✗ {folder_name}: NOT FOUND — {description}")

    # Check for labels list
    list_file = None
    for pattern in ["List_of_testing_videos.txt", "**/List_of_testing_videos.txt"]:
        matches = glob.glob(os.path.join(output_dir, pattern), recursive=True)
        if matches:
            list_file = matches[0]
            break

    if list_file:
        with open(list_file) as f:
            lines = f.readlines()
        print(f"\n  ✓ Test list found: {len(lines)} entries")
    else:
        print("\n  ⚠ List_of_testing_videos.txt not found (may be nested)")

    print(f"\n  Total videos found: {total_videos}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and verify Celeb-DF v2 dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from zip (Colab):
  python scripts/download_celebdf.py \\
      --zip_path /content/drive/MyDrive/Celeb-DF-v2.zip \\
      --output_dir /content/drive/MyDrive/Celeb-DF-v2

  # Verify only (already extracted):
  python scripts/download_celebdf.py \\
      --output_dir data/celeb_df --verify_only
        """,
    )
    parser.add_argument("--zip_path", type=str, default=None,
                        help="Path to the Celeb-DF-v2.zip file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to extract into / verify.")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only verify existing extraction.")
    args = parser.parse_args()

    if not args.verify_only:
        if args.zip_path is None:
            parser.error("--zip_path is required when not using --verify_only")
        extract_zip(args.zip_path, args.output_dir)

    verify_celebdf(args.output_dir)


if __name__ == "__main__":
    main()
