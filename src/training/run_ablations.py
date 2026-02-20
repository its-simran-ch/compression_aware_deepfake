#!/usr/bin/env python3
"""
Run ablation experiments: Spatial-only vs Frequency-only vs Hybrid.

Trains each variant and evaluates on all compression levels.
Produces a consolidated results CSV.

Usage (Colab / Kaggle):
    python src/training/run_ablations.py \\
        --metadata_csv /content/drive/MyDrive/ffpp_faces/metadata.csv \\
        --data_root /content/drive/MyDrive/ffpp_faces \\
        --epochs 10

Author: Simran Chaudhary
"""

import argparse
import csv
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


ABLATION_CONFIGS = [
    {"mode": "spatial",   "name": "Spatial-Only (EfficientNet-B0)"},
    {"mode": "frequency", "name": "Frequency-Only (DWT CNN)"},
    {"mode": "hybrid",    "name": "Hybrid (Spatial + Frequency)"},
]

COMPRESSION_COMBOS = [
    {"compressions": ["c0"],              "tag": "c0_only"},
    {"compressions": ["c23"],             "tag": "c23_only"},
    {"compressions": ["c40"],             "tag": "c40_only"},
    {"compressions": ["c0", "c23", "c40"], "tag": "c0_c23_c40"},
]


def run_training(metadata_csv, data_root, mode, compressions, epochs,
                 batch_size, output_dir, experiment_name):
    """Run a single training experiment via subprocess."""
    cmd = [
        sys.executable, "src/training/train_ffpp.py",
        "--metadata_csv", metadata_csv,
        "--data_root", data_root,
        "--mode", mode,
        "--compressions", *compressions,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--output_dir", output_dir,
        "--experiment_name", experiment_name,
    ]
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def run_evaluation(checkpoint, metadata_csv, data_root, mode, compressions, output_csv):
    """Run per-compression evaluation via subprocess."""
    cmd = [
        sys.executable, "src/training/evaluate_compression_levels.py",
        "--checkpoint", checkpoint,
        "--metadata_csv", metadata_csv,
        "--data_root", data_root,
        "--mode", mode,
        "--compressions", *compressions,
        "--output_csv", output_csv,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments (spatial vs frequency vs hybrid).",
    )
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs per ablation run.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training (only evaluate existing checkpoints).")
    parser.add_argument("--mode_only", type=str, default=None,
                        choices=["spatial", "frequency", "hybrid"],
                        help="Run only a specific mode ablation.")
    parser.add_argument("--compression_combo", type=str, default=None,
                        help="Run only a specific compression combo tag.")
    args = parser.parse_args()

    results_all = []

    configs = ABLATION_CONFIGS
    if args.mode_only:
        configs = [c for c in configs if c["mode"] == args.mode_only]

    combos = COMPRESSION_COMBOS
    if args.compression_combo:
        combos = [c for c in combos if c["tag"] == args.compression_combo]

    for config in configs:
        mode = config["mode"]

        for combo in combos:
            compressions = combo["compressions"]
            tag = combo["tag"]
            experiment_name = f"{mode}_{tag}"
            checkpoint = os.path.join(
                args.output_dir, "checkpoints", f"best_{experiment_name}.pth"
            )

            # Train
            if not args.skip_training:
                run_training(
                    args.metadata_csv, args.data_root, mode, compressions,
                    args.epochs, args.batch_size, args.output_dir, experiment_name,
                )

            # Evaluate on all compression levels
            if os.path.exists(checkpoint):
                eval_csv = os.path.join(
                    args.output_dir, "csv", f"eval_{experiment_name}.csv"
                )
                run_evaluation(
                    checkpoint, args.metadata_csv, args.data_root, mode,
                    ["c0", "c23", "c40"], eval_csv,
                )

                # Collect results
                if os.path.exists(eval_csv):
                    with open(eval_csv) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            row["mode"] = mode
                            row["train_compressions"] = tag
                            row["experiment"] = experiment_name
                            results_all.append(row)
            else:
                print(f"  [WARN] Checkpoint not found: {checkpoint}")

    # Save consolidated ablation results
    summary_csv = os.path.join(args.output_dir, "csv", "ablation_summary.csv")
    if results_all:
        os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
        keys = ["experiment", "mode", "train_compressions", "compression",
                "accuracy", "precision", "recall", "f1", "auc"]
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in results_all:
                writer.writerow(r)
        print(f"\n{'='*60}")
        print(f"Ablation summary saved to: {summary_csv}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
