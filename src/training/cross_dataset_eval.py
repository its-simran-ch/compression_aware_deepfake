#!/usr/bin/env python3
"""
Cross-dataset evaluation: FF++-trained model evaluated on Celeb-DF v2.

Tests generalization of the trained model on an unseen dataset.

Usage:
    python src/training/cross_dataset_eval.py \\
        --checkpoint results/checkpoints/best_hybrid_c23_c40.pth \\
        --celeb_root data/celeb_df_faces \\
        --mode hybrid

Author: Simran Chaudhary
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.celebdf_dataset import CelebDFFrameDataset
from src.models.fusion_classifier import HybridDeepfakeClassifier
from src.utils.metrics import compute_metrics, get_classification_report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FF++-trained model on Celeb-DF v2.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint.")
    parser.add_argument("--celeb_root", type=str, required=True,
                        help="Root directory of Celeb-DF face crops.")
    parser.add_argument("--celeb_csv", type=str, default=None,
                        help="Optional CSV index for Celeb-DF.")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["spatial", "frequency", "hybrid"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output_csv", type=str,
                        default="results/csv/cross_dataset_celebdf.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridDeepfakeClassifier(mode=args.mode, pretrained_spatial=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load Celeb-DF dataset
    include_dwt = args.mode in ("frequency", "hybrid")
    dataset = CelebDFFrameDataset(
        root_dir=args.celeb_root,
        metadata_csv=args.celeb_csv,
        include_dwt=include_dwt,
        max_samples=args.max_samples,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    # Evaluate
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Cross-dataset eval"):
            labels = batch["label"].float().to(device)
            rgb = batch["rgb"].to(device) if args.mode != "frequency" else None
            dwt = batch["dwt"].to(device) if args.mode != "spatial" else None

            logits = model(rgb_input=rgb, dwt_input=dwt)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    metrics = compute_metrics(np.array(all_labels), preds, np.array(all_probs))

    print(f"\n{'='*50}")
    print(f"Cross-Dataset Results (Celeb-DF v2)")
    print(f"{'='*50}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print()
    print(get_classification_report(np.array(all_labels), preds))

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        keys = ["dataset", "accuracy", "precision", "recall", "f1", "auc"]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        row = {**metrics, "dataset": "Celeb-DF-v2"}
        writer.writerow(row)
    print(f"Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
