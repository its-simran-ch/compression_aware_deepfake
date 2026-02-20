#!/usr/bin/env python3
"""
Evaluate a trained model on each FF++ compression level separately.

Loads a checkpoint and evaluates on c0, c23, c40 test splits independently.
Saves per-compression metrics to CSV.

Usage:
    python src/training/evaluate_compression_levels.py \\
        --checkpoint results/checkpoints/best_hybrid_c23_c40.pth \\
        --metadata_csv data/ffpp_faces/metadata.csv \\
        --data_root data/ffpp_faces \\
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

from src.datasets.ffpp_dataset import FFPPFrameDataset
from src.models.fusion_classifier import HybridDeepfakeClassifier
from src.utils.metrics import compute_metrics, get_classification_report


def evaluate_on_compression(model, metadata_csv, data_root, compression, mode, device,
                            batch_size=16, num_workers=2):
    """Evaluate model on a single compression level."""
    include_dwt = mode in ("frequency", "hybrid")

    ds = FFPPFrameDataset(
        metadata_csv=metadata_csv,
        root_dir=data_root,
        split="test",
        compressions=[compression],
        include_dwt=include_dwt,
    )

    if len(ds) == 0:
        print(f"  [WARN] No test samples for {compression}")
        return None

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Eval {compression}", leave=False):
            labels = batch["label"].float().to(device)
            rgb = batch["rgb"].to(device) if mode != "frequency" else None
            dwt = batch["dwt"].to(device) if mode != "spatial" else None

            logits = model(rgb_input=rgb, dwt_input=dwt)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    metrics = compute_metrics(np.array(all_labels), preds, np.array(all_probs))

    report = get_classification_report(np.array(all_labels), preds)
    print(f"\n  {compression} — Acc: {metrics['accuracy']:.4f}  "
          f"F1: {metrics['f1']:.4f}  AUC: {metrics['auc']:.4f}")
    print(report)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on each FF++ compression level.",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--metadata_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["spatial", "frequency", "hybrid"])
    parser.add_argument("--compressions", nargs="+", default=["c0", "c23", "c40"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_csv", type=str, default="results/csv/compression_eval.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = HybridDeepfakeClassifier(mode=args.mode, pretrained_spatial=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Trained epoch: {ckpt.get('epoch', '?')}  |  Val AUC: {ckpt.get('val_auc', '?')}")

    # Evaluate each compression
    results = []
    for comp in args.compressions:
        print(f"\n{'='*40}")
        print(f"Evaluating: {comp}")
        metrics = evaluate_on_compression(
            model, args.metadata_csv, args.data_root, comp, args.mode,
            device, args.batch_size,
        )
        if metrics:
            metrics["compression"] = comp
            results.append(metrics)

    # Save CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    if results:
        with open(args.output_csv, "w", newline="") as f:
            keys = ["compression", "accuracy", "precision", "recall", "f1", "auc"]
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
