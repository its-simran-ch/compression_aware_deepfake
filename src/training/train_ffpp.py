#!/usr/bin/env python3
"""
Training script for Compression-Aware Deepfake Detection on FaceForensics++.

Supports three modes:
- spatial:   EfficientNet-B0 only
- frequency: DWT CNN only
- hybrid:    Both branches fused (default)

Designed for free Colab / Kaggle GPUs (T4, P100).

Usage:
    # Hybrid model, c23+c40, default settings (Colab):
    python src/training/train_ffpp.py \\
        --metadata_csv /content/drive/MyDrive/ffpp_faces/metadata.csv \\
        --data_root /content/drive/MyDrive/ffpp_faces \\
        --mode hybrid --compressions c23 c40 --epochs 15

    # Spatial only, all compressions:
    python src/training/train_ffpp.py \\
        --metadata_csv data/ffpp_faces/metadata.csv \\
        --data_root data/ffpp_faces \\
        --mode spatial --compressions c0 c23 c40 --epochs 10

Author: Simran Chaudhary
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.datasets.ffpp_dataset import FFPPFrameDataset
from src.models.fusion_classifier import HybridDeepfakeClassifier
from src.utils.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the compression-aware deepfake detector on FF++.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="Path to metadata.csv from extract_faces_ffpp.py.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of face crops.")
    parser.add_argument("--compressions", nargs="+", default=["c23", "c40"],
                        choices=["c0", "c23", "c40"],
                        help="Compression levels to include in training.")

    # Model
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["spatial", "frequency", "hybrid"],
                        help="Branch mode: spatial, frequency, or hybrid.")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained weights for EfficientNet.")

    # Training
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (16 fits comfortably on free Colab T4).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for AdamW.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for quick testing).")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation samples.")

    # Output
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for logs and checkpoints.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name for this experiment run.")

    return parser.parse_args()


def create_dataloaders(args) -> tuple:
    """Create train and validation DataLoaders."""
    include_dwt = args.mode in ("frequency", "hybrid")

    train_ds = FFPPFrameDataset(
        metadata_csv=args.metadata_csv,
        root_dir=args.data_root,
        split="train",
        compressions=args.compressions,
        include_dwt=include_dwt,
        max_samples=args.max_train_samples,
    )

    val_ds = FFPPFrameDataset(
        metadata_csv=args.metadata_csv,
        root_dir=args.data_root,
        split="val",
        compressions=args.compressions,
        include_dwt=include_dwt,
        max_samples=args.max_val_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device, mode):
    """Run one training epoch."""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="  Train", leave=False):
        labels = batch["label"].float().to(device)

        # Prepare inputs
        rgb = batch["rgb"].to(device) if mode != "frequency" else None
        dwt = batch["dwt"].to(device) if mode != "spatial" else None

        # Forward
        logits = model(rgb_input=rgb, dwt_input=dwt)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels) if all_labels else 0
    preds = (np.array(all_probs) >= 0.5).astype(int)
    metrics = compute_metrics(np.array(all_labels), preds, np.array(all_probs))
    metrics["loss"] = avg_loss
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, mode):
    """Run evaluation."""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []

    for batch in tqdm(loader, desc="  Val", leave=False):
        labels = batch["label"].float().to(device)

        rgb = batch["rgb"].to(device) if mode != "frequency" else None
        dwt = batch["dwt"].to(device) if mode != "spatial" else None

        logits = model(rgb_input=rgb, dwt_input=dwt)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(all_labels) if all_labels else 0
    preds = (np.array(all_probs) >= 0.5).astype(int)
    metrics = compute_metrics(np.array(all_labels), preds, np.array(all_probs))
    metrics["loss"] = avg_loss
    return metrics


def main():
    args = parse_args()

    # ── Experiment name ──
    if args.experiment_name is None:
        comp_str = "_".join(args.compressions)
        args.experiment_name = f"{args.mode}_{comp_str}"

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_name}")
    print(f"Mode: {args.mode}  |  Compressions: {args.compressions}")
    print(f"Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──
    train_loader, val_loader = create_dataloaders(args)

    # ── Model ──
    model = HybridDeepfakeClassifier(
        mode=args.mode,
        pretrained_spatial=args.pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Loss + Optimizer ──
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── LR Scheduler ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Output directories ──
    csv_dir = os.path.join(args.output_dir, "csv")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    log_path = os.path.join(csv_dir, f"train_log_{args.experiment_name}.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "epoch", "train_loss", "train_acc", "train_f1", "train_auc",
        "val_loss", "val_acc", "val_f1", "val_auc", "lr",
    ])

    # ── Training loop ──
    best_val_auc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch}/{args.epochs} (lr={current_lr:.2e})")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.mode
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, args.mode
        )

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0

        # Print
        print(f"  Train — loss: {train_metrics['loss']:.4f}  acc: {train_metrics['accuracy']:.4f}  "
              f"f1: {train_metrics['f1']:.4f}  auc: {train_metrics['auc']:.4f}")
        print(f"  Val   — loss: {val_metrics['loss']:.4f}  acc: {val_metrics['accuracy']:.4f}  "
              f"f1: {val_metrics['f1']:.4f}  auc: {val_metrics['auc']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Log
        log_writer.writerow([
            epoch,
            f"{train_metrics['loss']:.4f}", f"{train_metrics['accuracy']:.4f}",
            f"{train_metrics['f1']:.4f}", f"{train_metrics['auc']:.4f}",
            f"{val_metrics['loss']:.4f}", f"{val_metrics['accuracy']:.4f}",
            f"{val_metrics['f1']:.4f}", f"{val_metrics['auc']:.4f}",
            f"{current_lr:.2e}",
        ])
        log_file.flush()

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            ckpt_path = os.path.join(ckpt_dir, f"best_{args.experiment_name}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "mode": args.mode,
                "compressions": args.compressions,
            }, ckpt_path)
            print(f"  ★ Best model saved (AUC={best_val_auc:.4f})")

    log_file.close()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best epoch: {best_epoch}  |  Best val AUC: {best_val_auc:.4f}")
    print(f"  Log: {log_path}")
    print(f"  Checkpoint: {os.path.join(ckpt_dir, f'best_{args.experiment_name}.pth')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
