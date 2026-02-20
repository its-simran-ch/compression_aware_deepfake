"""
Evaluation metrics for deepfake detection.

Provides functions for:
- Frame-level metrics: accuracy, precision, recall, F1, AUC
- Video-level aggregation: average frame probabilities → video prediction
- Metric logging and CSV export

Author: Simran Chaudhary
"""

import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities for class 1 (for AUC).

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally auc.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0
        metrics["ap"] = average_precision_score(y_true, y_prob)
    else:
        metrics["auc"] = 0.0
        metrics["ap"] = 0.0

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return the 2×2 confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a formatted classification report string."""
    return classification_report(
        y_true, y_pred,
        target_names=["Real", "Fake"],
        zero_division=0,
    )


def aggregate_frame_to_video(
    frame_probs: List[float],
    threshold_fake: float = 0.55,
    threshold_real: float = 0.45,
) -> Dict:
    """
    Aggregate frame-level probabilities into a video-level prediction.

    Args:
        frame_probs: List of per-frame fake probabilities.
        threshold_fake: If mean prob >= this, predict FAKE.
        threshold_real: If mean prob <= this, predict REAL.

    Returns:
        Dict with label, score, and num_frames_used.
    """
    if not frame_probs:
        return {
            "label": "UNCERTAIN",
            "score": 0.5,
            "num_frames_used": 0,
        }

    mean_prob = float(np.mean(frame_probs))

    if mean_prob >= threshold_fake:
        label = "FAKE"
    elif mean_prob <= threshold_real:
        label = "REAL"
    else:
        label = "UNCERTAIN"

    return {
        "label": label,
        "score": mean_prob,
        "num_frames_used": len(frame_probs),
    }


def format_metrics_table(results: List[Dict]) -> str:
    """
    Format a list of metric dicts into a readable table string.

    Each dict should have keys like: experiment, accuracy, f1, auc, etc.
    """
    if not results:
        return "No results to display."

    keys = list(results[0].keys())
    header = " | ".join(f"{k:>12}" for k in keys)
    sep = "-" * len(header)
    rows = [header, sep]

    for r in results:
        row = " | ".join(f"{r.get(k, ''):>12}" if isinstance(r.get(k), str)
                         else f"{r.get(k, 0):>12.4f}" for k in keys)
        rows.append(row)

    return "\n".join(rows)
