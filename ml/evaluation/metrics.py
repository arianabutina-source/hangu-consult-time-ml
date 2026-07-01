"""Metric computations shared by the classification and regression evaluators.

Pure functions only: no fitting, no plotting, no I/O -- so they can be unit
tested independently of any model or dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Compute accuracy, precision, recall, F1, and ROC-AUC.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probability of the positive class.

    Returns:
        Dict of metric name to value.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R^2.

    Args:
        y_true: Ground-truth continuous target (minutes).
        y_pred: Predicted continuous target (minutes).

    Returns:
        Dict of metric name to value.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    """Write a metrics dict to disk as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", path)
