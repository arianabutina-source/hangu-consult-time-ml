"""Honest, one-time held-out test-set evaluation for the regression task.

Mirrors ``ml.evaluation.evaluate_classifier``. Reconstructs the best pipeline
selected by CV in Milestone 10, refits it on the *full* training set, and
evaluates it on the test set exactly once.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

from ml.config import (
    REGRESSION_TUNING_RESULTS_PATH,
    RESIDUALS_FIGURE_PATH,
    TEST_METRICS_REGRESSION_PATH,
)
from ml.evaluation.metrics import regression_metrics, save_metrics
from ml.pipelines.regression_pipeline import build_regression_pipeline
from ml.training.search import build_best_pipeline_from_results, load_tuning_results
from ml.training.tuning_spaces import REGRESSION_MODELS

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path = RESIDUALS_FIGURE_PATH
) -> Path:
    """Save a residuals-vs-predicted scatter plot for the test-set predictions."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, edgecolor="none")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set_xlabel("Predicted duration (minutes)")
    ax.set_ylabel("Residual (actual - predicted, minutes)")
    ax.set_title("Residuals vs. Predicted Duration (Held-Out Test Set)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved residual plot to %s", output_path)
    return output_path


def build_tuned_regression_pipeline(
    tuning_results_path: Path = REGRESSION_TUNING_RESULTS_PATH,
) -> Pipeline:
    """Reconstruct the best regression pipeline selected during Milestone 10."""
    results = load_tuning_results(tuning_results_path)
    return build_best_pipeline_from_results(results, REGRESSION_MODELS, build_regression_pipeline)


def evaluate_regressor_on_test_set(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute test-set metrics and save the residual plot.

    Args:
        pipeline: A pipeline already fitted on the training set.
        X_test: Held-out test predictors.
        y_test: Held-out test target (minutes).

    Returns:
        Dict of test-set metrics (mae, rmse, r2).
    """
    y_pred = pipeline.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    return metrics


def run_regression_evaluation() -> tuple[Pipeline, dict[str, float]]:
    """Fit the tuned regression pipeline on the full training set and
    evaluate it once on the held-out test set.

    Returns:
        ``(fitted_pipeline, test_metrics)``.
    """
    from ml.data.feature_selection import select_regression_data
    from ml.data.split import get_train_test_split

    train_df, test_df = get_train_test_split()
    X_train, y_train = select_regression_data(train_df)
    X_test, y_test = select_regression_data(test_df)

    pipeline = build_tuned_regression_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_regressor_on_test_set(pipeline, X_test, y_test)
    save_metrics(metrics, TEST_METRICS_REGRESSION_PATH)
    logger.info("Held-out regression test metrics: %s", metrics)
    return pipeline, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, test_metrics = run_regression_evaluation()
    print("Held-out test metrics (regression):")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
