"""Honest, one-time held-out test-set evaluation for the classification task.

Reconstructs the best pipeline selected by CV in Milestone 10, refits it on
the *full* training set, and evaluates it on the test set **exactly once**.
These numbers are never fed back into further tuning -- if you need to
change the model, go back to Milestone 10, retune, and only re-run this
evaluation as the final step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.config import (
    CLASSIFICATION_TUNING_RESULTS_PATH,
    CONFUSION_MATRIX_FIGURE_PATH,
    RANDOM_STATE,
    ROC_CURVE_FIGURE_PATH,
    TEST_LEADERBOARD_CLASSIFICATION_PATH,
    TEST_METRICS_CLASSIFICATION_PATH,
)
from ml.evaluation.leaderboard import (
    build_test_leaderboard,
    evaluate_naive_model,
    save_leaderboard,
    sort_leaderboard,
)
from ml.evaluation.metrics import classification_metrics, save_metrics
from ml.pipelines.classification_pipeline import build_classification_pipeline
from ml.training.search import build_best_pipeline_from_results, load_tuning_results
from ml.training.tuning_spaces import CLASSIFICATION_MODELS

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path = CONFUSION_MATRIX_FIGURE_PATH
) -> Path:
    """Save a confusion-matrix heatmap for the test-set predictions."""
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Short", "Long"],
        yticklabels=["Short", "Long"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Held-Out Test Set)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", output_path)
    return output_path


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, output_path: Path = ROC_CURVE_FIGURE_PATH
) -> Path:
    """Save the ROC curve for the test-set predicted probabilities."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Held-Out Test Set)")
    ax.legend(loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curve to %s", output_path)
    return output_path


def build_tuned_classification_pipeline(
    tuning_results_path: Path = CLASSIFICATION_TUNING_RESULTS_PATH,
) -> Pipeline:
    """Reconstruct the best classification pipeline selected during Milestone 10."""
    results = load_tuning_results(tuning_results_path)
    return build_best_pipeline_from_results(
        results, CLASSIFICATION_MODELS, build_classification_pipeline
    )


def evaluate_classifier_on_test_set(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute test-set metrics and save the confusion matrix / ROC curve figures.

    Args:
        pipeline: A pipeline already fitted on the training set.
        X_test: Held-out test predictors.
        y_test: Held-out test target.

    Returns:
        Dict of test-set metrics (accuracy, precision, recall, f1, roc_auc).
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    return metrics


def build_classification_test_leaderboard(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tuning_results_path: Path = CLASSIFICATION_TUNING_RESULTS_PATH,
) -> list[dict[str, float]]:
    """Refit every candidate in ``CLASSIFICATION_MODELS`` (incl. ``dummy``) on
    the full training set and score all of them on the held-out test set,
    sorted best-first by ROC-AUC. Each row also includes ``train_<metric>``
    values, and one extra row -- ``decision_tree_unconstrained``, a
    ``DecisionTreeClassifier`` with its *default* (unlimited) depth -- is
    appended to make the overfitting of an untuned tree visible: expect its
    train metrics near-perfect and its test metrics well below the tuned
    ``decision_tree`` entry.
    """
    tuning_results = load_tuning_results(tuning_results_path)
    leaderboard = build_test_leaderboard(
        tuning_results,
        CLASSIFICATION_MODELS,
        build_classification_pipeline,
        X_train,
        y_train,
        X_test,
        y_test,
        classification_metrics,
        use_predict_proba=True,
        include_train_metrics=True,
    )
    leaderboard.append(
        evaluate_naive_model(
            "decision_tree_unconstrained",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            build_classification_pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            classification_metrics,
            use_predict_proba=True,
        )
    )
    return sort_leaderboard(leaderboard, "roc_auc", higher_is_better=True)


def run_classification_evaluation() -> tuple[Pipeline, dict[str, float]]:
    """Fit the tuned classification pipeline on the full training set and
    evaluate it once on the held-out test set. Also builds and saves the
    full model-ladder leaderboard (incl. the ``dummy`` baseline) on the
    same held-out test set.

    Returns:
        ``(fitted_pipeline, test_metrics)`` for the single best model.
    """
    from ml.data.feature_selection import select_classification_data
    from ml.data.split import get_train_test_split

    train_df, test_df = get_train_test_split()
    X_train, y_train = select_classification_data(train_df)
    X_test, y_test = select_classification_data(test_df)

    pipeline = build_tuned_classification_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_classifier_on_test_set(pipeline, X_test, y_test)
    save_metrics(metrics, TEST_METRICS_CLASSIFICATION_PATH)
    logger.info("Held-out classification test metrics: %s", metrics)

    leaderboard = build_classification_test_leaderboard(X_train, y_train, X_test, y_test)
    save_leaderboard(leaderboard, TEST_LEADERBOARD_CLASSIFICATION_PATH)

    return pipeline, metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, test_metrics = run_classification_evaluation()
    print("Held-out test metrics (classification):")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")
