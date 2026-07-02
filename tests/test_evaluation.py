"""Tests for Milestone 11: metrics, tuning-result reconstruction, and one-time
held-out test-set evaluation for both tasks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

from ml.config import (
    CLASSIFICATION_TUNING_RESULTS_PATH,
    REGRESSION_TUNING_RESULTS_PATH,
)
from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.evaluation.evaluate_classifier import (
    build_classification_test_leaderboard,
    build_tuned_classification_pipeline,
    evaluate_classifier_on_test_set,
    plot_confusion_matrix,
    plot_roc_curve,
    run_classification_evaluation,
)
from ml.evaluation.evaluate_regressor import (
    build_regression_test_leaderboard,
    build_tuned_regression_pipeline,
    evaluate_regressor_on_test_set,
    plot_residuals,
    run_regression_evaluation,
)
from ml.evaluation.leaderboard import (
    build_test_leaderboard,
    evaluate_naive_model,
    load_leaderboard,
    save_leaderboard,
    sort_leaderboard,
)
from ml.evaluation.metrics import classification_metrics, regression_metrics, save_metrics
from ml.pipelines.classification_pipeline import build_classification_pipeline
from ml.pipelines.regression_pipeline import build_regression_pipeline
from ml.training.search import build_best_pipeline_from_results
from ml.training.tuning_spaces import CLASSIFICATION_MODELS, REGRESSION_MODELS


@pytest.fixture(scope="module")
def train_test_split_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered = engineer_features(clean_data(load_raw_data()))
    return grouped_stratified_split(engineered)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_classification_metrics_on_perfect_predictions() -> None:
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_regression_metrics_on_perfect_predictions() -> None:
    y_true = np.array([10.0, 12.0, 15.0])
    y_pred = np.array([10.0, 12.0, 15.0])

    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_regression_metrics_matches_known_error() -> None:
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(2.0)


def test_save_metrics_round_trip(tmp_path: Path) -> None:
    metrics = {"accuracy": 0.9, "f1": 0.85}
    path = tmp_path / "metrics.json"
    save_metrics(metrics, path)
    assert json.loads(path.read_text()) == metrics


# --------------------------------------------------------------------------- #
# Reconstructing the best pipeline from a tuning-results summary
# --------------------------------------------------------------------------- #
def test_build_best_pipeline_from_results_selects_winning_model() -> None:
    fake_models = {
        "logistic_regression": {"estimator": LogisticRegression(max_iter=1000)},
        "ridge_like": {"estimator": Ridge()},
    }
    fake_results = {
        "logistic_regression": {"best_score": 0.5, "best_params": {"model__C": 2.0}},
        "ridge_like": {"best_score": 0.9, "best_params": {"model__alpha": 3.0}},
        "best_model": "ridge_like",
    }

    pipeline = build_best_pipeline_from_results(
        fake_results, fake_models, build_classification_pipeline
    )
    assert isinstance(pipeline.named_steps["model"], Ridge)
    assert pipeline.named_steps["model"].alpha == 3.0


def test_build_best_pipeline_from_results_clones_estimator_instance() -> None:
    """Building two pipelines must not share the same estimator object."""
    fake_models = {"logistic_regression": {"estimator": LogisticRegression(max_iter=1000)}}
    fake_results = {
        "logistic_regression": {"best_score": 0.5, "best_params": {"model__C": 5.0}},
        "best_model": "logistic_regression",
    }

    pipeline_a = build_best_pipeline_from_results(
        fake_results, fake_models, build_classification_pipeline
    )
    pipeline_b = build_best_pipeline_from_results(
        fake_results, fake_models, build_classification_pipeline
    )
    assert pipeline_a.named_steps["model"] is not pipeline_b.named_steps["model"]


# --------------------------------------------------------------------------- #
# Plot smoke tests
# --------------------------------------------------------------------------- #
def test_plot_confusion_matrix_creates_png(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    output_path = plot_confusion_matrix(y_true, y_pred, tmp_path / "cm.png")
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_roc_curve_creates_png(tmp_path: Path) -> None:
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
    output_path = plot_roc_curve(y_true, y_proba, tmp_path / "roc.png")
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_residuals_creates_png(tmp_path: Path) -> None:
    y_true = np.array([10.0, 12.0, 15.0, 20.0])
    y_pred = np.array([11.0, 11.5, 16.0, 18.0])
    output_path = plot_residuals(y_true, y_pred, tmp_path / "residuals.png")
    assert output_path.exists() and output_path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# End-to-end evaluation on the real dataset (using a fast baseline pipeline,
# not the (slower) fully-tuned one, to keep this test quick). These write
# their figures to the project's real report/figures directory -- the same
# pattern used by the Milestone 2 EDA plot tests -- since the filenames
# (confusion_matrix.png, roc_curve.png, residuals.png) don't collide with
# any EDA figure.
# --------------------------------------------------------------------------- #
def test_evaluate_classifier_on_test_set_real_data(
    train_test_split_df: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df = train_test_split_df
    X_train, y_train = select_classification_data(train_df)
    X_test, y_test = select_classification_data(test_df)

    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_classifier_on_test_set(pipeline, X_test, y_test)

    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_regressor_on_test_set_real_data(
    train_test_split_df: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df = train_test_split_df
    X_train, y_train = select_regression_data(train_df)
    X_test, y_test = select_regression_data(test_df)

    pipeline = build_regression_pipeline()
    pipeline.fit(X_train, y_train)

    metrics = evaluate_regressor_on_test_set(pipeline, X_test, y_test)

    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert metrics["mae"] > 0.0


# --------------------------------------------------------------------------- #
# Full Milestone 11 flow using the actual tuning-results artifacts produced
# in Milestone 10. Skipped gracefully if those artifacts are not present
# (e.g. a fresh clone where `train_classifier`/`train_regressor` haven't
# been run yet).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_build_tuned_classification_pipeline_from_real_artifacts() -> None:
    pipeline = build_tuned_classification_pipeline()
    assert pipeline.named_steps["model"] is not None


@pytest.mark.skipif(
    not REGRESSION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_build_tuned_regression_pipeline_from_real_artifacts() -> None:
    pipeline = build_tuned_regression_pipeline()
    assert pipeline.named_steps["model"] is not None


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_run_classification_evaluation_end_to_end() -> None:
    _, metrics = run_classification_evaluation()
    assert 0.0 <= metrics["roc_auc"] <= 1.0


@pytest.mark.skipif(
    not REGRESSION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_run_regression_evaluation_end_to_end() -> None:
    _, metrics = run_regression_evaluation()
    assert metrics["mae"] > 0.0


# --------------------------------------------------------------------------- #
# Full model-ladder leaderboard on the held-out test set (incl. the "dummy"
# baseline), distinct from the CV-only tuning-results comparison above.
# --------------------------------------------------------------------------- #
def test_build_test_leaderboard_scores_every_candidate() -> None:
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"numeric__x": rng.normal(size=200)})
    y_train = pd.Series(rng.integers(0, 2, size=200))
    X_test = pd.DataFrame({"numeric__x": rng.normal(size=50)})
    y_test = pd.Series(rng.integers(0, 2, size=50))

    fake_models = {
        "dummy": {"estimator": LogisticRegression(max_iter=1000)},
        "logistic_regression": {"estimator": LogisticRegression(max_iter=1000)},
    }
    fake_results = {
        "dummy": {"best_score": 0.5, "best_params": {"model__C": 1.0}},
        "logistic_regression": {"best_score": 0.6, "best_params": {"model__C": 2.0}},
        "best_model": "logistic_regression",
    }

    def identity_pipeline_builder(estimator):
        from sklearn.pipeline import Pipeline

        return Pipeline(steps=[("model", estimator)])

    leaderboard = build_test_leaderboard(
        fake_results,
        fake_models,
        identity_pipeline_builder,
        X_train,
        y_train,
        X_test,
        y_test,
        classification_metrics,
        use_predict_proba=True,
    )

    assert len(leaderboard) == 2
    assert {row["model"] for row in leaderboard} == {"dummy", "logistic_regression"}
    for row in leaderboard:
        assert 0.0 <= row["roc_auc"] <= 1.0
    # "best_model" bookkeeping key must never leak into the scored rows.
    assert all("best_model" not in row for row in leaderboard)

    leaderboard_with_train = build_test_leaderboard(
        fake_results,
        fake_models,
        identity_pipeline_builder,
        X_train,
        y_train,
        X_test,
        y_test,
        classification_metrics,
        use_predict_proba=True,
        include_train_metrics=True,
    )
    for row in leaderboard_with_train:
        assert "train_roc_auc" in row
        assert "roc_auc" in row
        assert 0.0 <= row["train_roc_auc"] <= 1.0


def test_evaluate_naive_model_shows_overfitting() -> None:
    """An unconstrained Decision Tree should fit training data almost
    perfectly while generalising far worse -- the overfitting signature this
    naive-model comparison exists to surface."""
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(0)
    X_train = pd.DataFrame({"numeric__x": rng.normal(size=150)})
    y_train = pd.Series(rng.normal(size=150))  # pure noise: no real signal to learn
    X_test = pd.DataFrame({"numeric__x": rng.normal(size=50)})
    y_test = pd.Series(rng.normal(size=50))

    def identity_pipeline_builder(estimator):
        return Pipeline(steps=[("model", estimator)])

    row = evaluate_naive_model(
        "decision_tree_unconstrained",
        DecisionTreeRegressor(random_state=0),
        identity_pipeline_builder,
        X_train,
        y_train,
        X_test,
        y_test,
        regression_metrics,
        use_predict_proba=False,
    )

    assert row["model"] == "decision_tree_unconstrained"
    # An unconstrained tree on pure noise memorises the training set almost
    # exactly (R^2 close to 1) but cannot generalise (test R^2 far lower).
    assert row["train_r2"] > 0.9
    assert row["train_r2"] > row["r2"]


def test_sort_leaderboard_orders_best_first() -> None:
    leaderboard = [
        {"model": "a", "roc_auc": 0.6},
        {"model": "b", "roc_auc": 0.9},
        {"model": "c", "roc_auc": 0.5},
    ]
    sorted_board = sort_leaderboard(leaderboard, "roc_auc", higher_is_better=True)
    assert [row["model"] for row in sorted_board] == ["b", "a", "c"]

    ascending_board = sort_leaderboard(leaderboard, "roc_auc", higher_is_better=False)
    assert [row["model"] for row in ascending_board] == ["c", "a", "b"]


def test_save_and_load_leaderboard_round_trip(tmp_path: Path) -> None:
    leaderboard = [{"model": "dummy", "roc_auc": 0.5}, {"model": "random_forest", "roc_auc": 0.63}]
    path = tmp_path / "leaderboard.json"
    save_leaderboard(leaderboard, path)
    assert load_leaderboard(path) == leaderboard


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_classification_leaderboard_includes_dummy_baseline(
    train_test_split_df: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df = train_test_split_df
    X_train, y_train = select_classification_data(train_df)
    X_test, y_test = select_classification_data(test_df)

    leaderboard = build_classification_test_leaderboard(X_train, y_train, X_test, y_test)

    # One extra row beyond the tuned ladder: decision_tree_unconstrained.
    assert len(leaderboard) == len(CLASSIFICATION_MODELS) + 1
    model_names = {row["model"] for row in leaderboard}
    assert "dummy" in model_names
    assert "decision_tree_unconstrained" in model_names
    assert {"logistic_regression", "decision_tree", "random_forest", "xgboost"} <= model_names
    for row in leaderboard:
        assert 0.0 <= row["roc_auc"] <= 1.0
        assert "train_roc_auc" in row
    # Leaderboard is sorted best-first by ROC-AUC.
    scores = [row["roc_auc"] for row in leaderboard]
    assert scores == sorted(scores, reverse=True)

    by_name = {row["model"]: row for row in leaderboard}
    # The overfitting signature this comparison exists to demonstrate: the
    # unconstrained tree fits training data far better than it predicts on
    # test, unlike the depth-tuned decision_tree entry.
    unconstrained = by_name["decision_tree_unconstrained"]
    tuned_tree = by_name["decision_tree"]
    assert unconstrained["train_accuracy"] > unconstrained["accuracy"]
    unconstrained_gap = unconstrained["train_accuracy"] - unconstrained["accuracy"]
    tuned_gap = tuned_tree["train_accuracy"] - tuned_tree["accuracy"]
    assert unconstrained_gap > tuned_gap


@pytest.mark.skipif(
    not REGRESSION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_regression_leaderboard_includes_dummy_baseline(
    train_test_split_df: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, test_df = train_test_split_df
    X_train, y_train = select_regression_data(train_df)
    X_test, y_test = select_regression_data(test_df)

    leaderboard = build_regression_test_leaderboard(X_train, y_train, X_test, y_test)

    # One extra row beyond the tuned ladder: decision_tree_unconstrained.
    assert len(leaderboard) == len(REGRESSION_MODELS) + 1
    model_names = {row["model"] for row in leaderboard}
    assert "dummy" in model_names
    assert "decision_tree_unconstrained" in model_names
    assert {"ridge", "decision_tree", "random_forest", "xgboost"} <= model_names
    for row in leaderboard:
        assert "train_r2" in row
    # Leaderboard is sorted best-first by R^2.
    scores = [row["r2"] for row in leaderboard]
    assert scores == sorted(scores, reverse=True)

    by_name = {row["model"]: row for row in leaderboard}
    # The overfitting signature this comparison exists to demonstrate: the
    # unconstrained tree fits training data far better than it predicts on
    # test, unlike the depth-tuned decision_tree entry.
    unconstrained = by_name["decision_tree_unconstrained"]
    tuned_tree = by_name["decision_tree"]
    # Real feature rows repeat across patients (many share identical
    # categorical bins), so even an unconstrained tree can't reach a
    # synthetic-data-style near-perfect fit -- but it still fits training
    # data far better than it predicts on held-out test data.
    assert unconstrained["train_r2"] > unconstrained["r2"]
    unconstrained_gap = unconstrained["train_r2"] - unconstrained["r2"]
    tuned_gap = tuned_tree["train_r2"] - tuned_tree["r2"]
    assert unconstrained_gap > tuned_gap
