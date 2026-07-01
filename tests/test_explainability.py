"""Tests for ml.evaluation.explainability: feature importance and SHAP summaries."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.evaluation.explainability import (
    compute_shap_values,
    get_feature_importance,
    get_feature_names,
    plot_feature_importance,
    plot_shap_summary,
)
from ml.pipelines.classification_pipeline import build_classification_pipeline
from ml.pipelines.regression_pipeline import build_regression_pipeline


@pytest.fixture(scope="module")
def classification_train() -> tuple[pd.DataFrame, pd.Series]:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, _ = grouped_stratified_split(engineered)
    return select_classification_data(train_df)


@pytest.fixture(scope="module")
def regression_train() -> tuple[pd.DataFrame, pd.Series]:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, _ = grouped_stratified_split(engineered)
    return select_regression_data(train_df)


@pytest.fixture(scope="module")
def fitted_rf_classifier(
    classification_train: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    X, y = classification_train
    pipeline = build_classification_pipeline(RandomForestClassifier(random_state=0, n_estimators=50))
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="module")
def fitted_rf_regressor(regression_train: tuple[pd.DataFrame, pd.Series]) -> Pipeline:
    X, y = regression_train
    pipeline = build_regression_pipeline(RandomForestRegressor(random_state=0, n_estimators=50))
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture(scope="module")
def fitted_logistic_regression(
    classification_train: tuple[pd.DataFrame, pd.Series],
) -> Pipeline:
    X, y = classification_train
    pipeline = build_classification_pipeline()  # default: LogisticRegression
    pipeline.fit(X, y)
    return pipeline


def test_get_feature_names_matches_preprocessor(
    fitted_rf_classifier: Pipeline,
) -> None:
    names = get_feature_names(fitted_rf_classifier)
    expected = list(fitted_rf_classifier.named_steps["preprocess"].get_feature_names_out())
    assert names == expected
    assert len(names) > 0


def test_get_feature_importance_tree_based_sums_to_one(
    fitted_rf_classifier: Pipeline,
) -> None:
    importance_df = get_feature_importance(fitted_rf_classifier)
    assert list(importance_df.columns) == ["feature", "importance"]
    assert importance_df["importance"].sum() == pytest.approx(1.0, abs=1e-6)
    assert importance_df["importance"].is_monotonic_decreasing


def test_get_feature_importance_linear_uses_abs_coefficients(
    fitted_logistic_regression: Pipeline,
) -> None:
    importance_df = get_feature_importance(fitted_logistic_regression)
    assert (importance_df["importance"] >= 0).all()
    assert importance_df["importance"].is_monotonic_decreasing


def test_get_feature_importance_falls_back_to_permutation(
    classification_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_train
    pipeline = build_classification_pipeline(KNeighborsClassifier())
    pipeline.fit(X, y)

    importance_df = get_feature_importance(pipeline, X=X, y=y, scoring="accuracy", n_repeats=2)
    # Permutation importance operates on the raw input columns (it shuffles
    # X before the whole pipeline, including preprocessing, runs), not the
    # one-hot-encoded feature names.
    assert len(importance_df) == len(X.columns)
    assert set(importance_df["feature"]) == set(X.columns)


def test_get_feature_importance_raises_without_fallback_data(
    classification_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = classification_train
    pipeline = build_classification_pipeline(KNeighborsClassifier())
    pipeline.fit(X, y)

    with pytest.raises(ValueError):
        get_feature_importance(pipeline)


def test_plot_feature_importance_creates_png(tmp_path: Path) -> None:
    importance_df = pd.DataFrame(
        {"feature": ["a", "b", "c"], "importance": [0.5, 0.3, 0.2]}
    )
    output_path = plot_feature_importance(importance_df, tmp_path / "importance.png")
    assert output_path.exists() and output_path.stat().st_size > 0


def test_compute_shap_values_shape_matches_transformed_features(
    fitted_rf_classifier: Pipeline,
    classification_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = classification_train
    shap_values, X_transformed = compute_shap_values(fitted_rf_classifier, X, max_samples=30)
    assert shap_values.shape == X_transformed.shape
    assert len(X_transformed) == 30


def test_compute_shap_values_regression_shape(
    fitted_rf_regressor: Pipeline,
    regression_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = regression_train
    shap_values, X_transformed = compute_shap_values(fitted_rf_regressor, X, max_samples=30)
    assert shap_values.shape == X_transformed.shape


def test_compute_shap_values_raises_for_non_tree_estimator(
    fitted_logistic_regression: Pipeline,
    classification_train: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = classification_train
    with pytest.raises(TypeError):
        compute_shap_values(fitted_logistic_regression, X, max_samples=30)


def test_plot_shap_summary_creates_png(
    fitted_rf_classifier: Pipeline,
    classification_train: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    X, _ = classification_train
    shap_values, X_transformed = compute_shap_values(fitted_rf_classifier, X, max_samples=30)
    output_path = plot_shap_summary(shap_values, X_transformed, tmp_path / "shap.png")
    assert output_path.exists() and output_path.stat().st_size > 0
