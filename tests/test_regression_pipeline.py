"""Tests for ml.pipelines.regression_pipeline: end-to-end plumbing smoke tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from ml.data.clean import clean_data
from ml.data.feature_selection import select_regression_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.pipelines.regression_pipeline import build_regression_pipeline


@pytest.fixture(scope="module")
def regression_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, test_df = grouped_stratified_split(engineered)
    X_train, y_train = select_regression_data(train_df)
    X_test, y_test = select_regression_data(test_df)
    return X_train, y_train, X_test, y_test


def test_build_regression_pipeline_default_estimator() -> None:
    pipeline = build_regression_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    assert isinstance(pipeline.named_steps["model"], Ridge)


def test_build_regression_pipeline_accepts_custom_estimator() -> None:
    custom_estimator = DecisionTreeRegressor(random_state=0)
    pipeline = build_regression_pipeline(custom_estimator)
    assert pipeline.named_steps["model"] is custom_estimator


def test_pipeline_fits_and_predicts_on_real_data(
    regression_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, y_test = regression_train_test
    pipeline = build_regression_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert not np.isnan(predictions).any()


def test_pipeline_predicts_plausible_durations(
    regression_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Predicted durations should stay within a sane range (minutes), not blow up."""
    X_train, y_train, X_test, _ = regression_train_test
    pipeline = build_regression_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert (predictions > 0).all()
    assert (predictions < 120).all()


def test_pipeline_is_reproducible_given_fixed_random_state(
    regression_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, _ = regression_train_test

    pipeline_a = build_regression_pipeline()
    pipeline_a.fit(X_train, y_train)

    pipeline_b = build_regression_pipeline()
    pipeline_b.fit(X_train, y_train)

    np.testing.assert_array_almost_equal(
        pipeline_a.predict(X_test), pipeline_b.predict(X_test)
    )


def test_pipeline_beats_mean_baseline(
    regression_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Smoke-level sanity check, not the official Milestone 11 evaluation."""
    X_train, y_train, X_test, y_test = regression_train_test
    pipeline = build_regression_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    model_mae = mean_absolute_error(y_test, predictions)

    naive_predictions = np.full(len(y_test), y_train.mean())
    naive_mae = mean_absolute_error(y_test, naive_predictions)

    assert model_mae <= naive_mae
