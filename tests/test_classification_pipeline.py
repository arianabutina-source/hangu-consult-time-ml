"""Tests for ml.pipelines.classification_pipeline: end-to-end plumbing smoke tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.pipelines.classification_pipeline import build_classification_pipeline


@pytest.fixture(scope="module")
def classification_train_test() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, test_df = grouped_stratified_split(engineered)
    X_train, y_train = select_classification_data(train_df)
    X_test, y_test = select_classification_data(test_df)
    return X_train, y_train, X_test, y_test


def test_build_classification_pipeline_default_estimator() -> None:
    pipeline = build_classification_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["preprocess", "model"]
    assert isinstance(pipeline.named_steps["model"], LogisticRegression)


def test_build_classification_pipeline_accepts_custom_estimator() -> None:
    custom_estimator = DecisionTreeClassifier(random_state=0)
    pipeline = build_classification_pipeline(custom_estimator)
    assert pipeline.named_steps["model"] is custom_estimator


def test_pipeline_fits_and_predicts_on_real_data(
    classification_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, y_test = classification_train_test
    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert set(np.unique(predictions)).issubset({False, True})


def test_pipeline_predict_proba_is_well_formed(
    classification_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, y_test = classification_train_test
    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)
    assert probabilities.shape == (len(X_test), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()


def test_pipeline_is_reproducible_given_fixed_random_state(
    classification_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    X_train, y_train, X_test, _ = classification_train_test

    pipeline_a = build_classification_pipeline()
    pipeline_a.fit(X_train, y_train)

    pipeline_b = build_classification_pipeline()
    pipeline_b.fit(X_train, y_train)

    np.testing.assert_array_equal(pipeline_a.predict(X_test), pipeline_b.predict(X_test))


def test_pipeline_beats_majority_class_baseline(
    classification_train_test: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """Smoke-level sanity check, not the official Milestone 11 evaluation."""
    X_train, y_train, X_test, y_test = classification_train_test
    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    majority_baseline = max(y_test.mean(), 1 - y_test.mean())
    accuracy = pipeline.score(X_test, y_test)
    assert accuracy >= majority_baseline
