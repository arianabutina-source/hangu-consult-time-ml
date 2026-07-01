"""Tests for ml.training.cv_utils: group-aware CV splitters and scoring."""

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import GroupLeakageError, grouped_stratified_split
from ml.pipelines.classification_pipeline import build_classification_pipeline
from ml.pipelines.regression_pipeline import build_regression_pipeline
from ml.training.cv_utils import (
    assert_cv_has_no_group_leakage,
    cross_validate_classification_pipeline,
    cross_validate_regression_pipeline,
    get_classification_cv,
    get_regression_cv,
)


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    engineered = engineer_features(clean_data(load_raw_data()))
    train, _ = grouped_stratified_split(engineered)
    return train


class _LeakyCV:
    """A fake CV splitter that deliberately puts the same group in both halves."""

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return 1

    def split(self, X, y=None, groups=None):
        n = len(X)
        half = n // 2
        # Overlap the last index of "train" with the first index of "val".
        train_idx = list(range(0, half + 1))
        val_idx = list(range(half, n))
        yield train_idx, val_idx


def test_get_classification_cv_returns_stratified_group_kfold() -> None:
    cv = get_classification_cv(n_splits=4, random_state=0)
    assert isinstance(cv, StratifiedGroupKFold)
    assert cv.get_n_splits() == 4


def test_get_regression_cv_returns_group_kfold() -> None:
    cv = get_regression_cv(n_splits=3, random_state=0)
    assert isinstance(cv, GroupKFold)
    assert cv.get_n_splits() == 3


def test_assert_cv_has_no_group_leakage_detects_leaky_splitter() -> None:
    X = pd.DataFrame({"x": range(10)})
    y = pd.Series([0, 1] * 5)
    groups = pd.Series(range(10))  # unique groups, but the fake CV overlaps indices anyway

    with pytest.raises(GroupLeakageError):
        assert_cv_has_no_group_leakage(_LeakyCV(), X, y, groups)


def test_classification_cv_has_no_group_leakage_on_real_data(train_df: pd.DataFrame) -> None:
    X, y = select_classification_data(train_df)
    cv = get_classification_cv()
    assert_cv_has_no_group_leakage(cv, X, y, train_df["ID"])  # should not raise


def test_regression_cv_has_no_group_leakage_on_real_data(train_df: pd.DataFrame) -> None:
    X, y = select_regression_data(train_df)
    cv = get_regression_cv()
    assert_cv_has_no_group_leakage(cv, X, y, train_df["ID"])  # should not raise


def test_cross_validate_classification_pipeline_returns_expected_metrics(
    train_df: pd.DataFrame,
) -> None:
    X, y = select_classification_data(train_df)
    results = cross_validate_classification_pipeline(
        build_classification_pipeline(), X, y, train_df["ID"], n_splits=3
    )

    assert set(results.keys()) == {"accuracy", "f1", "roc_auc"}
    for metric_summary in results.values():
        assert 0.0 <= metric_summary["mean"] <= 1.0
        assert metric_summary["std"] >= 0.0


def test_cross_validate_regression_pipeline_returns_expected_metrics(
    train_df: pd.DataFrame,
) -> None:
    X, y = select_regression_data(train_df)
    results = cross_validate_regression_pipeline(
        build_regression_pipeline(), X, y, train_df["ID"], n_splits=3
    )

    assert set(results.keys()) == {"mae", "rmse", "r2"}
    assert results["mae"]["mean"] > 0.0
    assert results["rmse"]["mean"] >= results["mae"]["mean"]


def test_cv_results_are_reproducible(train_df: pd.DataFrame) -> None:
    X, y = select_classification_data(train_df)
    results_a = cross_validate_classification_pipeline(
        build_classification_pipeline(), X, y, train_df["ID"], n_splits=3, random_state=7
    )
    results_b = cross_validate_classification_pipeline(
        build_classification_pipeline(), X, y, train_df["ID"], n_splits=3, random_state=7
    )
    assert results_a == results_b
