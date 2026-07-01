"""Tests for hyperparameter tuning: tuning_spaces, search helpers, and both trainers.

Uses small n_iter/n_splits throughout to keep runtime reasonable while still
exercising every candidate estimator (including XGBoost) end to end on the
real dataset.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import RandomizedSearchCV

from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.training.search import save_tuning_results, summarize_search
from ml.training.train_classifier import (
    select_best_classifier,
    tune_all_classifiers,
    tune_classifier,
)
from ml.training.train_regressor import select_best_regressor, tune_all_regressors, tune_regressor
from ml.training.tuning_spaces import CLASSIFICATION_MODELS, REGRESSION_MODELS

_SMALL_N_ITER = 2
_SMALL_N_SPLITS = 3


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    engineered = engineer_features(clean_data(load_raw_data()))
    train, _ = grouped_stratified_split(engineered)
    return train


@pytest.fixture(scope="module")
def classification_data(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y = select_classification_data(train_df)
    return X, y, train_df["ID"]


@pytest.fixture(scope="module")
def regression_data(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X, y = select_regression_data(train_df)
    return X, y, train_df["ID"]


def test_classification_models_have_prefixed_param_keys() -> None:
    for spec in CLASSIFICATION_MODELS.values():
        for key in spec["param_distributions"]:
            assert key.startswith("model__")


def test_regression_models_have_prefixed_param_keys() -> None:
    for spec in REGRESSION_MODELS.values():
        for key in spec["param_distributions"]:
            assert key.startswith("model__")


def test_tune_classifier_returns_fitted_search(
    classification_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = classification_data
    search = tune_classifier(
        "logistic_regression", X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER
    )
    assert isinstance(search, RandomizedSearchCV)
    assert hasattr(search, "best_score_")
    assert 0.0 <= search.best_score_ <= 1.0  # roc_auc
    assert all(k.startswith("model__") for k in search.best_params_)


def test_tune_all_classifiers_covers_every_candidate(
    classification_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = classification_data
    results = tune_all_classifiers(X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER)
    assert set(results.keys()) == set(CLASSIFICATION_MODELS.keys())
    for search in results.values():
        assert hasattr(search, "best_score_")


def test_select_best_classifier_picks_highest_score(
    classification_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = classification_data
    results = tune_all_classifiers(X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER)
    best_name, best_search = select_best_classifier(results)
    assert best_name in results
    assert best_search.best_score_ == max(r.best_score_ for r in results.values())


def test_tune_regressor_returns_fitted_search(
    regression_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = regression_data
    search = tune_regressor(
        "ridge", X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER
    )
    assert isinstance(search, RandomizedSearchCV)
    assert search.best_score_ <= 0.0  # neg_mean_absolute_error: negative or zero


def test_tune_all_regressors_covers_every_candidate(
    regression_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = regression_data
    results = tune_all_regressors(X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER)
    assert set(results.keys()) == set(REGRESSION_MODELS.keys())


def test_select_best_regressor_picks_highest_score(
    regression_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = regression_data
    results = tune_all_regressors(X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER)
    best_name, best_search = select_best_regressor(results)
    assert best_name in results
    assert best_search.best_score_ == max(r.best_score_ for r in results.values())


def test_summarize_search_is_json_serializable(
    classification_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = classification_data
    search = tune_classifier(
        "logistic_regression", X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER
    )
    summary = summarize_search(search)
    assert "best_score" in summary
    assert "best_params" in summary
    json.dumps(summary)  # must not raise


def test_save_tuning_results_writes_valid_json(tmp_path: Path) -> None:
    results = {"model_a": {"best_score": 0.9, "best_params": {"model__C": 1.0}}}
    output_path = tmp_path / "tuning.json"
    save_tuning_results(results, output_path)

    assert output_path.exists()
    loaded = json.loads(output_path.read_text())
    assert loaded == results


def test_tuning_is_reproducible(
    classification_data: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y, groups = classification_data
    search_a = tune_classifier(
        "logistic_regression", X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER,
        random_state=7,
    )
    search_b = tune_classifier(
        "logistic_regression", X, y, groups, n_splits=_SMALL_N_SPLITS, n_iter=_SMALL_N_ITER,
        random_state=7,
    )
    assert search_a.best_score_ == search_b.best_score_
    assert search_a.best_params_ == search_b.best_params_
