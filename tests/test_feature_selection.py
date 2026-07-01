"""Tests for ml.data.feature_selection: final predictor lists and leakage guards."""

import pandas as pd
import pytest

from ml.data.clean import clean_data
from ml.data.feature_selection import (
    EXCLUDED_COLUMNS,
    get_classification_features,
    get_regression_features,
    select_classification_data,
    select_regression_data,
)
from ml.data.features import engineer_features
from ml.data.load import load_raw_data


@pytest.fixture(scope="module")
def engineered_df() -> pd.DataFrame:
    return engineer_features(clean_data(load_raw_data()))


def test_classification_features_exclude_leakage_columns() -> None:
    features = get_classification_features()
    for excluded_column in EXCLUDED_COLUMNS:
        assert excluded_column not in features


def test_regression_features_exclude_leakage_columns() -> None:
    features = get_regression_features()
    for excluded_column in EXCLUDED_COLUMNS:
        assert excluded_column not in features


def test_classification_features_exclude_own_target() -> None:
    assert "IsLongConsultation" not in get_classification_features()


def test_regression_features_exclude_own_target() -> None:
    assert "ServTime_minutes" not in get_regression_features()


def test_id_column_never_a_predictor() -> None:
    assert "ID" not in get_classification_features()
    assert "ID" not in get_regression_features()


def test_session_column_never_a_predictor() -> None:
    assert "Session" not in get_classification_features()
    assert "Session" not in get_regression_features()


def test_timestamp_columns_never_predictors() -> None:
    for col in ("StartTime", "PayTime"):
        assert col not in get_classification_features()
        assert col not in get_regression_features()


def test_select_classification_data_shapes(engineered_df: pd.DataFrame) -> None:
    X, y = select_classification_data(engineered_df)
    assert len(X) == len(engineered_df)
    assert list(X.columns) == get_classification_features()
    assert y.name == "IsLongConsultation"
    assert y.isna().sum() == 0


def test_select_regression_data_shapes(engineered_df: pd.DataFrame) -> None:
    X, y = select_regression_data(engineered_df)
    assert len(X) == len(engineered_df)
    assert list(X.columns) == get_regression_features()
    assert y.name == "ServTime_minutes"
    assert y.isna().sum() == 0


def test_select_classification_data_raises_on_missing_column(
    engineered_df: pd.DataFrame,
) -> None:
    broken = engineered_df.drop(columns=["Gender"])
    with pytest.raises(KeyError):
        select_classification_data(broken)
