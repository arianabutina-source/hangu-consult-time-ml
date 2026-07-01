"""Tests for ml.preprocessing.column_transformer: leakage-safe fit/transform behaviour."""

import numpy as np
import pandas as pd
import pytest

from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.preprocessing.column_transformer import (
    build_classification_transformer,
    build_regression_transformer,
)


@pytest.fixture(scope="module")
def train_test_predictors() -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, test_df = grouped_stratified_split(engineered)
    X_train, _ = select_classification_data(train_df)
    X_test, _ = select_classification_data(test_df)
    return X_train, X_test


def test_build_classification_transformer_fits_and_transforms(
    train_test_predictors: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    X_train, X_test = train_test_predictors
    transformer = build_classification_transformer()

    transformed_train = transformer.fit_transform(X_train)
    transformed_test = transformer.transform(X_test)

    assert transformed_train.shape[0] == len(X_train)
    assert transformed_test.shape[0] == len(X_test)
    assert transformed_train.shape[1] == transformed_test.shape[1]


def test_build_regression_transformer_fits_and_transforms() -> None:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, test_df = grouped_stratified_split(engineered)
    X_train, _ = select_regression_data(train_df)
    X_test, _ = select_regression_data(test_df)

    transformer = build_regression_transformer()
    transformed_train = transformer.fit_transform(X_train)
    transformed_test = transformer.transform(X_test)

    assert transformed_train.shape[0] == len(X_train)
    assert transformed_test.shape[0] == len(X_test)


def test_transformer_is_not_refit_on_test_data(
    train_test_predictors: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Calling .transform (not .fit) on test data must not raise, even with
    categories unseen during training (handle_unknown="ignore")."""
    X_train, X_test = train_test_predictors
    transformer = build_classification_transformer()
    transformer.fit(X_train)

    X_test_with_unseen = X_test.copy()
    X_test_with_unseen.loc[X_test_with_unseen.index[0], "Address"] = "Unseen Category XYZ"

    transformed = transformer.transform(X_test_with_unseen)
    assert transformed.shape[0] == len(X_test_with_unseen)


def test_transformer_output_has_no_nans(
    train_test_predictors: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    X_train, X_test = train_test_predictors
    transformer = build_classification_transformer()
    transformer.fit(X_train)

    transformed = transformer.transform(X_test)
    assert not np.isnan(transformed).any()


def test_transformer_handles_missing_values_in_test_only(
    train_test_predictors: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The fitted imputer (learned from train) must be able to fill NaNs
    introduced only in the test set, without being refit."""
    X_train, X_test = train_test_predictors
    transformer = build_classification_transformer()
    transformer.fit(X_train)

    X_test_with_nan = X_test.copy()
    X_test_with_nan.loc[X_test_with_nan.index[0], "Visit.No"] = np.nan
    X_test_with_nan.loc[X_test_with_nan.index[1], "Gender"] = np.nan

    transformed = transformer.transform(X_test_with_nan)
    assert not np.isnan(transformed).any()


def test_classification_and_regression_transformers_are_independent_instances() -> None:
    clf_transformer = build_classification_transformer()
    reg_transformer = build_regression_transformer()
    assert clf_transformer is not reg_transformer
