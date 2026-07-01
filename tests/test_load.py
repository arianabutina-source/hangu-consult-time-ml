"""Tests for ml.data.load: raw dataset ingestion and schema validation."""

from pathlib import Path

import pandas as pd
import pytest

from ml.config import (
    EXPECTED_BOOLEAN_COLUMNS,
    EXPECTED_COLUMNS,
    EXPECTED_N_COLUMNS,
    EXPECTED_N_ROWS,
    EXPECTED_NUMERIC_COLUMNS,
)
from ml.data.load import DatasetSchemaError, load_raw_data


def test_load_raw_data_returns_expected_shape() -> None:
    df = load_raw_data()
    assert df.shape == (EXPECTED_N_ROWS, EXPECTED_N_COLUMNS)


def test_load_raw_data_returns_expected_columns() -> None:
    df = load_raw_data()
    assert list(df.columns) == EXPECTED_COLUMNS


def test_load_raw_data_numeric_dtypes() -> None:
    df = load_raw_data()
    for col in EXPECTED_NUMERIC_COLUMNS:
        assert pd.api.types.is_integer_dtype(df[col]), f"{col} is not integer dtype"


def test_load_raw_data_boolean_dtypes() -> None:
    df = load_raw_data()
    for col in EXPECTED_BOOLEAN_COLUMNS:
        assert pd.api.types.is_bool_dtype(df[col]), f"{col} is not boolean dtype"


def test_load_raw_data_is_deterministic() -> None:
    """Loading twice must yield identical data (no randomness in ingestion)."""
    first = load_raw_data()
    second = load_raw_data()
    pd.testing.assert_frame_equal(first, second)


def test_load_raw_data_missing_file_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(path=missing_path)


def test_load_raw_data_rejects_bad_schema(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad_csv, index=False)
    with pytest.raises(DatasetSchemaError):
        load_raw_data(path=bad_csv)


def test_load_raw_data_skips_validation_when_disabled(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]}).to_csv(bad_csv, index=False)
    df = load_raw_data(path=bad_csv, validate=False)
    assert list(df.columns) == ["col_a", "col_b"]
