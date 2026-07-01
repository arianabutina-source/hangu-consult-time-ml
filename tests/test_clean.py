"""Tests for ml.data.clean: documented cleaning rules over the raw dataset."""

import pandas as pd
import pytest

from ml.data.clean import (
    DataCleaningError,
    check_no_duplicate_rows,
    clean_data,
    fill_missing_address,
    report_patient_label_inconsistency,
    validate_categorical_domains,
    validate_service_time_positive,
)
from ml.data.load import load_raw_data


@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    """A minimal DataFrame matching the columns clean_data() touches."""
    return pd.DataFrame(
        {
            "ID": ["A", "A", "B", "C"],
            "Month": ["January", "January", "February", "March"],
            "DayOfWeek": ["Saturday", "Saturday", "Wednesday", "Friday"],
            "AM_PM": ["morning", "afternoon", "morning", "afternoon"],
            "Gender": ["F", "F", "M", "M"],
            "M.Cancer": [True, False, False, False],
            "Address": ["In the city", None, "Out of city", "Out of province"],
            "ServTime": [600, 700, 800, 900],
        }
    )


def test_fill_missing_address_replaces_nan(raw_like_df: pd.DataFrame) -> None:
    filled = fill_missing_address(raw_like_df)
    assert filled["Address"].isna().sum() == 0
    assert filled.loc[1, "Address"] == "Unknown"


def test_fill_missing_address_does_not_mutate_input(raw_like_df: pd.DataFrame) -> None:
    fill_missing_address(raw_like_df)
    assert raw_like_df["Address"].isna().sum() == 1


def test_check_no_duplicate_rows_passes_on_unique_rows(raw_like_df: pd.DataFrame) -> None:
    check_no_duplicate_rows(raw_like_df)  # should not raise


def test_check_no_duplicate_rows_raises_on_duplicates(raw_like_df: pd.DataFrame) -> None:
    with_dupe = pd.concat([raw_like_df, raw_like_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataCleaningError):
        check_no_duplicate_rows(with_dupe)


def test_validate_service_time_positive_passes(raw_like_df: pd.DataFrame) -> None:
    validate_service_time_positive(raw_like_df)  # should not raise


def test_validate_service_time_positive_raises_on_zero(raw_like_df: pd.DataFrame) -> None:
    bad = raw_like_df.copy()
    bad.loc[0, "ServTime"] = 0
    with pytest.raises(DataCleaningError):
        validate_service_time_positive(bad)


def test_validate_service_time_positive_raises_on_negative(raw_like_df: pd.DataFrame) -> None:
    bad = raw_like_df.copy()
    bad.loc[0, "ServTime"] = -5
    with pytest.raises(DataCleaningError):
        validate_service_time_positive(bad)


def test_report_patient_label_inconsistency_counts_correctly(raw_like_df: pd.DataFrame) -> None:
    # Patient "A" has M.Cancer = True on one visit and False on another.
    n_inconsistent = report_patient_label_inconsistency(raw_like_df)
    assert n_inconsistent == 1


def test_validate_categorical_domains_passes_on_clean_data(raw_like_df: pd.DataFrame) -> None:
    filled = fill_missing_address(raw_like_df)
    validate_categorical_domains(filled)  # should not raise


def test_validate_categorical_domains_raises_on_unexpected_value(raw_like_df: pd.DataFrame) -> None:
    filled = fill_missing_address(raw_like_df)
    filled.loc[0, "Gender"] = "X"
    with pytest.raises(DataCleaningError):
        validate_categorical_domains(filled)


def test_clean_data_leaves_no_missing_address(raw_like_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_like_df)
    assert cleaned["Address"].isna().sum() == 0


def test_clean_data_does_not_drop_rows_or_columns(raw_like_df: pd.DataFrame) -> None:
    cleaned = clean_data(raw_like_df)
    assert cleaned.shape[0] == raw_like_df.shape[0]
    assert list(cleaned.columns) == list(raw_like_df.columns)


def test_clean_data_on_real_dataset_runs_end_to_end() -> None:
    """Integration test: clean_data must succeed on the actual project dataset."""
    raw = load_raw_data()
    cleaned = clean_data(raw)
    assert cleaned.shape[0] == raw.shape[0]
    assert cleaned["Address"].isna().sum() == 0
    # PayTime/StartTime are intentionally left untouched.
    assert cleaned["PayTime"].isna().sum() == raw["PayTime"].isna().sum()
