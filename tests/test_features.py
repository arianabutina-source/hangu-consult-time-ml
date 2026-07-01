"""Tests for ml.data.features: target derivation and engineered features."""

import pandas as pd
import pytest

from ml.data.clean import clean_data
from ml.data.features import (
    add_combined_cancer_flag,
    add_long_consultation_label,
    add_public_holiday_makeup_flag,
    add_repeat_visit_flag,
    add_service_time_minutes,
    compute_long_consultation_threshold,
    engineer_features,
)
from ml.data.load import load_raw_data


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["A", "B", "C", "D"],
            "Visit.No": [1, 2, 1, 5],
            "DayOfWeek": ["Saturday", "Saturday", "Wednesday", "Friday"],
            "WorkingDay": [False, True, True, True],
            "M.Cancer": [True, False, False, False],
            "S.Cancer": [False, False, True, False],
            "ServTime": [600, 900, 1200, 1800],
        }
    )


def test_add_service_time_minutes_converts_correctly(sample_df: pd.DataFrame) -> None:
    out = add_service_time_minutes(sample_df)
    assert out["ServTime_minutes"].tolist() == [10.0, 15.0, 20.0, 30.0]


def test_add_service_time_minutes_does_not_mutate_input(sample_df: pd.DataFrame) -> None:
    add_service_time_minutes(sample_df)
    assert "ServTime_minutes" not in sample_df.columns


def test_compute_long_consultation_threshold_is_median(sample_df: pd.DataFrame) -> None:
    out = add_service_time_minutes(sample_df)
    threshold = compute_long_consultation_threshold(out)
    assert threshold == pytest.approx(out["ServTime_minutes"].median())
    assert threshold == pytest.approx(17.5)


def test_compute_long_consultation_threshold_rejects_unknown_method(
    sample_df: pd.DataFrame,
) -> None:
    out = add_service_time_minutes(sample_df)
    with pytest.raises(ValueError):
        compute_long_consultation_threshold(out, method="mean")


def test_add_long_consultation_label_matches_threshold(sample_df: pd.DataFrame) -> None:
    out = add_service_time_minutes(sample_df)
    labelled = add_long_consultation_label(out, threshold=17.5)
    assert labelled["IsLongConsultation"].tolist() == [False, False, True, True]


def test_add_repeat_visit_flag(sample_df: pd.DataFrame) -> None:
    out = add_repeat_visit_flag(sample_df)
    assert out["IsRepeatVisit"].tolist() == [False, True, False, True]


def test_add_public_holiday_makeup_flag(sample_df: pd.DataFrame) -> None:
    out = add_public_holiday_makeup_flag(sample_df)
    # Only row B is Saturday AND working (compensatory holiday makeup).
    assert out["IsPublicHolidayMakeup"].tolist() == [False, True, False, False]


def test_add_combined_cancer_flag(sample_df: pd.DataFrame) -> None:
    out = add_combined_cancer_flag(sample_df)
    assert out["HasCancerDiagnosis"].tolist() == [True, False, True, False]


def test_engineer_features_adds_all_expected_columns(sample_df: pd.DataFrame) -> None:
    out = engineer_features(sample_df)
    expected_new_columns = {
        "ServTime_minutes",
        "IsLongConsultation",
        "IsRepeatVisit",
        "IsPublicHolidayMakeup",
        "HasCancerDiagnosis",
    }
    assert expected_new_columns.issubset(out.columns)


def test_engineer_features_preserves_row_and_original_column_count(
    sample_df: pd.DataFrame,
) -> None:
    out = engineer_features(sample_df)
    assert len(out) == len(sample_df)
    assert set(sample_df.columns).issubset(out.columns)


def test_engineer_features_accepts_explicit_threshold(sample_df: pd.DataFrame) -> None:
    out = engineer_features(sample_df, threshold=0.0)
    assert out["IsLongConsultation"].all()


def test_engineer_features_on_real_dataset_runs_end_to_end() -> None:
    """Integration test: full clean -> engineer pipeline on the real dataset."""
    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)

    assert len(engineered) == len(raw)
    assert engineered["ServTime_minutes"].isna().sum() == 0
    assert engineered["IsLongConsultation"].isna().sum() == 0

    # Median-based threshold should yield a roughly balanced split.
    balance = engineered["IsLongConsultation"].mean()
    assert 0.3 < balance < 0.7
