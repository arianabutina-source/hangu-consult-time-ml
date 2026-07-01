"""Tests for ml.eda.summary: descriptive statistics computed from a DataFrame."""

import numpy as np
import pandas as pd
import pytest

from ml.eda.summary import (
    categorical_value_counts,
    descriptive_stats,
    full_eda_report,
    missing_value_report,
    patient_visit_report,
    service_time_summary,
    working_day_crosstab,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ["A", "A", "B", "C", "C", "C"],
            "DayOfWeek": ["Saturday", "Saturday", "Wednesday", "Saturday", "Friday", "Saturday"],
            "WorkingDay": [False, False, True, True, True, False],
            "Gender": ["F", "F", "M", "M", "F", "F"],
            "Address": ["In the city", None, "Out of city", None, None, "In the city"],
            "ServTime": [600, 700, 800, 900, 1000, 3600],
        }
    )


def test_missing_value_report_counts_and_percentages(sample_df: pd.DataFrame) -> None:
    report = missing_value_report(sample_df)
    assert report.loc["Address", "n_missing"] == 3
    assert report.loc["Address", "pct_missing"] == pytest.approx(50.0)
    assert report.loc["ID", "n_missing"] == 0


def test_missing_value_report_sorted_descending(sample_df: pd.DataFrame) -> None:
    report = missing_value_report(sample_df)
    assert report["pct_missing"].is_monotonic_decreasing


def test_descriptive_stats_default_numeric_columns(sample_df: pd.DataFrame) -> None:
    stats = descriptive_stats(sample_df)
    assert "ServTime" in stats.index
    assert stats.loc["ServTime", "min"] == 600
    assert stats.loc["ServTime", "max"] == 3600


def test_descriptive_stats_explicit_columns(sample_df: pd.DataFrame) -> None:
    stats = descriptive_stats(sample_df, columns=["ServTime"])
    assert list(stats.index) == ["ServTime"]


def test_categorical_value_counts(sample_df: pd.DataFrame) -> None:
    counts = categorical_value_counts(sample_df, "Gender")
    assert counts["F"] == 4
    assert counts["M"] == 2


def test_patient_visit_report_identifies_repeats(sample_df: pd.DataFrame) -> None:
    report = patient_visit_report(sample_df)
    assert report["n_rows"] == 6
    assert report["n_unique_patients"] == 3
    assert report["n_repeat_patients"] == 2  # A (2 visits), C (3 visits)
    assert report["max_visits_per_patient"] == 3


def test_working_day_crosstab_shape(sample_df: pd.DataFrame) -> None:
    crosstab = working_day_crosstab(sample_df)
    assert crosstab.loc["Saturday", False] == 3
    assert crosstab.loc["Saturday", True] == 1
    assert crosstab.loc["Wednesday", True] == 1


def test_service_time_summary_converts_to_minutes(sample_df: pd.DataFrame) -> None:
    summary = service_time_summary(sample_df)
    assert summary["count"] == 6
    assert summary["mean_minutes"] == pytest.approx(sample_df["ServTime"].mean() / 60)
    assert summary["min_seconds"] == 600
    assert summary["max_seconds"] == 3600


def test_service_time_summary_flags_iqr_outliers() -> None:
    # A tight cluster plus one extreme high value should be flagged.
    df = pd.DataFrame({"ServTime": [100, 105, 110, 108, 102, 100, 5000]})
    summary = service_time_summary(df)
    assert summary["n_iqr_outliers"] == 1


def test_full_eda_report_contains_all_sections(sample_df: pd.DataFrame) -> None:
    report = full_eda_report(sample_df)
    assert set(report.keys()) == {
        "missing_values",
        "descriptive_stats",
        "patient_visits",
        "working_day_crosstab",
        "service_time",
    }
