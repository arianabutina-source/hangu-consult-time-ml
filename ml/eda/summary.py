"""Descriptive summary statistics for the raw Hangu consultation dataset.

Every function here is a pure, side-effect-free computation over a
DataFrame: no plotting and no mutation of the input. Keeping summaries and
plots (``ml.eda.plots``) separate makes each independently unit-testable.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ml.config import PATIENT_ID_COLUMN, RAW_SERVICE_TIME_COLUMN, SECONDS_PER_MINUTE

logger = logging.getLogger(__name__)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column, worst first.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame indexed by column name with ``n_missing`` and
        ``pct_missing`` columns, sorted descending by ``pct_missing``.
    """
    n_missing = df.isna().sum()
    pct_missing = (n_missing / len(df) * 100).round(2)
    report = pd.DataFrame({"n_missing": n_missing, "pct_missing": pct_missing})
    report = report.sort_values("pct_missing", ascending=False)

    high_missing = report[report["pct_missing"] > 30]
    if not high_missing.empty:
        logger.warning(
            "Columns with >30%% missing values: %s", list(high_missing.index)
        )

    return report


def descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Standard descriptive statistics (count/mean/std/min/quartiles/max).

    Args:
        df: Input DataFrame.
        columns: Numeric columns to describe. Defaults to all numeric columns.

    Returns:
        DataFrame with one row per requested column.
    """
    subset = df[columns] if columns is not None else df.select_dtypes(include="number")
    return subset.describe().T


def categorical_value_counts(df: pd.DataFrame, column: str) -> pd.Series:
    """Value counts (including NaN) for a single categorical column.

    Args:
        df: Input DataFrame.
        column: Name of the categorical column.

    Returns:
        Series of counts, most frequent first.
    """
    return df[column].value_counts(dropna=False)


def patient_visit_report(
    df: pd.DataFrame, id_column: str = PATIENT_ID_COLUMN
) -> dict[str, Any]:
    """Summarise repeat-visit structure at the patient level.

    Rows sharing the same patient ID are not independent observations, so
    this informs the grouped train/test split required later (Milestone 6).

    Args:
        df: Input DataFrame.
        id_column: Column identifying the patient.

    Returns:
        Dict with row/patient counts and repeat-visit statistics.
    """
    visits_per_patient = df[id_column].value_counts()
    n_repeat_patients = int((visits_per_patient > 1).sum())

    report = {
        "n_rows": len(df),
        "n_unique_patients": int(df[id_column].nunique()),
        "n_repeat_patients": n_repeat_patients,
        "max_visits_per_patient": int(visits_per_patient.max()),
        "mean_visits_per_patient": float(visits_per_patient.mean()),
    }
    logger.info(
        "%d/%d rows (%.1f%%) belong to patients with more than one visit",
        n_repeat_patients,
        report["n_unique_patients"],
        100 * n_repeat_patients / report["n_unique_patients"],
    )
    return report


def working_day_crosstab(
    df: pd.DataFrame, day_column: str = "DayOfWeek", working_day_column: str = "WorkingDay"
) -> pd.DataFrame:
    """Cross-tabulate day of week against the working-day flag.

    Reveals whether ``WorkingDay`` carries information beyond ``DayOfWeek``
    (e.g. compensatory working Saturdays around public holidays).

    Args:
        df: Input DataFrame.
        day_column: Name of the day-of-week column.
        working_day_column: Name of the boolean working-day column.

    Returns:
        Contingency table of counts.
    """
    return pd.crosstab(df[day_column], df[working_day_column])


def service_time_summary(
    df: pd.DataFrame, seconds_column: str = RAW_SERVICE_TIME_COLUMN
) -> dict[str, Any]:
    """Distribution summary for the consultation-duration target column.

    Args:
        df: Input DataFrame.
        seconds_column: Name of the duration column, recorded in seconds.

    Returns:
        Dict with count/mean/std/quartiles/skew in both seconds and minutes,
        plus a Tukey IQR-based outlier count.
    """
    seconds = df[seconds_column]
    minutes = seconds / SECONDS_PER_MINUTE

    q1, q3 = seconds.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_fence, upper_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(((seconds < lower_fence) | (seconds > upper_fence)).sum())

    return {
        "count": int(seconds.count()),
        "mean_seconds": float(seconds.mean()),
        "std_seconds": float(seconds.std()),
        "min_seconds": float(seconds.min()),
        "max_seconds": float(seconds.max()),
        "mean_minutes": float(minutes.mean()),
        "median_minutes": float(minutes.median()),
        "std_minutes": float(minutes.std()),
        "skew": float(seconds.skew()),
        "n_iqr_outliers": n_outliers,
    }


def full_eda_report(df: pd.DataFrame) -> dict[str, Any]:
    """Convenience wrapper bundling every summary for notebook/report use.

    Args:
        df: Input DataFrame (raw, unmodified).

    Returns:
        Dict keyed by report section name.
    """
    logger.info("Computing full EDA report for %d rows", len(df))
    return {
        "missing_values": missing_value_report(df),
        "descriptive_stats": descriptive_stats(df),
        "patient_visits": patient_visit_report(df),
        "working_day_crosstab": working_day_crosstab(df),
        "service_time": service_time_summary(df),
    }
