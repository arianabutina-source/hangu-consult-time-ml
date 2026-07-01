"""Feature and target engineering for the cleaned Hangu consultation dataset.

Derives both modelling targets and a small set of row-level engineered
features. No cross-row aggregation is performed (aside from the one-time,
whole-population threshold described below), so nothing here is at risk of
leaking information across the future train/test split.

Targets:
    * ``ServTime_minutes`` (regression target) -- ``ServTime`` seconds / 60.
    * ``IsLongConsultation`` (classification target) -- ``ServTime_minutes``
      above a fixed cutoff. The cutoff is the dataset-wide median duration,
      computed once from the full historical population and frozen as a
      constant business-rule threshold, analogous to a fixed clinical
      cutoff (e.g. "BMI > 30 is obese"). This is a label-definition
      decision, not a fitted preprocessing statistic, so deriving it from
      the full population does not leak information into a fitted model
      parameter across the later train/test split.

Engineered features:
    * ``IsRepeatVisit`` -- ``Visit.No`` > 1.
    * ``IsPublicHolidayMakeup`` -- ``WorkingDay`` is True on a day that is
      normally non-working for this clinic (Saturday); operationalises the
      "public holiday" predictor named in the original project
      requirements using the compensatory-Saturday pattern found in EDA.
    * ``HasCancerDiagnosis`` -- ``M.Cancer`` OR ``S.Cancer``, since
      ``S.Cancer`` alone is rare (~1% True) and sparse as a standalone
      predictor.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml.config import (
    CLASSIFICATION_TARGET_COLUMN,
    HAS_CANCER_DIAGNOSIS_COLUMN,
    IS_PUBLIC_HOLIDAY_MAKEUP_COLUMN,
    IS_REPEAT_VISIT_COLUMN,
    NORMALLY_NON_WORKING_DAY,
    RAW_SERVICE_TIME_COLUMN,
    SECONDS_PER_MINUTE,
    SERVICE_TIME_MINUTES_COLUMN,
    VISIT_NO_COLUMN,
)

logger = logging.getLogger(__name__)


def add_service_time_minutes(
    df: pd.DataFrame,
    seconds_column: str = RAW_SERVICE_TIME_COLUMN,
    minutes_column: str = SERVICE_TIME_MINUTES_COLUMN,
) -> pd.DataFrame:
    """Convert the consultation duration from seconds to minutes.

    Args:
        df: Input DataFrame containing ``seconds_column``.
        seconds_column: Name of the raw duration column (seconds).
        minutes_column: Name of the new duration column (minutes).

    Returns:
        A copy of ``df`` with ``minutes_column`` added.
    """
    out = df.copy()
    out[minutes_column] = out[seconds_column] / SECONDS_PER_MINUTE
    return out


def compute_long_consultation_threshold(
    df: pd.DataFrame,
    minutes_column: str = SERVICE_TIME_MINUTES_COLUMN,
    method: str = "median",
) -> float:
    """Compute the fixed cutoff (in minutes) used to define a "long" consultation.

    Args:
        df: DataFrame containing ``minutes_column``.
        minutes_column: Name of the consultation-duration column (minutes).
        method: Statistic used as the cutoff. Only ``"median"`` is
            currently supported.

    Returns:
        The computed threshold, in minutes.

    Raises:
        ValueError: If ``method`` is not supported.
    """
    if method != "median":
        raise ValueError(f"Unsupported threshold method: {method!r}")

    threshold = float(df[minutes_column].median())
    logger.info("Computed long-consultation threshold (%s): %.4f minutes", method, threshold)
    return threshold


def add_long_consultation_label(
    df: pd.DataFrame,
    threshold: float,
    minutes_column: str = SERVICE_TIME_MINUTES_COLUMN,
    label_column: str = CLASSIFICATION_TARGET_COLUMN,
) -> pd.DataFrame:
    """Derive the binary classification target from a fixed duration cutoff.

    Args:
        df: Input DataFrame containing ``minutes_column``.
        threshold: Fixed cutoff, in minutes (see
            :func:`compute_long_consultation_threshold`).
        minutes_column: Name of the consultation-duration column (minutes).
        label_column: Name of the new binary label column.

    Returns:
        A copy of ``df`` with ``label_column`` added.
    """
    out = df.copy()
    out[label_column] = out[minutes_column] > threshold

    balance = out[label_column].mean()
    logger.info(
        "Derived '%s' at threshold=%.4f minutes: %.1f%% positive class",
        label_column,
        threshold,
        100 * balance,
    )
    return out


def add_repeat_visit_flag(
    df: pd.DataFrame,
    visit_no_column: str = VISIT_NO_COLUMN,
    output_column: str = IS_REPEAT_VISIT_COLUMN,
) -> pd.DataFrame:
    """Flag whether a consultation is a repeat visit (``Visit.No`` > 1).

    Args:
        df: Input DataFrame containing ``visit_no_column``.
        visit_no_column: Name of the visit-number column.
        output_column: Name of the new boolean flag column.

    Returns:
        A copy of ``df`` with ``output_column`` added.
    """
    out = df.copy()
    out[output_column] = out[visit_no_column] > 1
    return out


def add_public_holiday_makeup_flag(
    df: pd.DataFrame,
    day_column: str = "DayOfWeek",
    working_day_column: str = "WorkingDay",
    normally_non_working_day: str = NORMALLY_NON_WORKING_DAY,
    output_column: str = IS_PUBLIC_HOLIDAY_MAKEUP_COLUMN,
) -> pd.DataFrame:
    """Flag a working day that falls on the clinic's normally non-working day.

    Args:
        df: Input DataFrame containing ``day_column`` and ``working_day_column``.
        day_column: Name of the day-of-week column.
        working_day_column: Name of the boolean working-day column.
        normally_non_working_day: The day of week normally off (Saturday,
            per EDA).
        output_column: Name of the new boolean flag column.

    Returns:
        A copy of ``df`` with ``output_column`` added.
    """
    out = df.copy()
    out[output_column] = (out[day_column] == normally_non_working_day) & out[working_day_column]
    return out


def add_combined_cancer_flag(
    df: pd.DataFrame,
    primary_column: str = "M.Cancer",
    secondary_column: str = "S.Cancer",
    output_column: str = HAS_CANCER_DIAGNOSIS_COLUMN,
) -> pd.DataFrame:
    """Combine primary/secondary cancer flags into a single indicator.

    Args:
        df: Input DataFrame containing ``primary_column`` and ``secondary_column``.
        primary_column: Name of the primary-cancer boolean column.
        secondary_column: Name of the secondary-cancer boolean column.
        output_column: Name of the new combined boolean flag column.

    Returns:
        A copy of ``df`` with ``output_column`` added.
    """
    out = df.copy()
    out[output_column] = out[primary_column] | out[secondary_column]
    return out


def engineer_features(df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
    """Apply all target and feature derivations to a cleaned dataset.

    Args:
        df: Cleaned dataset, as returned by ``ml.data.clean.clean_data``.
        threshold: Fixed long-consultation cutoff, in minutes. If None
            (default), it is computed from ``df`` via
            :func:`compute_long_consultation_threshold`.

    Returns:
        A copy of ``df`` with the regression target, classification target,
        and engineered features added. No original columns are dropped or
        modified (column selection is handled in Milestone 5).
    """
    logger.info("Engineering features for %d rows", len(df))

    out = add_service_time_minutes(df)

    if threshold is None:
        threshold = compute_long_consultation_threshold(out)
    out = add_long_consultation_label(out, threshold=threshold)

    out = add_repeat_visit_flag(out)
    out = add_public_holiday_makeup_flag(out)
    out = add_combined_cancer_flag(out)

    logger.info("Feature engineering complete: %d rows, %d columns", *out.shape)
    return out


if __name__ == "__main__":
    from ml.data.clean import clean_data
    from ml.data.load import load_raw_data

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    features_df = engineer_features(clean_data(load_raw_data()))
    print(features_df[[SERVICE_TIME_MINUTES_COLUMN, CLASSIFICATION_TARGET_COLUMN]].describe(include="all"))
