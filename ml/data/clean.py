"""Data cleaning for the raw Hangu consultation dataset.

Every rule implemented here is a deliberate, documented decision grounded in
the Milestone 2 EDA findings -- not generic imputation. In particular:

* ``Address`` (~36% missing) is given an explicit "Unknown" category rather
  than imputed, since missingness is plausibly informative on its own.
* ``PayTime`` (~4% missing) and ``StartTime`` are left untouched: EDA showed
  ``PayTime - StartTime`` does not reliably reconstruct ``ServTime``, so
  these columns are not used to derive or validate the target and do not
  need imputation.
* Repeat-visit patients with an inconsistent ``M.Cancer`` flag across visits
  are **not** forcibly reconciled -- each row is an independent consultation
  record and a diagnosis can legitimately change between visits. The
  inconsistency is only reported for transparency.
* ``ServTime`` outliers (identified via IQR in the EDA) are **not** removed;
  long consultations are a plausible clinical phenomenon, not noise.

This module performs no feature engineering (see ``ml.data.features``) and
no column dropping (see ``ml.data.feature_selection``); it only fixes data
quality issues in the columns as they exist in the raw schema.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml.config import (
    ADDRESS_COLUMN,
    ADDRESS_MISSING_LABEL,
    EXPECTED_ADDRESS_VALUES,
    EXPECTED_AMPM_VALUES,
    EXPECTED_DAYOFWEEK_VALUES,
    EXPECTED_GENDER_VALUES,
    EXPECTED_MONTH_VALUES,
    PATIENT_ID_COLUMN,
    RAW_SERVICE_TIME_COLUMN,
)

logger = logging.getLogger(__name__)


class DataCleaningError(ValueError):
    """Raised when the data fails a cleaning-stage quality check."""


def fill_missing_address(
    df: pd.DataFrame,
    column: str = ADDRESS_COLUMN,
    missing_label: str = ADDRESS_MISSING_LABEL,
) -> pd.DataFrame:
    """Replace missing values in ``column`` with an explicit category.

    Args:
        df: Input DataFrame.
        column: Name of the column to fill.
        missing_label: Category used to represent missingness.

    Returns:
        A copy of ``df`` with missing values in ``column`` filled.
    """
    n_missing = int(df[column].isna().sum())
    out = df.copy()
    out[column] = out[column].fillna(missing_label)
    logger.info("Filled %d missing '%s' values with '%s'", n_missing, column, missing_label)
    return out


def check_no_duplicate_rows(df: pd.DataFrame) -> None:
    """Verify there are no fully duplicated rows.

    Raises:
        DataCleaningError: If any full-row duplicates are found.
    """
    n_duplicates = int(df.duplicated().sum())
    if n_duplicates > 0:
        raise DataCleaningError(f"Found {n_duplicates} fully duplicated rows.")
    logger.info("No duplicate rows found (%d rows checked)", len(df))


def validate_categorical_domains(df: pd.DataFrame) -> None:
    """Guard against unexpected categorical values (e.g. future data drift).

    Assumes ``Address`` has already been passed through
    :func:`fill_missing_address`.

    Raises:
        DataCleaningError: If any column contains a value outside its
            expected domain.
    """
    checks = {
        "Gender": EXPECTED_GENDER_VALUES,
        "Month": EXPECTED_MONTH_VALUES,
        "DayOfWeek": EXPECTED_DAYOFWEEK_VALUES,
        "AM_PM": EXPECTED_AMPM_VALUES,
        ADDRESS_COLUMN: EXPECTED_ADDRESS_VALUES,
    }
    for column, expected_values in checks.items():
        actual_values = set(df[column].dropna().unique())
        unexpected = actual_values - expected_values
        if unexpected:
            raise DataCleaningError(
                f"Column '{column}' contains unexpected values: {unexpected}"
            )
    logger.info("All categorical columns fall within their expected domains")


def validate_service_time_positive(
    df: pd.DataFrame, column: str = RAW_SERVICE_TIME_COLUMN
) -> None:
    """Verify the consultation-duration column contains only positive values.

    Raises:
        DataCleaningError: If any value in ``column`` is zero or negative.
    """
    n_non_positive = int((df[column] <= 0).sum())
    if n_non_positive > 0:
        raise DataCleaningError(
            f"Found {n_non_positive} non-positive values in '{column}'."
        )
    logger.info("All '%s' values are strictly positive", column)


def report_patient_label_inconsistency(
    df: pd.DataFrame,
    id_column: str = PATIENT_ID_COLUMN,
    flag_column: str = "M.Cancer",
) -> int:
    """Count patients whose ``flag_column`` differs across repeat visits.

    This is intentionally a report, not a correction: each row represents an
    independent consultation, so a changing diagnosis flag across visits is
    plausible clinical reality rather than a data error.

    Args:
        df: Input DataFrame.
        id_column: Column identifying the patient.
        flag_column: Boolean column to check for consistency.

    Returns:
        Number of patients with more than one distinct ``flag_column`` value.
    """
    n_inconsistent = int(df.groupby(id_column)[flag_column].nunique().gt(1).sum())
    logger.info(
        "%d patients have an inconsistent '%s' value across visits "
        "(left as-is: diagnosis may legitimately change over time)",
        n_inconsistent,
        flag_column,
    )
    return n_inconsistent


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full set of documented cleaning rules to the raw dataset.

    Args:
        df: Raw dataset, as returned by ``ml.data.load.load_raw_data``.

    Returns:
        A cleaned copy of ``df``: no other columns are dropped or altered
        beyond the explicit rules documented in this module.

    Raises:
        DataCleaningError: If any quality check fails.
    """
    logger.info("Cleaning dataset with %d rows", len(df))

    check_no_duplicate_rows(df)
    validate_service_time_positive(df)
    report_patient_label_inconsistency(df)

    cleaned = fill_missing_address(df)
    validate_categorical_domains(cleaned)

    logger.info("Cleaning complete: %d rows, %d columns", *cleaned.shape)
    return cleaned


if __name__ == "__main__":
    from ml.data.load import load_raw_data

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cleaned_df = clean_data(load_raw_data())
    print(cleaned_df.isna().sum())
