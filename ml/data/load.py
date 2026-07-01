"""Raw data ingestion for the Hangu outpatient consultation dataset.

This module is intentionally limited to *loading and schema validation*.
No cleaning, imputation, or feature engineering happens here -- those are
handled by later, dedicated pipeline stages so that each step of the
project remains independently testable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml.config import (
    EXPECTED_BOOLEAN_COLUMNS,
    EXPECTED_COLUMNS,
    EXPECTED_N_COLUMNS,
    EXPECTED_N_ROWS,
    EXPECTED_NUMERIC_COLUMNS,
    RAW_DATA_PATH,
)

logger = logging.getLogger(__name__)


class DatasetSchemaError(ValueError):
    """Raised when the loaded CSV does not match the expected raw schema."""


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate shape, column names, and key dtypes against expectations.

    Raises:
        DatasetSchemaError: If the DataFrame does not match the schema
            established during dataset inspection.
    """
    if df.shape != (EXPECTED_N_ROWS, EXPECTED_N_COLUMNS):
        raise DatasetSchemaError(
            f"Unexpected shape {df.shape}; expected "
            f"({EXPECTED_N_ROWS}, {EXPECTED_N_COLUMNS})."
        )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise DatasetSchemaError(
            f"Unexpected columns {list(df.columns)}; expected {EXPECTED_COLUMNS}."
        )

    non_numeric = [
        col for col in EXPECTED_NUMERIC_COLUMNS if not pd.api.types.is_integer_dtype(df[col])
    ]
    if non_numeric:
        raise DatasetSchemaError(
            f"Expected integer dtype for {non_numeric}, got "
            f"{[df[col].dtype for col in non_numeric]}."
        )

    non_boolean = [
        col for col in EXPECTED_BOOLEAN_COLUMNS if not pd.api.types.is_bool_dtype(df[col])
    ]
    if non_boolean:
        raise DatasetSchemaError(
            f"Expected boolean dtype for {non_boolean}, got "
            f"{[df[col].dtype for col in non_boolean]}."
        )

    logger.info("Schema validation passed: shape=%s, columns=%d", df.shape, df.shape[1])


def load_raw_data(path: Path | str | None = None, validate: bool = True) -> pd.DataFrame:
    """Load the raw Hangu outpatient consultation dataset from CSV.

    Args:
        path: Location of the CSV file. Defaults to ``config.RAW_DATA_PATH``.
        validate: If True (default), validate shape/columns/dtypes against
            the known schema before returning.

    Returns:
        The raw dataset, unmodified aside from pandas' own type inference.

    Raises:
        FileNotFoundError: If the CSV file does not exist at ``path``.
        DatasetSchemaError: If ``validate`` is True and the schema does not
            match expectations.
    """
    resolved_path = Path(path) if path is not None else RAW_DATA_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at '{resolved_path}'. Expected the Hangu "
            "outpatient CSV to be placed there before loading."
        )

    logger.info("Loading raw dataset from %s", resolved_path)
    df = pd.read_csv(resolved_path)
    logger.info("Loaded %d rows and %d columns", *df.shape)

    if validate:
        _validate_schema(df)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_raw_data()
    print(data.info())
