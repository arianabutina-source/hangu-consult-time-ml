"""Final predictor selection for the classification and regression tasks.

Every exclusion below is a documented, evidence-based decision (from the
Milestone 2 EDA and Milestone 4 target derivation), not a default. Predictor
lists are exposed as functions (not import-time constants) so a single
source of truth can be unit-tested against the actual engineered dataset.

Excluded from *both* tasks' predictors:
    * ``ID`` -- patient identifier, used only for grouped train/test
      splitting (Milestone 6); including it as a predictor would let a
      model memorise per-patient outcomes.
    * ``Session`` -- a near-unique sequential index (381 distinct values
      across 6,637 rows, see EDA). It behaves as a proxy for chronological
      order/collection sequence rather than a genuine, generalisable driver
      of consultation duration, so it is excluded to avoid the model
      exploiting a spurious ordering effect.
    * ``StartTime``, ``PayTime`` -- raw clock-time strings. EDA showed
      ``PayTime - StartTime`` does not reliably reconstruct ``ServTime``,
      so these columns are unreliable and are not used to build any
      feature.
    * ``ServTime`` -- the raw target source (seconds); direct leakage for
      both tasks.
    * ``ServTime_minutes`` -- the regression target itself. It also
      perfectly determines the classification label via the frozen
      median threshold, so it is target leakage for classification too.
    * ``IsLongConsultation`` -- the classification target itself. It also
      leaks a quantised version of the regression target, so it is
      excluded from regression predictors as well.

Both tasks currently share an identical predictor list: no feature engineered
so far is task-specific, and every exclusion above applies symmetrically.
"""

from __future__ import annotations

import logging

import pandas as pd

from ml.config import (
    CLASSIFICATION_TARGET_COLUMN,
    HAS_CANCER_DIAGNOSIS_COLUMN,
    IS_PUBLIC_HOLIDAY_MAKEUP_COLUMN,
    IS_REPEAT_VISIT_COLUMN,
    PATIENT_ID_COLUMN,
    RAW_SERVICE_TIME_COLUMN,
    SERVICE_TIME_MINUTES_COLUMN,
    TIMESTAMP_COLUMNS,
    VISIT_NO_COLUMN,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Predictor groups (shared by both tasks)
# --------------------------------------------------------------------------- #
NUMERIC_FEATURES: list[str] = [VISIT_NO_COLUMN]

BOOLEAN_FEATURES: list[str] = [
    "WorkingDay",
    "M.Cancer",
    "S.Cancer",
    IS_REPEAT_VISIT_COLUMN,
    IS_PUBLIC_HOLIDAY_MAKEUP_COLUMN,
    HAS_CANCER_DIAGNOSIS_COLUMN,
]

CATEGORICAL_FEATURES: list[str] = ["Month", "DayOfWeek", "AM_PM", "Gender", "Address"]

# --------------------------------------------------------------------------- #
# Excluded columns, with reasons (kept close to the lists above so the two
# stay consistent, and so tests can assert no overlap).
# --------------------------------------------------------------------------- #
EXCLUDED_COLUMNS: dict[str, str] = {
    PATIENT_ID_COLUMN: "patient identifier; used only for grouped splitting",
    "Session": "near-unique sequential index; proxy for chronological order",
    **{col: "raw timestamp, unreliable per EDA, not used to build features" for col in TIMESTAMP_COLUMNS},
    RAW_SERVICE_TIME_COLUMN: "raw target source (seconds); direct leakage",
    SERVICE_TIME_MINUTES_COLUMN: "regression target; determines classification label",
    CLASSIFICATION_TARGET_COLUMN: "classification target; leaks quantised regression target",
}


def get_classification_features() -> list[str]:
    """Final predictor list for the classification task (target: ``IsLongConsultation``)."""
    return NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def get_regression_features() -> list[str]:
    """Final predictor list for the regression task (target: ``ServTime_minutes``)."""
    return NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES


def _validate_no_leakage(feature_list: list[str], target_column: str) -> None:
    """Guard against a leakage column silently ending up in a predictor list."""
    leaking = (set(feature_list) & set(EXCLUDED_COLUMNS)) | ({target_column} & set(feature_list))
    if leaking:
        raise ValueError(f"Leakage-prone column(s) found in predictor list: {leaking}")


def select_classification_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered DataFrame into classification predictors and target.

    Args:
        df: Output of ``ml.data.features.engineer_features``.

    Returns:
        Tuple of ``(X, y)`` where ``X`` contains only
        :func:`get_classification_features` and ``y`` is
        ``IsLongConsultation``.

    Raises:
        ValueError: If a leakage-prone column would end up in ``X``.
    """
    features = get_classification_features()
    _validate_no_leakage(features, CLASSIFICATION_TARGET_COLUMN)
    logger.info("Selected %d classification predictors", len(features))
    return df[features].copy(), df[CLASSIFICATION_TARGET_COLUMN].copy()


def select_regression_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered DataFrame into regression predictors and target.

    Args:
        df: Output of ``ml.data.features.engineer_features``.

    Returns:
        Tuple of ``(X, y)`` where ``X`` contains only
        :func:`get_regression_features` and ``y`` is ``ServTime_minutes``.

    Raises:
        ValueError: If a leakage-prone column would end up in ``X``.
    """
    features = get_regression_features()
    _validate_no_leakage(features, SERVICE_TIME_MINUTES_COLUMN)
    logger.info("Selected %d regression predictors", len(features))
    return df[features].copy(), df[SERVICE_TIME_MINUTES_COLUMN].copy()


if __name__ == "__main__":
    from ml.data.clean import clean_data
    from ml.data.features import engineer_features
    from ml.data.load import load_raw_data

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engineered = engineer_features(clean_data(load_raw_data()))

    X_clf, y_clf = select_classification_data(engineered)
    X_reg, y_reg = select_regression_data(engineered)
    print("Classification features:", list(X_clf.columns))
    print("Regression features:", list(X_reg.columns))
    print("Excluded columns:", EXCLUDED_COLUMNS)
