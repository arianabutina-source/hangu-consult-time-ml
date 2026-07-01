"""Grouped, stratified, seeded train/test split.

Rows are not independent observations: many patients (identified by
``PATIENT_ID_COLUMN``) have multiple consultation records, so a plain random
split would let the same patient appear in both train and test -- a form of
leakage across the split itself. ``StratifiedGroupKFold`` is used instead so
that:

    * every row for a given patient falls entirely on one side of the split
      (grouped), and
    * the classification target's class balance is preserved on both sides
      (stratified).

The split is computed once and persisted to ``data/processed/{train,test}.csv``;
:func:`get_train_test_split` loads the persisted files on subsequent calls
instead of recomputing them, per the project's reproducibility requirement.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml.config import (
    CLASSIFICATION_TARGET_COLUMN,
    N_SPLITS_FOR_HOLDOUT,
    PATIENT_ID_COLUMN,
    RANDOM_STATE,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)

logger = logging.getLogger(__name__)


class GroupLeakageError(ValueError):
    """Raised when the same group (patient) appears in both train and test."""


def assert_no_group_leakage(
    train_df: pd.DataFrame, test_df: pd.DataFrame, group_column: str = PATIENT_ID_COLUMN
) -> None:
    """Verify no group id is present in both splits.

    Raises:
        GroupLeakageError: If any group id appears in both ``train_df`` and
            ``test_df``.
    """
    overlap = set(train_df[group_column]) & set(test_df[group_column])
    if overlap:
        raise GroupLeakageError(
            f"{len(overlap)} patient id(s) appear in both train and test: "
            f"{sorted(overlap)[:5]}..."
        )


def grouped_stratified_split(
    df: pd.DataFrame,
    group_column: str = PATIENT_ID_COLUMN,
    stratify_column: str = CLASSIFICATION_TARGET_COLUMN,
    n_splits: int = N_SPLITS_FOR_HOLDOUT,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/test sets, grouped by patient and stratified by target.

    One fold of a ``StratifiedGroupKFold`` is held out as the test set, giving
    an approximate test size of ``1 / n_splits``.

    Args:
        df: Engineered dataset (output of ``ml.data.features.engineer_features``).
        group_column: Column identifying the patient (grouping key).
        stratify_column: Binary column whose class balance is preserved.
        n_splits: Number of folds; the held-out test size is ``1/n_splits``.
        random_state: Seed for the shuffle.

    Returns:
        ``(train_df, test_df)``, each with a fresh integer index.

    Raises:
        GroupLeakageError: If the resulting split accidentally shares a
            patient between train and test (would indicate a scikit-learn
            contract violation, not expected in practice).
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(
        splitter.split(df, y=df[stratify_column], groups=df[group_column])
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    assert_no_group_leakage(train_df, test_df, group_column)

    logger.info(
        "Split into %d train rows (%d patients) and %d test rows (%d patients)",
        len(train_df),
        train_df[group_column].nunique(),
        len(test_df),
        test_df[group_column].nunique(),
    )
    logger.info(
        "Class balance -- train: %.1f%% positive, test: %.1f%% positive",
        100 * train_df[stratify_column].mean(),
        100 * test_df[stratify_column].mean(),
    )
    return train_df, test_df


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
) -> None:
    """Persist the train/test split to CSV, creating parent directories as needed."""
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved split to %s and %s", train_path, test_path)


def create_train_test_split(
    n_splits: int = N_SPLITS_FOR_HOLDOUT, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline (load -> clean -> engineer -> split) from scratch."""
    from ml.data.clean import clean_data
    from ml.data.features import engineer_features
    from ml.data.load import load_raw_data

    engineered = engineer_features(clean_data(load_raw_data()))
    return grouped_stratified_split(engineered, n_splits=n_splits, random_state=random_state)


def get_train_test_split(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted train/test split, computing and saving it once if absent.

    Args:
        train_path: Location of the persisted training CSV.
        test_path: Location of the persisted test CSV.
        force_recompute: If True, recompute and overwrite the persisted split
            even if it already exists.

    Returns:
        ``(train_df, test_df)`` loaded from disk.
    """
    if not force_recompute and train_path.exists() and test_path.exists():
        logger.info("Loading existing train/test split from %s and %s", train_path, test_path)
        return pd.read_csv(train_path), pd.read_csv(test_path)

    logger.info("No persisted split found (or force_recompute=True); computing one")
    train_df, test_df = create_train_test_split()
    save_split(train_df, test_df, train_path, test_path)
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train, test = get_train_test_split()
    print(f"train: {train.shape}, test: {test.shape}")
