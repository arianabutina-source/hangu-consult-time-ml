"""Tests for ml.data.split: grouped, stratified, seeded train/test splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data.clean import clean_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import (
    GroupLeakageError,
    assert_no_group_leakage,
    get_train_test_split,
    grouped_stratified_split,
    save_split,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_patients = 60
    rows = []
    for patient_id in range(n_patients):
        n_visits = rng.integers(1, 4)
        label = bool(patient_id % 2)  # deterministic-ish balance across patients
        for _ in range(n_visits):
            rows.append({"ID": f"P{patient_id}", "IsLongConsultation": label})
    return pd.DataFrame(rows)


def test_grouped_stratified_split_has_no_group_leakage(synthetic_df: pd.DataFrame) -> None:
    train_df, test_df = grouped_stratified_split(synthetic_df, n_splits=5, random_state=42)
    assert_no_group_leakage(train_df, test_df)  # should not raise


def test_grouped_stratified_split_covers_all_rows(synthetic_df: pd.DataFrame) -> None:
    train_df, test_df = grouped_stratified_split(synthetic_df, n_splits=5, random_state=42)
    assert len(train_df) + len(test_df) == len(synthetic_df)


def test_grouped_stratified_split_is_deterministic(synthetic_df: pd.DataFrame) -> None:
    train_a, test_a = grouped_stratified_split(synthetic_df, n_splits=5, random_state=42)
    train_b, test_b = grouped_stratified_split(synthetic_df, n_splits=5, random_state=42)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)


def test_grouped_stratified_split_approximate_test_size(synthetic_df: pd.DataFrame) -> None:
    _, test_df = grouped_stratified_split(synthetic_df, n_splits=5, random_state=42)
    test_fraction = len(test_df) / len(synthetic_df)
    assert 0.1 < test_fraction < 0.3


def test_assert_no_group_leakage_raises_on_overlap() -> None:
    train_df = pd.DataFrame({"ID": ["A", "B"]})
    test_df = pd.DataFrame({"ID": ["B", "C"]})
    with pytest.raises(GroupLeakageError):
        assert_no_group_leakage(train_df, test_df)


def test_save_split_round_trip(tmp_path: Path) -> None:
    train_df = pd.DataFrame({"ID": ["A", "B"], "value": [1, 2]})
    test_df = pd.DataFrame({"ID": ["C"], "value": [3]})
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    save_split(train_df, test_df, train_path, test_path)
    loaded_train = pd.read_csv(train_path)
    loaded_test = pd.read_csv(test_path)

    pd.testing.assert_frame_equal(loaded_train, train_df)
    pd.testing.assert_frame_equal(loaded_test, test_df)


def test_get_train_test_split_computes_and_persists_once(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    train_df, test_df = get_train_test_split(train_path=train_path, test_path=test_path)
    assert train_path.exists()
    assert test_path.exists()

    # Second call must load from disk, not recompute (same content either way).
    train_again, test_again = get_train_test_split(train_path=train_path, test_path=test_path)
    pd.testing.assert_frame_equal(train_df, train_again)
    pd.testing.assert_frame_equal(test_df, test_again)


def test_split_on_real_dataset_has_no_patient_leakage() -> None:
    """Integration test: the actual repeat-visit structure must not leak across the split."""
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, test_df = grouped_stratified_split(engineered)

    assert_no_group_leakage(train_df, test_df)
    assert len(train_df) + len(test_df) == len(engineered)

    train_balance = train_df["IsLongConsultation"].mean()
    test_balance = test_df["IsLongConsultation"].mean()
    assert abs(train_balance - test_balance) < 0.05
