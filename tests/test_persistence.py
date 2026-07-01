"""Tests for ml.persistence: pipeline serialization and metadata assembly."""

from pathlib import Path

import numpy as np
import pytest

from ml.config import CLASSIFICATION_TUNING_RESULTS_PATH
from ml.data.clean import clean_data
from ml.data.feature_selection import select_classification_data
from ml.data.features import engineer_features
from ml.data.load import load_raw_data
from ml.data.split import grouped_stratified_split
from ml.persistence import (
    build_metadata,
    get_git_commit_hash,
    get_library_versions,
    load_pipeline,
    save_pipeline,
)
from ml.pipelines.classification_pipeline import build_classification_pipeline


@pytest.fixture(scope="module")
def fitted_pipeline_and_data() -> tuple:
    engineered = engineer_features(clean_data(load_raw_data()))
    train_df, test_df = grouped_stratified_split(engineered)
    X_train, y_train = select_classification_data(train_df)
    X_test, _ = select_classification_data(test_df)

    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline, X_test


def test_save_and_load_pipeline_round_trip(
    fitted_pipeline_and_data: tuple, tmp_path: Path
) -> None:
    pipeline, X_test = fitted_pipeline_and_data
    path = tmp_path / "pipeline.joblib"

    save_pipeline(pipeline, path)
    assert path.exists()

    loaded = load_pipeline(path)
    np.testing.assert_array_equal(loaded.predict(X_test), pipeline.predict(X_test))


def test_save_pipeline_creates_parent_directories(
    fitted_pipeline_and_data: tuple, tmp_path: Path
) -> None:
    pipeline, _ = fitted_pipeline_and_data
    path = tmp_path / "nested" / "dir" / "pipeline.joblib"
    save_pipeline(pipeline, path)
    assert path.exists()


def test_get_git_commit_hash_returns_none_or_valid_sha(tmp_path: Path) -> None:
    # This project is not a git repository, so this should return None. The
    # assertion also tolerates a valid hash, so the test still passes if run
    # inside a git-initialised copy of the project.
    commit = get_git_commit_hash()
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)


def test_get_git_commit_hash_handles_nonexistent_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    assert get_git_commit_hash(missing_dir) is None


def test_get_library_versions_contains_expected_keys() -> None:
    versions = get_library_versions()
    assert set(versions.keys()) == {"python", "pandas", "numpy", "scikit_learn", "xgboost"}
    # xgboost is optional (training-only, not required for inference); every
    # other key is always populated.
    required = {"python", "pandas", "numpy", "scikit_learn"}
    assert all(isinstance(versions[k], str) and versions[k] for k in required)
    assert versions["xgboost"] is None or isinstance(versions["xgboost"], str)


def test_get_library_versions_reports_none_when_xgboost_unavailable(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("simulated: xgboost not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    versions = get_library_versions()
    assert versions["xgboost"] is None


def test_build_metadata_assembles_expected_structure() -> None:
    clf_metrics = {"accuracy": 0.6, "roc_auc": 0.63}
    reg_metrics = {"mae": 4.4, "rmse": 6.0, "r2": 0.07}
    clf_tuning_results = {
        "logistic_regression": {"best_score": 0.6, "best_params": {"model__C": 1.0}},
        "best_model": "logistic_regression",
    }
    reg_tuning_results = {
        "ridge": {"best_score": -4.4, "best_params": {"model__alpha": 2.0}},
        "best_model": "ridge",
    }

    metadata = build_metadata(clf_metrics, reg_metrics, clf_tuning_results, reg_tuning_results)

    assert "generated_at" in metadata
    assert "git_commit" in metadata
    assert metadata["random_state"] == 42
    assert metadata["classification"]["best_model"] == "logistic_regression"
    assert metadata["classification"]["best_params"] == {"model__C": 1.0}
    assert metadata["classification"]["test_metrics"] == clf_metrics
    assert metadata["regression"]["best_model"] == "ridge"
    assert metadata["regression"]["test_metrics"] == reg_metrics
    assert len(metadata["classification"]["features"]) == 12
    assert len(metadata["regression"]["features"]) == 12


def test_build_metadata_is_json_serializable() -> None:
    import json

    clf_tuning_results = {
        "logistic_regression": {"best_score": 0.6, "best_params": {"model__C": 1.0}},
        "best_model": "logistic_regression",
    }
    reg_tuning_results = {
        "ridge": {"best_score": -4.4, "best_params": {"model__alpha": 2.0}},
        "best_model": "ridge",
    }
    metadata = build_metadata({"accuracy": 0.6}, {"mae": 4.4}, clf_tuning_results, reg_tuning_results)
    json.dumps(metadata)  # must not raise


@pytest.mark.skipif(
    not CLASSIFICATION_TUNING_RESULTS_PATH.exists(), reason="tuning results not yet generated"
)
def test_export_metadata_script_runs_end_to_end(tmp_path: Path) -> None:
    """Integration test for the actual scripts.export_metadata CLI entry point."""
    from ml.config import CLASSIFIER_PIPELINE_PATH, METADATA_PATH, REGRESSOR_PIPELINE_PATH
    from scripts.export_metadata import main

    main()

    assert CLASSIFIER_PIPELINE_PATH.exists()
    assert REGRESSOR_PIPELINE_PATH.exists()
    assert METADATA_PATH.exists()

    loaded_clf = load_pipeline(CLASSIFIER_PIPELINE_PATH)
    assert hasattr(loaded_clf, "predict")
