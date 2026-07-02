"""Serialising fitted pipelines and a reproducibility-focused metadata record.

``metadata.json`` exists so a serialized pipeline is never opaque: it
records exactly which library versions, code revision, feature schema,
hyperparameters, and test-set metrics produced it, so a future run (or the
FastAPI backend loading it) can be traced back to how it was trained.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from ml.config import PROJECT_ROOT, RANDOM_STATE

logger = logging.getLogger(__name__)


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Serialize a fitted pipeline to disk with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info("Saved pipeline to %s", path)


def load_pipeline(path: Path) -> Pipeline:
    """Load a pipeline previously saved with :func:`save_pipeline`."""
    return joblib.load(path)


def get_git_commit_hash(repo_dir: Path = PROJECT_ROOT) -> str | None:
    """Return the current git commit hash, or None if unavailable.

    Returns None (rather than raising) when the project is not inside a
    git repository, or git itself is not installed -- both are expected in
    some environments and should not block artifact export.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_library_versions() -> dict[str, str]:
    """Snapshot the versions of every library that affects reproducibility.

    ``xgboost`` is imported lazily here so this function still degrades
    gracefully (reports ``None`` instead of raising) in any environment
    that genuinely lacks it -- but note the deployed FastAPI backend *does*
    require it at runtime now: it loads every model in the ladder
    (``ml.training.tuning_spaces``) for side-by-side predictions, including
    an XGBClassifier/Regressor, not just the deployed RandomForest pipeline.
    """
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = None

    return {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost_version,
    }


def build_metadata(
    classification_metrics: dict[str, float],
    regression_metrics: dict[str, float],
    classification_tuning_results: dict[str, Any],
    regression_tuning_results: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the full reproducibility record for both serialized pipelines.

    Args:
        classification_metrics: Held-out test metrics from
            ``run_classification_evaluation`` (Milestone 11).
        regression_metrics: Held-out test metrics from
            ``run_regression_evaluation`` (Milestone 11).
        classification_tuning_results: The saved tuning-results summary
            (Milestone 10) for the classification task.
        regression_tuning_results: The saved tuning-results summary
            (Milestone 10) for the regression task.

    Returns:
        A JSON-serializable metadata dict.
    """
    from ml.config import CLASSIFICATION_TARGET_COLUMN, REGRESSION_TARGET_COLUMN
    from ml.data.feature_selection import get_classification_features, get_regression_features

    clf_best_model = classification_tuning_results["best_model"]
    reg_best_model = regression_tuning_results["best_model"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "library_versions": get_library_versions(),
        "random_state": RANDOM_STATE,
        "classification": {
            "target_column": CLASSIFICATION_TARGET_COLUMN,
            "features": get_classification_features(),
            "best_model": clf_best_model,
            "best_params": classification_tuning_results[clf_best_model]["best_params"],
            "cv_score": classification_tuning_results[clf_best_model]["best_score"],
            "test_metrics": classification_metrics,
        },
        "regression": {
            "target_column": REGRESSION_TARGET_COLUMN,
            "features": get_regression_features(),
            "best_model": reg_best_model,
            "best_params": regression_tuning_results[reg_best_model]["best_params"],
            "cv_score": regression_tuning_results[reg_best_model]["best_score"],
            "test_metrics": regression_metrics,
        },
    }
