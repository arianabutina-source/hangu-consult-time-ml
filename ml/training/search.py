"""Shared ``RandomizedSearchCV`` execution and result-serialisation helpers.

Used by both ``ml.training.train_classifier`` and
``ml.training.train_regressor`` so the search/summarise/save logic is not
duplicated between the two nearly-identical task-specific trainers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import pandas as pd

from ml.config import N_TUNING_ITER, RANDOM_STATE

logger = logging.getLogger(__name__)


def run_randomized_search(
    pipeline: BaseEstimator,
    param_distributions: dict[str, Any],
    cv: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    scoring: str,
    n_iter: int = N_TUNING_ITER,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Fit a ``RandomizedSearchCV`` over ``pipeline`` using a group-aware ``cv``.

    Args:
        pipeline: Unfitted ``Pipeline`` (preprocessing + estimator).
        param_distributions: Search space, keyed with the ``"model__"`` prefix.
        cv: A group-aware CV splitter (see ``ml.training.cv_utils``).
        X: Training predictors.
        y: Training target.
        groups: Patient id per row, forwarded to ``cv.split``.
        scoring: A scikit-learn scoring string.
        n_iter: Number of parameter settings sampled.
        random_state: Seed for the parameter sampling.

    Returns:
        The fitted ``RandomizedSearchCV`` (refit on the full training data
        with the best found parameters).
    """
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search


def _json_safe(value: Any) -> Any:
    """Convert numpy scalar types (from scipy distributions) to native Python types."""
    return value.item() if hasattr(value, "item") else value


def summarize_search(search: RandomizedSearchCV) -> dict[str, Any]:
    """Extract the best score/params from a fitted search as JSON-safe values."""
    return {
        "best_score": float(search.best_score_),
        "best_params": {k: _json_safe(v) for k, v in search.best_params_.items()},
    }


def save_tuning_results(results: dict[str, Any], path: Path) -> None:
    """Write a tuning-results summary to disk as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved tuning results to %s", path)


def load_tuning_results(path: Path) -> dict[str, Any]:
    """Load a tuning-results summary previously written by :func:`save_tuning_results`."""
    with open(path) as f:
        return json.load(f)


def build_best_pipeline_from_results(
    results: dict[str, Any],
    models: dict[str, dict[str, Any]],
    pipeline_builder: Callable[[BaseEstimator], Pipeline],
) -> Pipeline:
    """Reconstruct the best-performing (unfitted) pipeline from a tuning-results summary.

    Args:
        results: A tuning-results dict as produced by
            ``tune_all_classifiers``/``tune_all_regressors`` and saved via
            :func:`save_tuning_results` (must contain ``"best_model"`` and,
            for that model, ``"best_params"``).
        models: The candidate-estimator dict the tuning was run over (e.g.
            ``CLASSIFICATION_MODELS``), used to look up a fresh, unfitted
            estimator instance for the winning model.
        pipeline_builder: The task's pipeline-building function (e.g.
            ``build_classification_pipeline``).

    Returns:
        An unfitted ``Pipeline`` with the winning estimator and hyperparameters set.
    """
    best_name = results["best_model"]
    best_params = results[best_name]["best_params"]

    # Clone the shared estimator instance so this pipeline owns an
    # independent, unfitted copy (module-level estimators in tuning_spaces
    # are otherwise reused by reference across calls).
    pipeline = pipeline_builder(clone(models[best_name]["estimator"]))
    pipeline.set_params(**best_params)
    return pipeline
