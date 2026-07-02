"""Full model-ladder comparison on the held-out test set.

``ml.training.search.summarize_search`` (via ``tuning_results_*.json``) only
records each candidate's *cross-validated training* score, used purely to
select a winner. This module answers a different question: refit every
candidate in the ladder -- including the ``dummy`` baseline -- with its
tuned hyperparameters on the full training set, and score all of them on
the held-out test set exactly once, the same test set and the same rule
(evaluate once, never re-tune from it) as the single best-model evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _score(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    metrics_fn: Callable[..., dict[str, float]],
    use_predict_proba: bool,
) -> dict[str, float]:
    y_pred = pipeline.predict(X)
    if use_predict_proba:
        y_proba = pipeline.predict_proba(X)[:, 1]
        return metrics_fn(y, y_pred, y_proba)
    return metrics_fn(y, y_pred)


def build_test_leaderboard(
    tuning_results: dict[str, Any],
    models: dict[str, dict[str, Any]],
    pipeline_builder: Callable[[BaseEstimator], Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_fn: Callable[..., dict[str, float]],
    use_predict_proba: bool = False,
    include_train_metrics: bool = False,
) -> list[dict[str, Any]]:
    """Refit every tuned candidate on train and score it once on test.

    Args:
        tuning_results: A tuning-results dict as saved by
            ``save_tuning_results`` (must contain one entry per model name,
            each with ``"best_params"``, plus a ``"best_model"`` key that is
            skipped here).
        models: The candidate-estimator dict the tuning was run over (e.g.
            ``CLASSIFICATION_MODELS``), used for a fresh unfitted estimator.
        pipeline_builder: The task's pipeline-building function.
        X_train, y_train: Full training data.
        X_test, y_test: Held-out test data.
        metrics_fn: ``classification_metrics`` or ``regression_metrics``.
        use_predict_proba: If True, also pass predicted positive-class
            probabilities to ``metrics_fn`` (classification only).
        include_train_metrics: If True, also score each fitted pipeline on
            the training set it was just fit on, added as ``train_<metric>``
            keys alongside the test-set ``<metric>`` keys -- lets callers
            spot overfitting (a large train/test gap) per candidate, most
            relevant for an unconstrained decision tree.

    Returns:
        One dict per candidate model: ``{"model": name, **test_metrics}``
        (plus ``train_<metric>`` keys if requested), in the same order as
        ``tuning_results`` (unsorted -- callers sort by whichever metric is
        the primary ranking criterion for their task).
    """
    leaderboard: list[dict[str, Any]] = []
    for name, spec in tuning_results.items():
        if name == "best_model":
            continue

        pipeline = pipeline_builder(clone(models[name]["estimator"]))
        pipeline.set_params(**spec["best_params"])
        pipeline.fit(X_train, y_train)

        row_metrics = _score(pipeline, X_test, y_test, metrics_fn, use_predict_proba)
        row: dict[str, Any] = {"model": name}
        if include_train_metrics:
            train_metrics = _score(pipeline, X_train, y_train, metrics_fn, use_predict_proba)
            row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update(row_metrics)

        leaderboard.append(row)
        logger.info("Leaderboard -- %s test metrics: %s", name, row_metrics)

    return leaderboard


def evaluate_naive_model(
    name: str,
    estimator: BaseEstimator,
    pipeline_builder: Callable[[BaseEstimator], Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics_fn: Callable[..., dict[str, float]],
    use_predict_proba: bool = False,
) -> dict[str, Any]:
    """Fit an estimator with its *default* (untuned) hyperparameters and
    score it on both train and test, for an illustrative comparison against
    the tuned candidates -- e.g. an unconstrained ``DecisionTreeClassifier``
    (``max_depth=None``) is expected to fit the training set almost
    perfectly while generalising poorly, unlike its tuned counterpart in the
    main leaderboard.

    Returns:
        ``{"model": name, "train_<metric>": ..., "<metric>": ...}``.
    """
    pipeline = pipeline_builder(clone(estimator))
    pipeline.fit(X_train, y_train)

    train_metrics = _score(pipeline, X_train, y_train, metrics_fn, use_predict_proba)
    test_metrics = _score(pipeline, X_test, y_test, metrics_fn, use_predict_proba)

    row = {"model": name, **{f"train_{k}": v for k, v in train_metrics.items()}, **test_metrics}
    logger.info("Naive model -- %s train metrics: %s, test metrics: %s", name, train_metrics, test_metrics)
    return row


def sort_leaderboard(
    leaderboard: list[dict[str, Any]], metric: str, higher_is_better: bool = True
) -> list[dict[str, Any]]:
    """Return a new leaderboard sorted best-first by ``metric``."""
    return sorted(leaderboard, key=lambda row: row[metric], reverse=higher_is_better)


def save_leaderboard(leaderboard: list[dict[str, Any]], path: Path) -> None:
    """Write a leaderboard to disk as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    logger.info("Saved leaderboard to %s", path)


def load_leaderboard(path: Path) -> list[dict[str, Any]]:
    """Load a leaderboard previously written by :func:`save_leaderboard`."""
    with open(path) as f:
        return json.load(f)
