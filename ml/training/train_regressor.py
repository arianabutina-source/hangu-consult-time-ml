"""Hyperparameter tuning and estimator comparison for the regression task.

Mirrors ``ml.training.train_classifier``. Compares every candidate in
``ml.training.tuning_spaces.REGRESSION_MODELS`` via ``RandomizedSearchCV``
over the group-aware CV from Milestone 9, then selects the best-performing
pipeline by mean CV score (MAE, minimised). No test-set data is touched here.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from ml.config import (
    N_CV_FOLDS,
    N_TUNING_ITER,
    RANDOM_STATE,
    REGRESSION_TUNING_RESULTS_PATH,
    REGRESSION_TUNING_SCORING,
)
from ml.pipelines.regression_pipeline import build_regression_pipeline
from ml.training.cv_utils import get_regression_cv
from ml.training.search import run_randomized_search, save_tuning_results, summarize_search
from ml.training.tuning_spaces import REGRESSION_MODELS

logger = logging.getLogger(__name__)


def tune_regressor(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_CV_FOLDS,
    n_iter: int = N_TUNING_ITER,
    random_state: int = RANDOM_STATE,
    scoring: str = REGRESSION_TUNING_SCORING,
) -> RandomizedSearchCV:
    """Run a randomized hyperparameter search for one candidate regressor.

    Args:
        model_name: Key into ``REGRESSION_MODELS``.
        X: Training predictors.
        y: Training regression target (minutes).
        groups: Patient id per row, for grouped CV.
        n_splits: Number of CV folds.
        n_iter: Number of parameter settings sampled.
        random_state: Seed for CV shuffling and parameter sampling.
        scoring: A scikit-learn scoring string (higher is better, so
            ``"neg_mean_absolute_error"`` by default).

    Returns:
        The fitted ``RandomizedSearchCV``.
    """
    spec = REGRESSION_MODELS[model_name]
    pipeline = build_regression_pipeline(spec["estimator"])
    cv = get_regression_cv(n_splits, random_state)
    return run_randomized_search(
        pipeline, spec["param_distributions"], cv, X, y, groups, scoring, n_iter, random_state
    )


def tune_all_regressors(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, **kwargs
) -> dict[str, RandomizedSearchCV]:
    """Tune every candidate estimator in ``REGRESSION_MODELS``.

    Returns:
        Dict mapping model name to its fitted ``RandomizedSearchCV``.
    """
    scoring = kwargs.get("scoring", REGRESSION_TUNING_SCORING)
    results: dict[str, RandomizedSearchCV] = {}
    for model_name in REGRESSION_MODELS:
        logger.info("Tuning regressor candidate: %s", model_name)
        results[model_name] = tune_regressor(model_name, X, y, groups, **kwargs)
        logger.info(
            "%s -- best CV %s: %.4f", model_name, scoring, results[model_name].best_score_
        )
    return results


def select_best_regressor(
    results: dict[str, RandomizedSearchCV],
) -> tuple[str, RandomizedSearchCV]:
    """Select the candidate with the highest mean CV score.

    Returns:
        ``(model_name, fitted_search)`` for the best candidate.
    """
    best_name = max(results, key=lambda name: results[name].best_score_)
    return best_name, results[best_name]


if __name__ == "__main__":
    from ml.data.feature_selection import select_regression_data
    from ml.data.split import get_train_test_split

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_df, _ = get_train_test_split()
    X_train, y_train = select_regression_data(train_df)

    all_results = tune_all_regressors(X_train, y_train, train_df["ID"])
    best_model_name, best_search = select_best_regressor(all_results)

    summary = {name: summarize_search(search) for name, search in all_results.items()}
    summary["best_model"] = best_model_name
    save_tuning_results(summary, REGRESSION_TUNING_RESULTS_PATH)

    print(
        f"Best regressor: {best_model_name} "
        f"(CV {REGRESSION_TUNING_SCORING}={best_search.best_score_:.4f})"
    )
    print("Best params:", best_search.best_params_)
