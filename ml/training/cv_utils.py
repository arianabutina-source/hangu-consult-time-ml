"""Group-aware cross-validation, applied within the training set only.

Repeat-visit patients (see EDA) make rows non-independent, so ordinary
``KFold``/``StratifiedKFold`` would leak the same patient across a fold's
train and validation portions. As with the Milestone 6 train/test split,
every CV splitter here is grouped by ``PATIENT_ID_COLUMN``:

    * :func:`get_classification_cv` -- ``StratifiedGroupKFold``, additionally
      preserving the ``IsLongConsultation`` class balance across folds.
    * :func:`get_regression_cv` -- ``GroupKFold`` (no stratification, since
      the regression target is continuous).

The held-out test set produced in Milestone 6 is never touched here; CV
folds are always carved out of the training set only.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, cross_validate

from ml.config import N_CV_FOLDS, RANDOM_STATE
from ml.data.split import GroupLeakageError

logger = logging.getLogger(__name__)

CLASSIFICATION_SCORING: list[str] = ["accuracy", "f1", "roc_auc"]
REGRESSION_SCORING: dict[str, str] = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}


def get_classification_cv(
    n_splits: int = N_CV_FOLDS, random_state: int = RANDOM_STATE
) -> StratifiedGroupKFold:
    """Group-aware, stratified CV splitter for the classification task."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def get_regression_cv(n_splits: int = N_CV_FOLDS, random_state: int = RANDOM_STATE) -> GroupKFold:
    """Group-aware CV splitter for the regression task."""
    return GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def assert_cv_has_no_group_leakage(
    cv: BaseEstimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series
) -> None:
    """Verify that no group id appears in both the train and validation part of any fold.

    Raises:
        GroupLeakageError: If any fold leaks a group across train/validation.
    """
    groups = pd.Series(groups).reset_index(drop=True)
    for fold_index, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])
        if overlap:
            raise GroupLeakageError(
                f"Fold {fold_index}: {len(overlap)} group(s) leak across "
                f"train/validation: {sorted(overlap)[:5]}..."
            )
    logger.info("No group leakage across %d CV folds", cv.get_n_splits())


def cross_validate_classification_pipeline(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Run group-aware, stratified CV and summarise accuracy/F1/ROC-AUC.

    Args:
        pipeline: An unfitted classification ``Pipeline``.
        X: Predictor DataFrame (training set only).
        y: Classification target.
        groups: Patient id per row, for grouping.
        n_splits: Number of CV folds.
        random_state: Seed for the fold shuffle.

    Returns:
        Dict keyed by metric name, each with ``{"mean": ..., "std": ...}``
        across folds.
    """
    cv = get_classification_cv(n_splits, random_state)
    results = cross_validate(pipeline, X, y, groups=groups, cv=cv, scoring=CLASSIFICATION_SCORING)

    summary = {
        metric: {"mean": float(scores.mean()), "std": float(scores.std())}
        for metric, scores in ((m, results[f"test_{m}"]) for m in CLASSIFICATION_SCORING)
    }
    logger.info("Classification CV results: %s", summary)
    return summary


def cross_validate_regression_pipeline(
    pipeline: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Run group-aware CV and summarise MAE/RMSE/R^2.

    Args:
        pipeline: An unfitted regression ``Pipeline``.
        X: Predictor DataFrame (training set only).
        y: Regression target (minutes).
        groups: Patient id per row, for grouping.
        n_splits: Number of CV folds.
        random_state: Seed for the fold shuffle.

    Returns:
        Dict keyed by metric name (``mae``, ``rmse``, ``r2``), each with
        ``{"mean": ..., "std": ...}`` across folds. MAE/RMSE are reported as
        positive errors (sklearn's negated scorers are flipped back).
    """
    cv = get_regression_cv(n_splits, random_state)
    results = cross_validate(
        pipeline, X, y, groups=groups, cv=cv, scoring=list(REGRESSION_SCORING.values())
    )

    summary = {}
    for metric, sklearn_name in REGRESSION_SCORING.items():
        scores = results[f"test_{sklearn_name}"]
        if metric in ("mae", "rmse"):
            scores = -scores
        summary[metric] = {"mean": float(scores.mean()), "std": float(np.std(scores))}
    logger.info("Regression CV results: %s", summary)
    return summary


if __name__ == "__main__":
    from ml.data.feature_selection import select_classification_data, select_regression_data
    from ml.data.split import get_train_test_split
    from ml.pipelines.classification_pipeline import build_classification_pipeline
    from ml.pipelines.regression_pipeline import build_regression_pipeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_df, _ = get_train_test_split()

    X_clf, y_clf = select_classification_data(train_df)
    clf_cv = get_classification_cv()
    assert_cv_has_no_group_leakage(clf_cv, X_clf, y_clf, train_df["ID"])
    clf_results = cross_validate_classification_pipeline(
        build_classification_pipeline(), X_clf, y_clf, train_df["ID"]
    )
    print("Classification CV (baseline LogisticRegression):", clf_results)

    X_reg, y_reg = select_regression_data(train_df)
    reg_cv = get_regression_cv()
    assert_cv_has_no_group_leakage(reg_cv, X_reg, y_reg, train_df["ID"])
    reg_results = cross_validate_regression_pipeline(
        build_regression_pipeline(), X_reg, y_reg, train_df["ID"]
    )
    print("Regression CV (baseline Ridge):", reg_results)
