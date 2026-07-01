"""Feature importance and (optional) SHAP explainability for a fitted pipeline.

Works with any of the fitted pipelines produced by
``ml.evaluation.evaluate_classifier``/``evaluate_regressor``:

    * Tree-based estimators (``RandomForest*``, ``XGB*``) expose native
      ``feature_importances_`` -- used directly, and also SHAP-explainable
      via ``shap.TreeExplainer``.
    * Linear estimators (``LogisticRegression``, ``Ridge``) expose
      ``coef_`` -- absolute coefficient magnitude is used as the importance
      ranking.
    * Anything else falls back to permutation importance, which requires a
      labelled dataset (``X``, ``y``) to score against.

SHAP is an optional, heavier dependency; it is imported lazily inside
:func:`compute_shap_values` so the rest of this module works even if SHAP is
unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml.config import RANDOM_STATE, SHAP_MAX_SAMPLES

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Feature names produced by the pipeline's ``"preprocess"`` step."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def get_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    scoring: str | None = None,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Extract a ranked feature-importance table from a fitted pipeline.

    Args:
        pipeline: A pipeline already fitted on training data.
        X: Predictors, required only for the permutation-importance fallback.
        y: Target, required only for the permutation-importance fallback.
        scoring: Scikit-learn scoring string for the permutation fallback.
        n_repeats: Number of shuffles for the permutation fallback.
        random_state: Seed for the permutation fallback.

    Returns:
        DataFrame with ``feature`` and ``importance`` columns, sorted
        descending by importance.

    Raises:
        ValueError: If neither a native importance/coefficient is available
            nor ``X``/``y`` were provided for the permutation fallback.
    """
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        feature_names = get_feature_names(pipeline)
        importances = np.asarray(model.feature_importances_)
        method = "impurity-based (native)"
    elif hasattr(model, "coef_"):
        feature_names = get_feature_names(pipeline)
        importances = np.abs(np.ravel(model.coef_))
        method = "absolute coefficient magnitude"
    elif X is not None and y is not None:
        # permutation_importance shuffles columns of the *raw* input X (it
        # scores the whole pipeline, preprocessing included), so importances
        # align with X.columns, not the one-hot-encoded feature names.
        feature_names = list(X.columns)
        result = permutation_importance(
            pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state
        )
        importances = result.importances_mean
        method = "permutation importance"
    else:
        raise ValueError(
            f"Estimator {type(model).__name__} has no native importance/coefficient "
            "attribute; pass X and y to fall back to permutation importance."
        )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Computed feature importance via %s for %d features", method, len(importance_df))
    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 15,
    title: str = "Feature Importance",
) -> Path:
    """Save a horizontal bar chart of the top-``top_n`` most important features."""
    subset = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x="importance", y="feature", data=subset, hue="feature", legend=False, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", output_path)
    return output_path


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = SHAP_MAX_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a fitted pipeline's tree-based estimator.

    Only supported for estimators exposing ``feature_importances_``
    (``RandomForest*``, ``XGB*``), via ``shap.TreeExplainer``.

    Args:
        pipeline: A pipeline already fitted on training data.
        X: Predictors to explain (a random subsample of up to
            ``max_samples`` rows is used, for speed).
        max_samples: Maximum number of rows to explain.
        random_state: Seed for subsampling.

    Returns:
        ``(shap_values, X_transformed)``: SHAP values with shape
        ``(n_samples, n_features)`` -- for binary classifiers the positive
        class's contribution is returned -- and the corresponding
        transformed (one-hot-encoded) feature DataFrame used to compute them.

    Raises:
        TypeError: If the estimator is not tree-based.
    """
    import shap  # optional, heavier dependency; imported lazily

    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"SHAP TreeExplainer requires a tree-based estimator; got {type(model).__name__}"
        )

    X_sample = X if len(X) <= max_samples else X.sample(max_samples, random_state=random_state)

    preprocess = pipeline.named_steps["preprocess"]
    feature_names = get_feature_names(pipeline)
    X_transformed = pd.DataFrame(
        preprocess.transform(X_sample), columns=feature_names, index=X_sample.index
    )

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_transformed))

    if shap_values.ndim == 3:
        # (n_samples, n_features, n_classes) for binary classifiers -- keep
        # the positive class's contribution.
        shap_values = shap_values[:, :, -1]

    return shap_values, X_transformed


def plot_shap_summary(
    shap_values: np.ndarray,
    X_transformed: pd.DataFrame,
    output_path: Path,
    title: str = "SHAP Summary",
) -> Path:
    """Save a SHAP beeswarm summary plot."""
    import shap  # optional, heavier dependency; imported lazily

    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_transformed, show=False)
    plt.title(title)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary plot to %s", output_path)
    return output_path


if __name__ == "__main__":
    import logging as _logging

    from ml.config import (
        FEATURE_IMPORTANCE_CLASSIFICATION_FIGURE_PATH,
        FEATURE_IMPORTANCE_REGRESSION_FIGURE_PATH,
        SHAP_SUMMARY_CLASSIFICATION_FIGURE_PATH,
        SHAP_SUMMARY_REGRESSION_FIGURE_PATH,
    )
    from ml.data.feature_selection import select_classification_data, select_regression_data
    from ml.data.split import get_train_test_split
    from ml.evaluation.evaluate_classifier import build_tuned_classification_pipeline
    from ml.evaluation.evaluate_regressor import build_tuned_regression_pipeline

    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_df, _ = get_train_test_split()

    X_clf, y_clf = select_classification_data(train_df)
    clf_pipeline = build_tuned_classification_pipeline()
    clf_pipeline.fit(X_clf, y_clf)
    clf_importance = get_feature_importance(clf_pipeline)
    plot_feature_importance(
        clf_importance,
        FEATURE_IMPORTANCE_CLASSIFICATION_FIGURE_PATH,
        title="Feature Importance -- Classification (Long Consultation)",
    )
    try:
        shap_values, X_t = compute_shap_values(clf_pipeline, X_clf)
        plot_shap_summary(
            shap_values, X_t, SHAP_SUMMARY_CLASSIFICATION_FIGURE_PATH,
            title="SHAP Summary -- Classification",
        )
    except TypeError as exc:
        print(f"Skipping classification SHAP summary: {exc}")

    X_reg, y_reg = select_regression_data(train_df)
    reg_pipeline = build_tuned_regression_pipeline()
    reg_pipeline.fit(X_reg, y_reg)
    reg_importance = get_feature_importance(reg_pipeline)
    plot_feature_importance(
        reg_importance,
        FEATURE_IMPORTANCE_REGRESSION_FIGURE_PATH,
        title="Feature Importance -- Regression (Consultation Duration)",
    )
    try:
        shap_values, X_t = compute_shap_values(reg_pipeline, X_reg)
        plot_shap_summary(
            shap_values, X_t, SHAP_SUMMARY_REGRESSION_FIGURE_PATH,
            title="SHAP Summary -- Regression",
        )
    except TypeError as exc:
        print(f"Skipping regression SHAP summary: {exc}")

    print("Top classification features:")
    print(clf_importance.head(10).to_string(index=False))
    print("\nTop regression features:")
    print(reg_importance.head(10).to_string(index=False))
