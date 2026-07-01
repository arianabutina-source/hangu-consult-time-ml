"""Export small, static JSON datasets for the React dashboard (Milestone 15).

Reuses the tested Milestone 11/12 evaluation functions -- nothing here
refits or reruns tuning; it only reformats already-computed results
(confusion matrix, ROC curve, feature importances, residuals) as compact
JSON so the frontend can render interactive charts without a live backend
call or any Python dependency at request time.

Run as: ``python -m scripts.export_dashboard_data``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from ml.config import ARTIFACTS_DIR
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.split import get_train_test_split
from ml.evaluation.evaluate_classifier import build_tuned_classification_pipeline
from ml.evaluation.evaluate_regressor import build_tuned_regression_pipeline
from ml.evaluation.explainability import get_feature_importance

logger = logging.getLogger(__name__)

DASHBOARD_DIR: Path = ARTIFACTS_DIR / "dashboard"
MAX_RESIDUAL_POINTS = 300
TOP_N_FEATURES = 10


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote %s", path)


def export_confusion_matrix(pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred).tolist()
    _write_json(
        {"labels": ["Short", "Long"], "matrix": matrix},
        DASHBOARD_DIR / "confusion_matrix.json",
    )


def export_roc_curve(pipeline, X_test, y_test) -> None:
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    # Downsample to keep the JSON small; ROC curves are smooth enough that
    # ~50 evenly spaced points look identical to the full curve.
    n_points = min(50, len(fpr))
    indices = np.linspace(0, len(fpr) - 1, n_points).astype(int)
    points = [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in indices]
    _write_json(points, DASHBOARD_DIR / "roc_curve.json")


def export_feature_importance(importance_df, filename: str) -> None:
    top = importance_df.head(TOP_N_FEATURES)
    records = [
        {"feature": row.feature, "importance": float(row.importance)}
        for row in top.itertuples()
    ]
    _write_json(records, DASHBOARD_DIR / filename)


def export_residuals(pipeline, X_test, y_test) -> None:
    y_pred = pipeline.predict(X_test)
    residuals = np.asarray(y_test) - np.asarray(y_pred)

    n = len(y_pred)
    if n > MAX_RESIDUAL_POINTS:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(n, size=MAX_RESIDUAL_POINTS, replace=False)
    else:
        sample_idx = np.arange(n)

    points = [
        {"predicted": float(y_pred[i]), "residual": float(residuals[i])} for i in sample_idx
    ]
    _write_json(points, DASHBOARD_DIR / "residuals.json")


def main() -> None:
    train_df, test_df = get_train_test_split()

    clf_pipeline = build_tuned_classification_pipeline()
    X_clf_train, y_clf_train = select_classification_data(train_df)
    X_clf_test, y_clf_test = select_classification_data(test_df)
    clf_pipeline.fit(X_clf_train, y_clf_train)

    export_confusion_matrix(clf_pipeline, X_clf_test, y_clf_test)
    export_roc_curve(clf_pipeline, X_clf_test, y_clf_test)
    export_feature_importance(
        get_feature_importance(clf_pipeline), "feature_importance_classification.json"
    )

    reg_pipeline = build_tuned_regression_pipeline()
    X_reg_train, y_reg_train = select_regression_data(train_df)
    X_reg_test, y_reg_test = select_regression_data(test_df)
    reg_pipeline.fit(X_reg_train, y_reg_train)

    export_residuals(reg_pipeline, X_reg_test, y_reg_test)
    export_feature_importance(
        get_feature_importance(reg_pipeline), "feature_importance_regression.json"
    )

    print(f"Dashboard data exported to {DASHBOARD_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
