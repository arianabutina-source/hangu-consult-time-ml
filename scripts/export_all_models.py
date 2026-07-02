"""Persist every tuned candidate pipeline, not just the CV-selected winner.

``scripts.export_metadata`` only ever serializes the single best model per
task (what the API actually deploys). This script additionally saves every
other candidate in the model ladder -- ``dummy``, the linear model,
``decision_tree``, and whichever wasn't selected between ``random_forest``
and ``xgboost`` -- each refit on the full training set with its own tuned
hyperparameters, so the API can run a request through all of them and the
frontend can show every model's prediction side by side, not just the
deployed one.

Run as: ``python -m scripts.export_all_models`` (after
``train_classifier``/``train_regressor`` have produced tuning results).
"""

from __future__ import annotations

import logging

from sklearn.base import clone
from sklearn.pipeline import Pipeline

from ml.config import (
    CLASSIFICATION_MODELS_DIR,
    CLASSIFICATION_TUNING_RESULTS_PATH,
    REGRESSION_MODELS_DIR,
    REGRESSION_TUNING_RESULTS_PATH,
)
from ml.data.feature_selection import select_classification_data, select_regression_data
from ml.data.split import get_train_test_split
from ml.persistence import save_pipeline
from ml.pipelines.classification_pipeline import build_classification_pipeline
from ml.pipelines.regression_pipeline import build_regression_pipeline
from ml.training.search import load_tuning_results
from ml.training.tuning_spaces import CLASSIFICATION_MODELS, REGRESSION_MODELS

logger = logging.getLogger(__name__)


def _export_task_models(
    tuning_results_path,
    models: dict,
    pipeline_builder,
    X_train,
    y_train,
    output_dir,
) -> list[str]:
    tuning_results = load_tuning_results(tuning_results_path)
    exported = []
    for name, spec in tuning_results.items():
        if name == "best_model":
            continue
        pipeline: Pipeline = pipeline_builder(clone(models[name]["estimator"]))
        pipeline.set_params(**spec["best_params"])
        pipeline.fit(X_train, y_train)
        save_pipeline(pipeline, output_dir / f"{name}.joblib")
        exported.append(name)
    return exported


def main() -> None:
    train_df, _ = get_train_test_split()

    X_clf, y_clf = select_classification_data(train_df)
    clf_exported = _export_task_models(
        CLASSIFICATION_TUNING_RESULTS_PATH,
        CLASSIFICATION_MODELS,
        build_classification_pipeline,
        X_clf,
        y_clf,
        CLASSIFICATION_MODELS_DIR,
    )
    logger.info("Exported classification models: %s", clf_exported)

    X_reg, y_reg = select_regression_data(train_df)
    reg_exported = _export_task_models(
        REGRESSION_TUNING_RESULTS_PATH,
        REGRESSION_MODELS,
        build_regression_pipeline,
        X_reg,
        y_reg,
        REGRESSION_MODELS_DIR,
    )
    logger.info("Exported regression models: %s", reg_exported)

    print(f"Classification models saved to: {CLASSIFICATION_MODELS_DIR} ({clf_exported})")
    print(f"Regression models saved to:     {REGRESSION_MODELS_DIR} ({reg_exported})")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
