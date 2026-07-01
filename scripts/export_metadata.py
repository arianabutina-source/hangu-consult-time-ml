"""Final export step: serialize both tuned pipelines and a metadata record.

Reuses Milestone 11's evaluation entry points (which fit the tuned pipeline
on the full training set and evaluate it once on the held-out test set), so
the exact same fitted pipelines and test metrics reported earlier are what
get serialized here -- nothing is retrained or re-evaluated.

Run as: ``python -m scripts.export_metadata``
"""

from __future__ import annotations

import logging

from ml.config import (
    CLASSIFICATION_TUNING_RESULTS_PATH,
    CLASSIFIER_PIPELINE_PATH,
    METADATA_PATH,
    REGRESSION_TUNING_RESULTS_PATH,
    REGRESSOR_PIPELINE_PATH,
)
from ml.evaluation.evaluate_classifier import run_classification_evaluation
from ml.evaluation.evaluate_regressor import run_regression_evaluation
from ml.evaluation.metrics import save_metrics
from ml.persistence import build_metadata, save_pipeline
from ml.training.search import load_tuning_results

logger = logging.getLogger(__name__)


def main() -> None:
    clf_pipeline, clf_test_metrics = run_classification_evaluation()
    reg_pipeline, reg_test_metrics = run_regression_evaluation()

    save_pipeline(clf_pipeline, CLASSIFIER_PIPELINE_PATH)
    save_pipeline(reg_pipeline, REGRESSOR_PIPELINE_PATH)

    clf_tuning_results = load_tuning_results(CLASSIFICATION_TUNING_RESULTS_PATH)
    reg_tuning_results = load_tuning_results(REGRESSION_TUNING_RESULTS_PATH)

    metadata = build_metadata(
        clf_test_metrics, reg_test_metrics, clf_tuning_results, reg_tuning_results
    )
    save_metrics(metadata, METADATA_PATH)

    commit = metadata["git_commit"] or "unknown (not a git repository)"
    print(f"Classifier saved to: {CLASSIFIER_PIPELINE_PATH}")
    print(f"Regressor saved to:  {REGRESSOR_PIPELINE_PATH}")
    print(f"Metadata saved to:   {METADATA_PATH}")
    print(f"Git commit: {commit}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
