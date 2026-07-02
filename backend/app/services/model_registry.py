"""In-memory holder for every tuned pipeline, loaded once at startup.

Loading ``joblib`` files is not free (disk I/O, deserialization), so this
registry is built exactly once during the FastAPI ``lifespan`` (see
``backend.app.main``) and then reused for every request, rather than
reloading from disk per prediction.

Beyond the single CV-selected "best" pipeline per task (``classifier``/
``regressor``, used to answer "what does the deployed model predict?"),
this registry also loads every other candidate in the model ladder
(``classifiers``/``regressors``, keyed by name) so a request can be scored
against all of them at once -- see ``backend.app.services.inference``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from ml.persistence import load_pipeline

logger = logging.getLogger(__name__)


def _load_all_pipelines(models_dir: Path) -> dict[str, Pipeline]:
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Model-ladder directory not found at '{models_dir}'. "
            "Run `python -m scripts.export_all_models` first."
        )
    pipelines = {}
    for path in sorted(models_dir.glob("*.joblib")):
        pipelines[path.stem] = load_pipeline(path)
    if not pipelines:
        raise FileNotFoundError(f"No '*.joblib' files found in '{models_dir}'.")
    return pipelines


@dataclass(frozen=True)
class ModelRegistry:
    classifier: Pipeline
    regressor: Pipeline
    classifiers: dict[str, Pipeline]
    regressors: dict[str, Pipeline]
    best_classifier_name: str
    best_regressor_name: str

    @classmethod
    def load(
        cls,
        classifier_path: Path,
        regressor_path: Path,
        classification_models_dir: Path,
        regression_models_dir: Path,
        metadata_path: Path,
    ) -> "ModelRegistry":
        """Load the deployed pipelines, the full model ladder, and which
        ladder entry is deployed for each task.

        Raises:
            FileNotFoundError: If any artifact is missing (e.g.
                ``scripts.export_metadata`` / ``scripts.export_all_models``
                have not been run yet).
        """
        if not classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier artifact not found at '{classifier_path}'. "
                "Run `python -m scripts.export_metadata` first."
            )
        if not regressor_path.exists():
            raise FileNotFoundError(
                f"Regressor artifact not found at '{regressor_path}'. "
                "Run `python -m scripts.export_metadata` first."
            )
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at '{metadata_path}'. "
                "Run `python -m scripts.export_metadata` first."
            )

        logger.info("Loading classifier from %s", classifier_path)
        classifier = load_pipeline(classifier_path)
        logger.info("Loading regressor from %s", regressor_path)
        regressor = load_pipeline(regressor_path)

        logger.info("Loading classification model ladder from %s", classification_models_dir)
        classifiers = _load_all_pipelines(classification_models_dir)
        logger.info("Loading regression model ladder from %s", regression_models_dir)
        regressors = _load_all_pipelines(regression_models_dir)

        with open(metadata_path) as f:
            metadata = json.load(f)

        return cls(
            classifier=classifier,
            regressor=regressor,
            classifiers=classifiers,
            regressors=regressors,
            best_classifier_name=metadata["classification"]["best_model"],
            best_regressor_name=metadata["regression"]["best_model"],
        )
