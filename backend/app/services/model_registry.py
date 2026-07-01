"""In-memory holder for the two fitted pipelines, loaded once at startup.

Loading ``joblib`` files is not free (disk I/O, deserialization), so this
registry is built exactly once during the FastAPI ``lifespan`` (see
``backend.app.main``) and then reused for every request, rather than
reloading from disk per prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from ml.persistence import load_pipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelRegistry:
    classifier: Pipeline
    regressor: Pipeline

    @classmethod
    def load(cls, classifier_path: Path, regressor_path: Path) -> "ModelRegistry":
        """Load both serialized pipelines from disk.

        Raises:
            FileNotFoundError: If either artifact is missing (e.g. Milestone
                13's ``scripts.export_metadata`` has not been run yet).
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

        logger.info("Loading classifier from %s", classifier_path)
        classifier = load_pipeline(classifier_path)
        logger.info("Loading regressor from %s", regressor_path)
        regressor = load_pipeline(regressor_path)

        return cls(classifier=classifier, regressor=regressor)
