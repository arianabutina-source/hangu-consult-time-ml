"""Classification prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_model_registry
from backend.app.schemas.classification import ClassificationOutput
from backend.app.schemas.common import ConsultationInput
from backend.app.services.inference import predict_classification
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/api/v1", tags=["classification"])


@router.post(
    "/predict/classification",
    response_model=ClassificationOutput,
    summary="Predict whether a consultation will be long",
)
def predict_classification_endpoint(
    payload: ConsultationInput, registry: ModelRegistry = Depends(get_model_registry)
) -> ClassificationOutput:
    return predict_classification(registry.classifiers, registry.best_classifier_name, payload)
