"""Regression prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.dependencies import get_model_registry
from backend.app.schemas.common import ConsultationInput
from backend.app.schemas.regression import RegressionOutput
from backend.app.services.inference import predict_regression
from backend.app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/api/v1", tags=["regression"])


@router.post(
    "/predict/regression",
    response_model=RegressionOutput,
    summary="Predict consultation duration in minutes",
)
def predict_regression_endpoint(
    payload: ConsultationInput, registry: ModelRegistry = Depends(get_model_registry)
) -> RegressionOutput:
    return predict_regression(registry.regressor, payload)
