"""Response schema for the regression endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegressionModelPrediction(BaseModel):
    model: str = Field(..., description="Model-ladder key, e.g. 'random_forest'.")
    predicted_duration_minutes: float = Field(
        ..., ge=0.0, description="Predicted consultation duration, in minutes."
    )


class RegressionOutput(BaseModel):
    best_model: str = Field(
        ..., description="Which model-ladder entry is the deployed, CV-selected model."
    )
    predictions: list[RegressionModelPrediction] = Field(
        ..., description="One prediction per model in the ladder (dummy, linear, decision "
        "tree, random forest, xgboost), including the deployed model."
    )
