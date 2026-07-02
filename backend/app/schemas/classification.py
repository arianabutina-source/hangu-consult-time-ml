"""Response schema for the classification endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationModelPrediction(BaseModel):
    model: str = Field(..., description="Model-ladder key, e.g. 'random_forest'.")
    is_long_consultation: bool = Field(
        ...,
        description="Predicted label: True if the consultation is predicted to "
        "exceed the clinic's historical median duration.",
    )
    probability_long: float = Field(..., ge=0.0, le=1.0)
    probability_short: float = Field(..., ge=0.0, le=1.0)


class ClassificationOutput(BaseModel):
    best_model: str = Field(
        ..., description="Which model-ladder entry is the deployed, CV-selected model."
    )
    predictions: list[ClassificationModelPrediction] = Field(
        ..., description="One prediction per model in the ladder (dummy, linear, decision "
        "tree, random forest, xgboost), including the deployed model."
    )
