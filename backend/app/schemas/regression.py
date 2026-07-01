"""Response schema for the regression endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RegressionOutput(BaseModel):
    predicted_duration_minutes: float = Field(
        ..., ge=0.0, description="Predicted consultation duration, in minutes."
    )
