"""Response schema for the classification endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClassificationOutput(BaseModel):
    is_long_consultation: bool = Field(
        ...,
        description="Predicted label: True if the consultation is predicted to "
        "exceed the clinic's historical median duration.",
    )
    probability_long: float = Field(..., ge=0.0, le=1.0)
    probability_short: float = Field(..., ge=0.0, le=1.0)
