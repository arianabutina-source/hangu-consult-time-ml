"""Request -> feature frame -> prediction, shared by both endpoints.

Feature engineering here reuses the exact same functions applied during
training (``ml.data.features``), so a request can never be scored against
logic that has drifted from how the model was fit.
"""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from ml.data.features import (
    add_combined_cancer_flag,
    add_public_holiday_makeup_flag,
    add_repeat_visit_flag,
)

from backend.app.schemas.classification import ClassificationOutput
from backend.app.schemas.common import ConsultationInput
from backend.app.schemas.regression import RegressionOutput


def build_feature_frame(payload: ConsultationInput) -> pd.DataFrame:
    """Convert a validated request into the exact predictor schema the
    trained pipelines expect (raw columns + engineered features)."""
    raw = {
        "Visit.No": payload.visit_number,
        "WorkingDay": payload.is_working_day,
        "M.Cancer": payload.has_primary_cancer,
        "S.Cancer": payload.has_secondary_cancer,
        "Month": payload.month.value,
        "DayOfWeek": payload.day_of_week.value,
        "AM_PM": payload.session.value,
        "Gender": payload.gender.value,
        "Address": payload.address.value,
    }
    frame = pd.DataFrame([raw])
    frame = add_repeat_visit_flag(frame)
    frame = add_public_holiday_makeup_flag(frame)
    frame = add_combined_cancer_flag(frame)
    return frame


def predict_classification(pipeline: Pipeline, payload: ConsultationInput) -> ClassificationOutput:
    """Run the classification pipeline on a single request payload."""
    frame = build_feature_frame(payload)
    is_long = bool(pipeline.predict(frame)[0])
    probability_short, probability_long = pipeline.predict_proba(frame)[0]
    return ClassificationOutput(
        is_long_consultation=is_long,
        probability_long=float(probability_long),
        probability_short=float(probability_short),
    )


def predict_regression(pipeline: Pipeline, payload: ConsultationInput) -> RegressionOutput:
    """Run the regression pipeline on a single request payload."""
    frame = build_feature_frame(payload)
    predicted_minutes = float(pipeline.predict(frame)[0])
    # Duration cannot be negative; clip at the API boundary only (the model
    # itself is never modified).
    return RegressionOutput(predicted_duration_minutes=max(predicted_minutes, 0.0))
