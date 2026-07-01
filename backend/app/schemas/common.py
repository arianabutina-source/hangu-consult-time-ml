"""Shared request schema for both prediction endpoints.

This mirrors the *raw* predictor columns used during training (see
``ml.data.feature_selection``), not the engineered ones. Callers never
supply ``IsRepeatVisit``, ``IsPublicHolidayMakeup``, or
``HasCancerDiagnosis`` directly -- those are deterministic functions of the
fields below and are derived server-side (see
``backend.app.services.inference``) using the exact same functions applied
during training, so inference can never diverge from how the model was fit.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Month(str, Enum):
    JANUARY = "January"
    FEBRUARY = "February"
    MARCH = "March"
    APRIL = "April"
    MAY = "May"
    JUNE = "June"
    JULY = "July"
    AUGUST = "August"
    SEPTEMBER = "September"
    OCTOBER = "October"
    NOVEMBER = "November"
    DECEMBER = "December"


class DayOfWeek(str, Enum):
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"


class SessionPeriod(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"


class Gender(str, Enum):
    FEMALE = "F"
    MALE = "M"


class Address(str, Enum):
    IN_THE_CITY = "In the city"
    OUT_OF_CITY = "Out of city"
    OUT_OF_PROVINCE = "Out of province"
    UNKNOWN = "Unknown"


class ConsultationInput(BaseModel):
    """Natural, clinic-facing predictor set for a single consultation."""

    visit_number: int = Field(
        ...,
        gt=0,
        examples=[3],
        description="Patient's visit number at this clinic (1 = first visit).",
    )
    is_working_day: bool = Field(
        ...,
        examples=[True],
        description="Whether this calendar day is an official working day for the clinic.",
    )
    has_primary_cancer: bool = Field(
        ..., examples=[False], description="Primary cancer diagnosis flag."
    )
    has_secondary_cancer: bool = Field(
        ..., examples=[False], description="Secondary cancer diagnosis flag."
    )
    month: Month = Field(..., examples=[Month.JANUARY])
    day_of_week: DayOfWeek = Field(..., examples=[DayOfWeek.WEDNESDAY])
    session: SessionPeriod = Field(..., examples=[SessionPeriod.MORNING])
    gender: Gender = Field(..., examples=[Gender.FEMALE])
    address: Address = Field(..., examples=[Address.IN_THE_CITY])
