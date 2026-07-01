"""Shared fixtures for the backend test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A TestClient whose lifespan (model loading) runs once per test module."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict:
    return {
        "visit_number": 3,
        "is_working_day": True,
        "has_primary_cancer": False,
        "has_secondary_cancer": False,
        "month": "January",
        "day_of_week": "Wednesday",
        "session": "morning",
        "gender": "F",
        "address": "In the city",
    }
