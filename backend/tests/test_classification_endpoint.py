"""Tests for POST /api/v1/predict/classification."""

import pytest
from fastapi.testclient import TestClient


def test_predict_classification_returns_valid_response(
    client: TestClient, valid_payload: dict
) -> None:
    response = client.post("/api/v1/predict/classification", json=valid_payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"is_long_consultation", "probability_long", "probability_short"}
    assert isinstance(body["is_long_consultation"], bool)
    assert 0.0 <= body["probability_long"] <= 1.0
    assert 0.0 <= body["probability_short"] <= 1.0
    assert body["probability_long"] + body["probability_short"] == pytest.approx(1.0, abs=1e-6)


def test_predict_classification_rejects_non_positive_visit_number(
    client: TestClient, valid_payload: dict
) -> None:
    payload = {**valid_payload, "visit_number": 0}
    response = client.post("/api/v1/predict/classification", json=payload)
    assert response.status_code == 422


def test_predict_classification_rejects_invalid_enum_value(
    client: TestClient, valid_payload: dict
) -> None:
    payload = {**valid_payload, "gender": "X"}
    response = client.post("/api/v1/predict/classification", json=payload)
    assert response.status_code == 422


def test_predict_classification_rejects_missing_field(
    client: TestClient, valid_payload: dict
) -> None:
    payload = {k: v for k, v in valid_payload.items() if k != "month"}
    response = client.post("/api/v1/predict/classification", json=payload)
    assert response.status_code == 422


def test_predict_classification_is_deterministic(
    client: TestClient, valid_payload: dict
) -> None:
    first = client.post("/api/v1/predict/classification", json=valid_payload).json()
    second = client.post("/api/v1/predict/classification", json=valid_payload).json()
    assert first == second
