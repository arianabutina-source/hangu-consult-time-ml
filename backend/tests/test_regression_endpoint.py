"""Tests for POST /api/v1/predict/regression."""

from fastapi.testclient import TestClient


def test_predict_regression_returns_valid_response(
    client: TestClient, valid_payload: dict
) -> None:
    response = client.post("/api/v1/predict/regression", json=valid_payload)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"best_model", "predictions"}
    assert isinstance(body["best_model"], str)

    model_names = {row["model"] for row in body["predictions"]}
    assert body["best_model"] in model_names
    assert {"dummy", "ridge", "decision_tree", "random_forest", "xgboost"} == model_names

    for row in body["predictions"]:
        assert set(row.keys()) == {"model", "predicted_duration_minutes"}
        assert row["predicted_duration_minutes"] >= 0.0
        assert row["predicted_duration_minutes"] < 120.0  # plausible upper bound


def test_predict_regression_rejects_non_positive_visit_number(
    client: TestClient, valid_payload: dict
) -> None:
    payload = {**valid_payload, "visit_number": -1}
    response = client.post("/api/v1/predict/regression", json=payload)
    assert response.status_code == 422


def test_predict_regression_rejects_invalid_enum_value(
    client: TestClient, valid_payload: dict
) -> None:
    payload = {**valid_payload, "address": "Nowhere"}
    response = client.post("/api/v1/predict/regression", json=payload)
    assert response.status_code == 422


def test_predict_regression_rejects_missing_field(
    client: TestClient, valid_payload: dict
) -> None:
    payload = {k: v for k, v in valid_payload.items() if k != "session"}
    response = client.post("/api/v1/predict/regression", json=payload)
    assert response.status_code == 422


def test_predict_regression_is_deterministic(client: TestClient, valid_payload: dict) -> None:
    first = client.post("/api/v1/predict/regression", json=valid_payload).json()
    second = client.post("/api/v1/predict/regression", json=valid_payload).json()
    assert first == second
