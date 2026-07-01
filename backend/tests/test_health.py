"""Tests for the /api/v1/health endpoint."""

from fastapi.testclient import TestClient

from backend.app.main import app


def test_health_check_returns_ok() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_docs_are_served() -> None:
    with TestClient(app) as client:
        docs_response = client.get("/docs")
        openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 200
    assert openapi_response.status_code == 200

    schema = openapi_response.json()
    assert "/api/v1/predict/classification" in schema["paths"]
    assert "/api/v1/predict/regression" in schema["paths"]
    assert "/api/v1/health" in schema["paths"]
