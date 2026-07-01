"""FastAPI dependency providers."""

from __future__ import annotations

from fastapi import Request

from backend.app.services.model_registry import ModelRegistry


def get_model_registry(request: Request) -> ModelRegistry:
    """Return the ``ModelRegistry`` built once at startup (see ``app.main``)."""
    return request.app.state.model_registry
