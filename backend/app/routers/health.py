"""Liveness/readiness endpoint."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", summary="Liveness/readiness check")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
