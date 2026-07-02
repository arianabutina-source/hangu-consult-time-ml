"""FastAPI application entry point.

Run locally from the repository root with:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.core.logging import configure_logging
from backend.app.routers import classification, health, regression
from backend.app.services.model_registry import ModelRegistry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(settings.log_level)
    app.state.model_registry = ModelRegistry.load(
        settings.classifier_pipeline_path, settings.regressor_pipeline_path
    )
    yield


app = FastAPI(
    title="Optimised Scheduling Tool API",
    description=(
        "Serves a classification model (long vs. short consultation) and a "
        "regression model (predicted duration in minutes) trained on the "
        "Hangu outpatient oncology clinic dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(classification.router)
app.include_router(regression.router)
