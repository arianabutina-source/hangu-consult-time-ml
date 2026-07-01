"""Backend settings, sourced from environment variables (with `.env` support).

Default artifact paths point at ``ml/artifacts/`` in this same repository
(a monorepo layout: the backend imports ``ml.data.features`` and
``ml.persistence`` directly, so it must be run with the repository root on
``PYTHONPATH`` -- e.g. ``uvicorn backend.app.main:app`` invoked from the repo
root, which is also how every ``ml`` CLI module in this project is run).
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_DIR.parent
_DEFAULT_ARTIFACTS_DIR = _REPO_ROOT / "ml" / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    classifier_pipeline_path: Path = _DEFAULT_ARTIFACTS_DIR / "classifier_pipeline.joblib"
    regressor_pipeline_path: Path = _DEFAULT_ARTIFACTS_DIR / "regressor_pipeline.joblib"

    # Comma-separated list of origins allowed to call this API (the deployed
    # Vercel frontend URL, plus localhost during development).
    allowed_origins: str = "http://localhost:5173"

    log_level: str = "INFO"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
