# Backend — FastAPI

Serves the classification and regression models trained in `ml/`. Both serialized
pipelines are loaded once at startup (see `app/services/model_registry.py`) and reused
for every request — no retraining, no per-request disk I/O.

> This service imports `ml.data.features` and `ml.persistence` directly (a monorepo
> layout), so it must be run with the **repository root** on `PYTHONPATH` — e.g. invoked
> from the repo root as shown below, not from inside `backend/`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness/readiness check |
| `POST` | `/api/v1/predict/classification` | Predict long vs. short consultation |
| `POST` | `/api/v1/predict/regression` | Predict consultation duration in minutes |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated) |
| `GET` | `/openapi.json` | Raw OpenAPI schema |

Both predict endpoints take the same request body — the *raw*, clinic-facing fields
(`visit_number`, `is_working_day`, `has_primary_cancer`, `has_secondary_cancer`, `month`,
`day_of_week`, `session`, `gender`, `address`). Engineered features
(`IsRepeatVisit`, `IsPublicHolidayMakeup`, `HasCancerDiagnosis`) are derived server-side
using the exact same functions applied during training — callers never compute them.

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/predict/classification \
  -H "Content-Type: application/json" \
  -d '{
    "visit_number": 3,
    "is_working_day": true,
    "has_primary_cancer": false,
    "has_secondary_cancer": false,
    "month": "January",
    "day_of_week": "Wednesday",
    "session": "morning",
    "gender": "F",
    "address": "In the city"
  }'
```

## Setup

```bash
cd backend
pip install -r requirements-dev.txt   # includes pytest + httpx for the test suite
cd ..
uvicorn backend.app.main:app --reload
```

Requires `ml/artifacts/classifier_pipeline.joblib` and `regressor_pipeline.joblib` to
exist — run `python -m scripts.export_metadata` from the repo root first if they don't
(see the root [README](../README.md)).

## Environment Variables

See [`.env.example`](.env.example):

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated CORS allow-list |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CLASSIFIER_PIPELINE_PATH` | `ml/artifacts/classifier_pipeline.joblib` | Override the classifier artifact path |
| `REGRESSOR_PIPELINE_PATH` | `ml/artifacts/regressor_pipeline.joblib` | Override the regressor artifact path |

## Tests

```bash
pytest backend/tests
```

Covers health, both predict endpoints (valid input, validation errors), and that
`/docs`/`/openapi.json` are served.

## Docker

```bash
docker build -f backend/Dockerfile -t optimised-scheduling-backend .   # from repo root
docker run -p 8000:8000 optimised-scheduling-backend
```

The image copies the whole `ml/` package (the serialized pipelines reference
`ml.preprocessing.column_transformer` by module path when unpickled) plus `backend/`.
Note: `xgboost` is intentionally **not** in `backend/requirements.txt` — the currently
serialized models are Random Forest, and `xgboost`'s Linux wheel pulls in a large CUDA
dependency that isn't needed for inference. Add it back if a future retrain selects an
XGBoost model as best.

## Deployment

Deployed on Render via `render.yaml` at the repo root (Docker-based web service,
auto-deploys on push to `main`). `ALLOWED_ORIGINS` must be updated in Render's dashboard
(or `render.yaml`) whenever the deployed frontend's domain changes.
