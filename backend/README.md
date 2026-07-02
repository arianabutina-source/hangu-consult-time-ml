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

Each response scores the request against **every model in the ladder** (`dummy`,
`logistic_regression`/`ridge`, `decision_tree`, `random_forest`, `xgboost`), not just the
deployed one — `best_model` names which entry in `predictions` is the CV-selected,
production model.

Example request/response:

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

```json
{
  "best_model": "random_forest",
  "predictions": [
    {"model": "dummy", "is_long_consultation": false, "probability_long": 0.0, "probability_short": 1.0},
    {"model": "logistic_regression", "is_long_consultation": true, "probability_long": 0.601, "probability_short": 0.399},
    {"model": "decision_tree", "is_long_consultation": true, "probability_long": 0.540, "probability_short": 0.460},
    {"model": "random_forest", "is_long_consultation": true, "probability_long": 0.566, "probability_short": 0.434},
    {"model": "xgboost", "is_long_consultation": true, "probability_long": 0.531, "probability_short": 0.469}
  ]
}
```

`/predict/regression` mirrors this shape with `predicted_duration_minutes` in place of the
three classification fields.

## Setup

```bash
cd backend
pip install -r requirements-dev.txt   # includes pytest + httpx for the test suite
cd ..
uvicorn backend.app.main:app --reload
```

Requires two sets of artifacts under `ml/artifacts/` to exist — run both from the repo
root if they don't (see the root [README](../README.md)):

```bash
python -m scripts.export_metadata      # classifier_pipeline.joblib, regressor_pipeline.joblib, metadata.json
python -m scripts.export_all_models    # models/classification/*.joblib, models/regression/*.joblib (the full ladder)
```

## Environment Variables

See [`.env.example`](.env.example):

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:5173` | Comma-separated CORS allow-list |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `CLASSIFIER_PIPELINE_PATH` | `ml/artifacts/classifier_pipeline.joblib` | Override the deployed classifier artifact path |
| `REGRESSOR_PIPELINE_PATH` | `ml/artifacts/regressor_pipeline.joblib` | Override the deployed regressor artifact path |
| `METADATA_PATH` | `ml/artifacts/metadata.json` | Override the metadata path (used to identify `best_model`) |
| `CLASSIFICATION_MODELS_DIR` | `ml/artifacts/models/classification` | Override the classification model-ladder directory |
| `REGRESSION_MODELS_DIR` | `ml/artifacts/models/regression` | Override the regression model-ladder directory |

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
