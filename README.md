# Optimised Scheduling Tool

Predicting outpatient consultation duration at the Hangu outpatient oncology clinic, to
support **Time-Driven Activity-Based Costing (TDABC)** research into outpatient
scheduling. A reproducible, leakage-audited ML pipeline (scikit-learn + XGBoost), served
through a FastAPI backend and an interactive React dashboard.

## Live

- **App:** https://frontend-psi-seven-76.vercel.app
- **API docs:** https://hangu-consult-backend.onrender.com/docs
- **Repo:** https://github.com/arianabutina-source/hangu-consult-time-ml

> The backend is on Render's free tier and spins down when idle — the first request after
> a while can take 30–60 seconds to wake up.

## Course Deliverables

| # | Deliverable | Where |
|---|---|---|
| D1 | Reproducible analysis report (Quarto → HTML) | [`report/report.qmd`](report/report.qmd) → [`report.html`](report/report.html) |
| D2 | Deployed web app (React) | https://frontend-psi-seven-76.vercel.app |
| D3 | AI-workflow reflection | [`report/ai_workflow_reflection.qmd`](report/ai_workflow_reflection.qmd) → [`.html`](report/ai_workflow_reflection.html) / [`.pdf`](report/ai_workflow_reflection.pdf) |
| D4 | Presentation (Quarto reveal.js, non-technical) | [`report/slides.qmd`](report/slides.qmd) → [`slides.html`](report/slides.html) |
| D5 | Executive summary (1 page, plain language) | [`report/executive_summary.qmd`](report/executive_summary.qmd) → [`.html`](report/executive_summary.html) / [`.pdf`](report/executive_summary.pdf) |

To view any rendered `.html` deliverable, download it and open it in a browser (each is
a self-contained file, images and all — no server needed). D1's reproducibility is
verified by cloning this repo fresh, creating a new virtual environment from
`requirements.txt`, and running `quarto render report.qmd` from `report/` (see
**Setup** below) — no other steps required.

Two supplementary, non-required documents also live in `report/`:
[`presentation.html`](report/presentation.html) (a short written summary from an earlier
project phase) and [`slides_technical_appendix.html`](report/slides_technical_appendix.html)
(a deeper, code-level walkthrough of the `ml/` pipeline for anyone who wants the
function-by-function detail — not the D4 deliverable itself, which is audience-appropriate
for a non-technical reader).

## Reports

Rendered, reproducible outputs live under [`report/`](report/):

| Document | Purpose |
|---|---|
| [`report.html`](report/report.html) | Full academic report — methodology, results, discussion (**D1**) |
| [`slides.html`](report/slides.html) | Presentation deck for a non-technical audience (**D4**) |
| [`executive_summary.pdf`](report/executive_summary.pdf) | 1-page, jargon-light summary (**D5**) |
| [`ai_workflow_reflection.pdf`](report/ai_workflow_reflection.pdf) | AI-workflow reflection (**D3**) |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Executed exploratory data analysis notebook |

## The Problem, in Brief

Two models are trained on 6,637 historical consultations:

- **Classification** — will this consultation run *long* relative to the clinic's own
  history? (Random Forest, held-out test ROC-AUC 0.632)
- **Regression** — how many minutes will it take? (Random Forest, held-out test MAE 4.41
  min, R² 0.066)

The headline finding: **whether a patient has visited before is, by a wide margin, the
strongest predictor of visit length** — well ahead of diagnosis flags or calendar
context. See [`report/report.html`](report/report.html) for the full discussion,
including honest limitations.

## Project Structure

```
consult-time-ml/
├── ml/                    # Data pipeline, preprocessing, models, evaluation (see below)
├── backend/               # FastAPI REST API — see backend/README.md
├── frontend/              # React + Vite dashboard/forms — see frontend/README.md
├── scripts/               # CLI entry points (retrain, export, integration check)
├── data/raw/              # Source dataset (Data.csv)
├── data/processed/        # Persisted train/test split (gitignored, regenerated on demand)
├── notebooks/             # Executed EDA notebook
├── report/                # Quarto report, presentation, executive summary
├── tests/                 # ML pipeline test suite (pytest)
├── render.yaml            # Render deployment blueprint (backend)
└── requirements.txt       # Python dependencies for the ml/ pipeline
```

`ml/` is organised by pipeline stage:

```
ml/
├── config.py              # Paths, RANDOM_STATE, schema constants -- single source of truth
├── data/                  # load -> clean -> features -> feature_selection -> split
├── preprocessing/         # Shared ColumnTransformer (leakage-safe: fit on train only)
├── pipelines/             # Classification/regression Pipeline builders
├── training/              # Cross-validation + hyperparameter tuning (RandomizedSearchCV)
├── evaluation/            # Held-out test metrics, figures, feature importance, SHAP
├── eda/                   # Reusable summary statistics + plotting functions
├── artifacts/             # Serialized models + metadata (gitignored except final outputs)
└── persistence.py         # joblib save/load, reproducibility metadata
```

## Dataset

Sourced from **Hangu Open Data** (Feng et al., 2023, 2024), licensed **CC BY-NC-SA 4.0**.
6,637 consultation records from 2,429 unique patients, 2018–2019. See
[`report/report.html`](report/report.html) §2 for the full description, or the citation
in the frontend footer.

## Setup

Requires Python 3.14+, Node 22+, and (optionally) Docker.

```bash
git clone https://github.com/arianabutina-source/hangu-consult-time-ml.git
cd hangu-consult-time-ml
pip install -r requirements.txt
```

### Run the ML test suite

```bash
pytest tests/
```

### Render the report (D1), presentation (D4), executive summary (D5), and reflection (D3)

Requires [Quarto](https://quarto.org/docs/get-started/) installed separately (not a pip
package). All figures and metrics these documents use are already committed to the repo
(`ml/artifacts/`, `report/figures/`), so rendering does **not** require retraining first:

```bash
cd report
quarto render report.qmd                        # D1 -> report.html
quarto render slides.qmd                        # D4 -> slides.html
quarto render executive_summary.qmd --to html   # D5 -> executive_summary.html
quarto render executive_summary.qmd --to pdf    # D5 -> executive_summary.pdf (requires LaTeX/typst)
quarto render ai_workflow_reflection.qmd --to html
quarto render ai_workflow_reflection.qmd --to pdf
```

### Retrain the models from scratch

Each step is a standalone CLI module; run them in order (later steps depend on earlier
artifacts):

```bash
python -m ml.data.split                    # builds data/processed/{train,test}.csv
python -m ml.training.train_classifier     # tunes + saves classification tuning results
python -m ml.training.train_regressor      # tunes + saves regression tuning results
python -m ml.evaluation.evaluate_classifier # one-time honest test-set evaluation
python -m ml.evaluation.evaluate_regressor
python -m ml.evaluation.explainability      # feature importance + SHAP figures
python -m scripts.export_metadata           # serializes final models + metadata.json
python -m scripts.export_dashboard_data     # exports JSON the frontend dashboard reads
```

All randomness is seeded (`RANDOM_STATE = 42` in `ml/config.py`), so re-running this
sequence reproduces the same split, tuning results, and metrics.

### Run the backend locally

```bash
cd backend
pip install -r requirements-dev.txt
cd ..
uvicorn backend.app.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive OpenAPI docs. See
[`backend/README.md`](backend/README.md) for endpoints and environment variables.

### Run the frontend locally

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`. See [`frontend/README.md`](frontend/README.md) for routes,
environment variables, and build/deploy commands.

### Run both together

```bash
./scripts/integration_check.sh
```

Starts the backend, exercises both predict endpoints, validates input errors and CORS,
then tears it down — a quick end-to-end sanity check without needing the frontend
running.

## Deployment

- **Backend → Render**, via `render.yaml` (Docker-based web service). Auto-deploys on
  push to `main`. `ALLOWED_ORIGINS` must include the deployed frontend's domain.
- **Frontend → Vercel**, via `vercel --prod` from `frontend/` (or connect the repo in the
  Vercel dashboard for git-based auto-deploy). Requires `VITE_API_BASE_URL` set to the
  live backend URL.

## License

The code in this repository is provided for academic/capstone purposes. The dataset is
licensed CC BY-NC-SA 4.0 by its original authors (Feng et al.) — see the citation in
[`report/references.bib`](report/references.bib) and the frontend footer.
