import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ConfusionMatrixChart } from "../components/dashboard/ConfusionMatrixChart";
import { FeatureImportanceChart } from "../components/dashboard/FeatureImportanceChart";
import { MetricsCard } from "../components/dashboard/MetricsCard";
import { RocCurveChart } from "../components/dashboard/RocCurveChart";
import { useJsonData } from "../hooks/useJsonData";
import type { TestMetricsClassification, TestMetricsRegression } from "../types/dashboard";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-12">
      <h2 className="mb-5 font-serif text-2xl font-medium text-espresso">{title}</h2>
      {children}
    </section>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-xl border border-espresso/5 bg-white p-5 shadow-sm">
      <h3 className="mb-3 text-sm font-semibold text-espresso-light">{title}</h3>
      {children}
    </div>
  );
}

const features = [
  {
    title: "Long Consultation Risk",
    description:
      "A tuned Random Forest classifier flags whether a visit will exceed the clinic's historical median duration, with calibrated probabilities.",
  },
  {
    title: "Duration Prediction",
    description:
      "A companion regression model predicts consultation length in minutes directly from visit and scheduling details.",
  },
  {
    title: "Leakage-Safe Pipelines",
    description:
      "Grouped, patient-aware train/test splitting and cross-validation prevent repeat visits from leaking across folds.",
  },
  {
    title: "Real-Time API",
    description:
      "A FastAPI backend serves both models behind separate, documented endpoints with automatic OpenAPI docs.",
  },
  {
    title: "Honest Evaluation",
    description:
      "Metrics reported here come from a single held-out test set, evaluated once, never used for tuning.",
  },
  {
    title: "Interactive Dashboard",
    description:
      "Confusion matrix, ROC curve, feature importance, and residuals — rendered live from the actual test-set results.",
  },
];

export function DashboardPage() {
  const { data: clfMetrics } = useJsonData<TestMetricsClassification>(
    "/data/test_metrics_classification.json",
  );
  const { data: regMetrics } = useJsonData<TestMetricsRegression>(
    "/data/test_metrics_regression.json",
  );

  const stats = [
    {
      value: clfMetrics ? clfMetrics.roc_auc.toFixed(3) : "…",
      label: "ROC-AUC (classification)",
      hint: "How often the model correctly ranks a “long” consultation above a “short” one (0.5 = random guessing, 1.0 = perfect).",
    },
    {
      value: regMetrics ? regMetrics.r2.toFixed(3) : "…",
      label: "R² (duration prediction)",
      hint: "The share of variation in actual consultation duration the model explains, from 0 (none) to 1 (all of it).",
    },
    {
      value: "6,637",
      label: "Consultations analysed",
      hint: "The total number of historical outpatient visit records from the Hangu clinic used to train and test both models.",
    },
  ];

  return (
    <div className="mx-auto max-w-5xl px-6 py-14">
      <div className="mb-16 flex justify-center">
        <Link
          to="/predict/classification"
          className="rounded-full bg-terracotta px-12 py-5 text-xl font-semibold text-cream shadow-md transition-colors hover:bg-terracotta-dark"
        >
          Try Prediction →
        </Link>
      </div>

      <div className="mb-16 text-center">
        <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
          What This Tool Does
        </span>
        <h2 className="mt-3 font-serif text-3xl font-medium text-espresso sm:text-4xl">
          Prediction, evaluation, and transparency
        </h2>
        <p className="mx-auto mt-3 max-w-xl text-espresso-light">
          Two production-grade models serve every request, backed by an honest,
          leakage-free evaluation you can inspect on the dashboard.
        </p>

        <div className="mt-10 grid grid-cols-1 gap-5 text-left sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="rounded-xl border border-espresso/5 bg-white p-6 shadow-sm"
            >
              <h3 className="font-serif text-lg font-medium text-espresso">{feature.title}</h3>
              <p className="mt-2 text-sm text-espresso-light">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
        Model Performance
      </span>
      <h1 className="mt-2 mb-6 font-serif text-3xl font-medium text-espresso sm:text-4xl">
        Results on the held-out test set
      </h1>

      <div className="mb-12 grid grid-cols-1 gap-4 rounded-2xl bg-white/70 p-2 shadow-sm sm:grid-cols-3">
        {stats.map((stat) => (
          <div key={stat.label} className="rounded-xl px-4 py-5">
            <p className="font-serif text-3xl font-medium text-terracotta-dark">{stat.value}</p>
            <p className="mt-1 text-xs font-medium text-espresso-light">{stat.label}</p>
            <p className="mt-2 text-xs text-espresso-light/70">{stat.hint}</p>
          </div>
        ))}
      </div>

      <Section title="Classification — Long vs. Short Consultation">
        <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-5">
          <MetricsCard
            label="Accuracy"
            value={clfMetrics ? clfMetrics.accuracy.toFixed(3) : "…"}
            hint="Share of all predictions — long or short — that were correct."
          />
          <MetricsCard
            label="Precision"
            value={clfMetrics ? clfMetrics.precision.toFixed(3) : "…"}
            hint="Of the visits flagged “long,” how many actually were."
          />
          <MetricsCard
            label="Recall"
            value={clfMetrics ? clfMetrics.recall.toFixed(3) : "…"}
            hint="Of the visits that were actually long, how many got flagged."
          />
          <MetricsCard
            label="F1"
            value={clfMetrics ? clfMetrics.f1.toFixed(3) : "…"}
            hint="A single balance of precision and recall."
          />
          <MetricsCard
            label="ROC-AUC"
            value={clfMetrics ? clfMetrics.roc_auc.toFixed(3) : "…"}
            hint="How often the model ranks a “long” consultation above a “short” one (0.5 = random, 1.0 = perfect)."
          />
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <ChartCard title="Confusion Matrix">
            <ConfusionMatrixChart dataUrl="/data/confusion_matrix.json" />
          </ChartCard>
          <ChartCard title="ROC Curve">
            <RocCurveChart dataUrl="/data/roc_curve.json" />
          </ChartCard>
          <div className="lg:col-span-2">
            <ChartCard title="Feature Importance">
              <FeatureImportanceChart dataUrl="/data/feature_importance_classification.json" />
            </ChartCard>
          </div>
        </div>
      </Section>

      <section className="mt-4">
        <span className="mb-6 block text-xs font-semibold tracking-wide text-terracotta uppercase">
          About This Project
        </span>

        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <div className="rounded-xl border border-espresso/5 bg-white p-6 shadow-sm">
            <h3 className="font-serif text-lg font-medium text-espresso">Dataset</h3>
            <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-espresso-light">
              <li>
                Sourced from Hangu Open Data (Feng et al., 2023), licensed CC BY-NC-SA 4.0 —
                see the citation in the footer.
              </li>
              <li>6,637 consultation records from 2,429 unique patients, many with repeat visits.</li>
              <li>
                Predictors include visit number, gender, primary/secondary cancer flags, day of
                week, month, morning/afternoon session, working-day status, and patient address
                region.
              </li>
            </ul>
          </div>

          <div className="rounded-xl border border-espresso/5 bg-white p-6 shadow-sm">
            <h3 className="font-serif text-lg font-medium text-espresso">Methodology</h3>
            <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-espresso-light">
              <li>
                Data is cleaned and feature-engineered (e.g. repeat-visit flag, public-holiday
                makeup flag, combined cancer-diagnosis flag).
              </li>
              <li>
                Train/test splitting and cross-validation are grouped by patient, so no
                patient's visits leak across folds.
              </li>
              <li>
                Preprocessing (imputation, scaling, one-hot encoding) is fit only on training
                data via a scikit-learn ColumnTransformer.
              </li>
              <li>
                A Dummy baseline, Logistic Regression / Ridge, a Decision Tree, Random Forest,
                and XGBoost were compared via randomized hyperparameter search; Random Forest
                was selected for both tasks. Try the Live Prediction pages to see every
                model's prediction side by side.
              </li>
              <li>The held-out test set was evaluated exactly once, after model selection.</li>
            </ul>
          </div>

          <div className="rounded-xl border border-espresso/5 bg-white p-6 shadow-sm">
            <h3 className="font-serif text-lg font-medium text-espresso">Technology Stack</h3>
            <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-espresso-light">
              <li>Machine learning: scikit-learn, pandas, NumPy</li>
              <li>Backend: FastAPI, serving models via a REST API with automatic OpenAPI docs</li>
              <li>Frontend: React, TypeScript, Vite, Tailwind CSS, Recharts</li>
              <li>Deployment: Render (backend), Vercel (frontend)</li>
            </ul>
          </div>

          <div className="rounded-xl border border-terracotta/20 bg-terracotta-light p-6">
            <h3 className="font-serif text-lg font-medium text-terracotta-dark">
              Known Limitations
            </h3>
            <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-espresso-light">
              <li>
                Regression R² is low — most variation in consultation duration isn't explained
                by the available features.
              </li>
              <li>Classification ROC-AUC is modest — the model captures real but limited signal.</li>
              <li>
                Predictions compress into a narrower range than actual durations, so genuinely
                long consultations tend to be underestimated.
              </li>
              <li>
                Data comes from a single clinic over a limited time period, so results may not
                generalize to other clinics or years.
              </li>
              <li>
                Predictions are estimates for scheduling support only, not clinical guidance.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
