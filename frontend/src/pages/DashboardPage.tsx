import type { ReactNode } from "react";
import { ConfusionMatrixChart } from "../components/dashboard/ConfusionMatrixChart";
import { FeatureImportanceChart } from "../components/dashboard/FeatureImportanceChart";
import { MetricsCard } from "../components/dashboard/MetricsCard";
import { ResidualPlot } from "../components/dashboard/ResidualPlot";
import { RocCurveChart } from "../components/dashboard/RocCurveChart";
import { useJsonData } from "../hooks/useJsonData";
import type { TestMetricsClassification, TestMetricsRegression } from "../types/dashboard";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="mb-4 text-xl font-semibold text-slate-900">{title}</h2>
      {children}
    </section>
  );
}

export function DashboardPage() {
  const { data: clfMetrics } = useJsonData<TestMetricsClassification>(
    "/data/test_metrics_classification.json",
  );
  const { data: regMetrics } = useJsonData<TestMetricsRegression>(
    "/data/test_metrics_regression.json",
  );

  return (
    <div className="mx-auto max-w-5xl px-6 py-10">
      <h1 className="mb-8 text-2xl font-bold text-slate-900">Model Results (Held-Out Test Set)</h1>

      <Section title="Classification — Long vs. Short Consultation">
        <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-5">
          <MetricsCard label="Accuracy" value={clfMetrics ? clfMetrics.accuracy.toFixed(3) : "…"} />
          <MetricsCard label="Precision" value={clfMetrics ? clfMetrics.precision.toFixed(3) : "…"} />
          <MetricsCard label="Recall" value={clfMetrics ? clfMetrics.recall.toFixed(3) : "…"} />
          <MetricsCard label="F1" value={clfMetrics ? clfMetrics.f1.toFixed(3) : "…"} />
          <MetricsCard label="ROC-AUC" value={clfMetrics ? clfMetrics.roc_auc.toFixed(3) : "…"} />
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">Confusion Matrix</h3>
            <ConfusionMatrixChart dataUrl="/data/confusion_matrix.json" />
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">ROC Curve</h3>
            <RocCurveChart dataUrl="/data/roc_curve.json" />
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4 lg:col-span-2">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">Feature Importance</h3>
            <FeatureImportanceChart dataUrl="/data/feature_importance_classification.json" />
          </div>
        </div>
      </Section>

      <Section title="Regression — Consultation Duration (Minutes)">
        <div className="mb-6 grid grid-cols-3 gap-4 sm:w-2/3">
          <MetricsCard label="MAE" value={regMetrics ? `${regMetrics.mae.toFixed(2)} min` : "…"} />
          <MetricsCard label="RMSE" value={regMetrics ? `${regMetrics.rmse.toFixed(2)} min` : "…"} />
          <MetricsCard label="R²" value={regMetrics ? regMetrics.r2.toFixed(3) : "…"} />
        </div>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">Residuals vs. Predicted</h3>
            <ResidualPlot dataUrl="/data/residuals.json" />
          </div>
          <div className="rounded-lg border border-slate-200 bg-white p-4">
            <h3 className="mb-3 text-sm font-semibold text-slate-700">Feature Importance</h3>
            <FeatureImportanceChart dataUrl="/data/feature_importance_regression.json" />
          </div>
        </div>
      </Section>
    </div>
  );
}
