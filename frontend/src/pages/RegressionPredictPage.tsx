import { RegressionForm } from "../components/forms/RegressionForm";

export function RegressionPredictPage() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-10">
      <h1 className="text-2xl font-bold text-slate-900">
        Predict consultation duration
      </h1>
      <p className="mt-2 text-slate-600">
        Enter the visit details below to predict the consultation's expected duration in
        minutes.
      </p>
      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <RegressionForm />
      </div>
    </div>
  );
}
