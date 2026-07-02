import { ClassificationForm } from "../components/forms/ClassificationForm";

export function ClassificationPredictPage() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-14">
      <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
        Live Prediction
      </span>
      <h1 className="mt-2 font-serif text-3xl font-medium text-espresso">
        Will this consultation be long?
      </h1>
      <p className="mt-2 text-espresso-light">
        Enter the visit details below to predict whether the consultation will exceed the
        clinic's historical median duration.
      </p>
      <div className="mt-8 rounded-xl border border-espresso/5 bg-white p-6 shadow-sm">
        <ClassificationForm />
      </div>
    </div>
  );
}
