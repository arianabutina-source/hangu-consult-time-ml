import { ClassificationForm } from "../components/forms/ClassificationForm";

export function ClassificationPredictPage() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-10">
      <h1 className="text-2xl font-bold text-slate-900">
        Will this consultation be long?
      </h1>
      <p className="mt-2 text-slate-600">
        Enter the visit details below to predict whether the consultation will exceed the
        clinic's historical median duration.
      </p>
      <div className="mt-8 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
        <ClassificationForm />
      </div>
    </div>
  );
}
