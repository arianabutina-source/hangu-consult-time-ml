import { Link } from "react-router-dom";

export function HomePage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-16 text-center">
      <h1 className="text-3xl font-bold text-slate-900 sm:text-4xl">
        Predicting Consultation Duration at the Hangu Outpatient Oncology Clinic
      </h1>
      <p className="mt-4 text-lg text-slate-600">
        This tool serves two models trained on 6,637 historical consultation records:
        a classifier predicting whether a consultation will be long or short, and a
        regressor predicting its duration in minutes. It supports research into
        Time-Driven Activity-Based Costing (TDABC) for outpatient scheduling.
      </p>
      <div className="mt-8 flex justify-center gap-4">
        <Link
          to="/predict/classification"
          className="rounded-md bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500"
        >
          Try the classifier
        </Link>
        <Link
          to="/predict/regression"
          className="rounded-md bg-white px-5 py-3 text-sm font-semibold text-indigo-700 shadow-sm ring-1 ring-inset ring-indigo-200 hover:bg-indigo-50"
        >
          Try the regressor
        </Link>
        <Link
          to="/dashboard"
          className="rounded-md bg-white px-5 py-3 text-sm font-semibold text-slate-700 shadow-sm ring-1 ring-inset ring-slate-200 hover:bg-slate-50"
        >
          View model results
        </Link>
      </div>
    </div>
  );
}
