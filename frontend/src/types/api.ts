// Mirrors backend/app/schemas/*.py exactly. Keep these in sync by hand --
// there is no shared schema generator in this project.

export const MONTHS = [
  "January",
  "February",
  "March",
  "April",
  "May",
  "June",
  "July",
  "August",
  "September",
  "October",
  "November",
  "December",
] as const;
export type Month = (typeof MONTHS)[number];

export const DAYS_OF_WEEK = ["Friday", "Saturday", "Tuesday", "Wednesday"] as const;
export type DayOfWeek = (typeof DAYS_OF_WEEK)[number];

export const SESSION_PERIODS = ["morning", "afternoon"] as const;
export type SessionPeriod = (typeof SESSION_PERIODS)[number];

export const GENDERS = ["F", "M"] as const;
export type Gender = (typeof GENDERS)[number];

export const ADDRESSES = [
  "In the city",
  "Out of city",
  "Out of province",
  "Unknown",
] as const;
export type Address = (typeof ADDRESSES)[number];

/** Request body for /predict/classification. */
export interface ConsultationInput {
  visit_number: number;
  is_working_day: boolean;
  has_primary_cancer: boolean;
  has_secondary_cancer: boolean;
  month: Month;
  day_of_week: DayOfWeek;
  session: SessionPeriod;
  gender: Gender;
  address: Address;
}

export interface ClassificationModelPrediction {
  model: string;
  is_long_consultation: boolean;
  probability_long: number;
  probability_short: number;
}

export interface ClassificationOutput {
  best_model: string;
  predictions: ClassificationModelPrediction[];
}

export interface RegressionModelPrediction {
  model: string;
  predicted_duration_minutes: number;
}

export interface RegressionOutput {
  best_model: string;
  predictions: RegressionModelPrediction[];
}

/** Human-readable labels for model-ladder keys returned by the API. */
export const MODEL_LABELS: Record<string, string> = {
  dummy: "Baseline (Dummy)",
  logistic_regression: "Logistic Regression",
  ridge: "Ridge Regression",
  decision_tree: "Decision Tree",
  random_forest: "Random Forest",
  xgboost: "XGBoost",
};

/** Baseline -> linear -> single tree -> ensembles: simplest model first. */
const MODEL_DISPLAY_ORDER = [
  "dummy",
  "logistic_regression",
  "ridge",
  "decision_tree",
  "random_forest",
  "xgboost",
];

export function sortByModelOrder<T extends { model: string }>(rows: T[]): T[] {
  return [...rows].sort(
    (a, b) => MODEL_DISPLAY_ORDER.indexOf(a.model) - MODEL_DISPLAY_ORDER.indexOf(b.model),
  );
}

/** Shape of a FastAPI validation error response (HTTP 422). */
export interface ApiValidationError {
  detail: Array<{ loc: (string | number)[]; msg: string; type: string }>;
}
