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

/** Request body for both /predict/classification and /predict/regression. */
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

export interface ClassificationOutput {
  is_long_consultation: boolean;
  probability_long: number;
  probability_short: number;
}

export interface RegressionOutput {
  predicted_duration_minutes: number;
}

/** Shape of a FastAPI validation error response (HTTP 422). */
export interface ApiValidationError {
  detail: Array<{ loc: (string | number)[]; msg: string; type: string }>;
}
