// The ML pipeline's ColumnTransformer names its output columns like
// "numeric__Visit.No" or "categorical__Address_Out of city" -- accurate for
// debugging, unreadable for a dashboard. This maps them to plain-language
// labels for display only; the underlying data/keys are untouched.

const BASE_FEATURE_LABELS: Record<string, string> = {
  "Visit.No": "Visit Number",
  IsRepeatVisit: "Repeat Visit",
  WorkingDay: "Working Day",
  IsPublicHolidayMakeup: "Public Holiday Makeup",
  HasCancerDiagnosis: "Cancer Diagnosis",
  "M.Cancer": "Primary Cancer",
  "S.Cancer": "Secondary Cancer",
};

// Checked longest-prefix-first isn't required here since these are matched
// as whole "<Group>_" prefixes, but the order still avoids "AM_PM" being
// mistaken for a group named "AM".
const CATEGORY_GROUP_LABELS: [prefix: string, label: string][] = [
  ["DayOfWeek_", "Day"],
  ["AM_PM_", "Session"],
  ["Address_", "Address"],
  ["Month_", "Month"],
  ["Gender_", "Gender"],
];

const VALUE_LABELS: Record<string, string> = {
  F: "Female",
  M: "Male",
  morning: "Morning",
  afternoon: "Afternoon",
  "In the city": "In the City",
  "Out of city": "Out of City",
  "Out of province": "Out of Province",
  Unknown: "Unknown",
};

/** Convert a raw ColumnTransformer output name into a human-readable label. */
export function prettifyFeatureName(rawName: string): string {
  const withoutPrefix = rawName.replace(/^(numeric|boolean|categorical)__/, "");

  for (const [prefix, groupLabel] of CATEGORY_GROUP_LABELS) {
    if (withoutPrefix.startsWith(prefix)) {
      const rawValue = withoutPrefix.slice(prefix.length);
      const value = VALUE_LABELS[rawValue] ?? rawValue;
      return `${groupLabel}: ${value}`;
    }
  }

  return BASE_FEATURE_LABELS[withoutPrefix] ?? withoutPrefix;
}
