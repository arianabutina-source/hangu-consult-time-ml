import { z } from "zod";
import { ADDRESSES, DAYS_OF_WEEK, GENDERS, MONTHS, SESSION_PERIODS } from "../../types/api";

// Mirrors backend/app/schemas/common.py's ConsultationInput validation.
// Numeric/boolean coercion happens in react-hook-form's `register` options
// (valueAsNumber / native checkbox `checked`), not here, so the schema's
// input and output types match exactly -- this keeps zodResolver's generics
// simple and avoids a widened `unknown` input type.
export const consultationSchema = z.object({
  visit_number: z.number().int().positive("Visit number must be a positive integer."),
  is_working_day: z.boolean(),
  has_primary_cancer: z.boolean(),
  has_secondary_cancer: z.boolean(),
  month: z.enum(MONTHS),
  day_of_week: z.enum(DAYS_OF_WEEK),
  session: z.enum(SESSION_PERIODS),
  gender: z.enum(GENDERS),
  address: z.enum(ADDRESSES),
});

export type ConsultationFormValues = z.infer<typeof consultationSchema>;

export const defaultConsultationValues: ConsultationFormValues = {
  visit_number: 1,
  is_working_day: true,
  has_primary_cancer: false,
  has_secondary_cancer: false,
  month: "January",
  day_of_week: "Wednesday",
  session: "morning",
  gender: "F",
  address: "In the city",
};
