import type { UseFormRegister, FieldErrors } from "react-hook-form";
import { ADDRESSES, DAYS_OF_WEEK, GENDERS, MONTHS, SESSION_PERIODS } from "../../types/api";
import type { ConsultationFormValues } from "./schema";

interface ConsultationFormFieldsProps {
  register: UseFormRegister<ConsultationFormValues>;
  errors: FieldErrors<ConsultationFormValues>;
}

/** The 9 raw predictor fields shared by both prediction forms. */
export function ConsultationFormFields({ register, errors }: ConsultationFormFieldsProps) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Visit number
        <input
          type="number"
          min={1}
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
          {...register("visit_number", { valueAsNumber: true })}
        />
        {errors.visit_number && (
          <span className="text-xs text-red-600">{errors.visit_number.message}</span>
        )}
      </label>

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Month
        <select
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
          {...register("month")}
        >
          {MONTHS.map((month) => (
            <option key={month} value={month}>
              {month}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Day of week
        <select
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
          {...register("day_of_week")}
        >
          {DAYS_OF_WEEK.map((day) => (
            <option key={day} value={day}>
              {day}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Session
        <select
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
          {...register("session")}
        >
          {SESSION_PERIODS.map((session) => (
            <option key={session} value={session}>
              {session}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Gender
        <select
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
          {...register("gender")}
        >
          {GENDERS.map((gender) => (
            <option key={gender} value={gender}>
              {gender === "F" ? "Female" : "Male"}
            </option>
          ))}
        </select>
      </label>

      <label className="flex flex-col gap-1 text-sm font-medium text-slate-700">
        Address
        <select
          className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
          {...register("address")}
        >
          {ADDRESSES.map((address) => (
            <option key={address} value={address}>
              {address}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
        <input type="checkbox" className="h-4 w-4 rounded border-slate-300" {...register("is_working_day")} />
        Working day
      </label>

      <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
        <input
          type="checkbox"
          className="h-4 w-4 rounded border-slate-300"
          {...register("has_primary_cancer")}
        />
        Primary cancer diagnosis
      </label>

      <label className="flex items-center gap-2 text-sm font-medium text-slate-700">
        <input
          type="checkbox"
          className="h-4 w-4 rounded border-slate-300"
          {...register("has_secondary_cancer")}
        />
        Secondary cancer diagnosis
      </label>
    </div>
  );
}
