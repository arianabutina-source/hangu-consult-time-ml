import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { usePredictRegression } from "../../hooks/usePredictRegression";
import { MODEL_LABELS, sortByModelOrder } from "../../types/api";
import { ConsultationFormFields } from "./ConsultationFormFields";
import { consultationSchema, defaultConsultationValues, type ConsultationFormValues } from "./schema";

export function RegressionForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ConsultationFormValues>({
    resolver: zodResolver(consultationSchema),
    defaultValues: defaultConsultationValues,
  });
  const { data, error, isLoading, predict } = usePredictRegression();

  const onSubmit = handleSubmit((values) => predict(values));

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <ConsultationFormFields register={register} errors={errors} />

      <button
        type="submit"
        disabled={isLoading}
        className="rounded-full bg-terracotta px-6 py-2.5 text-sm font-semibold text-cream shadow-sm transition-colors hover:bg-terracotta-dark disabled:opacity-50"
      >
        {isLoading ? "Predicting…" : "Predict"}
      </button>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {data && (
        <div className="space-y-2">
          <p className="text-xs font-semibold tracking-wide text-espresso-light uppercase">
            Prediction by model
          </p>
          {sortByModelOrder(data.predictions).map((row) => {
            const isBest = row.model === data.best_model;
            return (
              <div
                key={row.model}
                className={`flex items-center justify-between gap-2 rounded-xl border p-4 ${
                  isBest
                    ? "border-terracotta/30 bg-terracotta-light"
                    : "border-espresso/10 bg-white"
                }`}
              >
                <p
                  className={`font-serif text-lg font-medium ${
                    isBest ? "text-terracotta-dark" : "text-espresso"
                  }`}
                >
                  {row.predicted_duration_minutes.toFixed(1)} minutes
                </p>
                <span
                  className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                    isBest ? "bg-terracotta-dark text-cream" : "bg-espresso/5 text-espresso-light"
                  }`}
                >
                  {MODEL_LABELS[row.model] ?? row.model}
                  {isBest ? " · deployed" : ""}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </form>
  );
}
