import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { usePredictClassification } from "../../hooks/usePredictClassification";
import { ConsultationFormFields } from "./ConsultationFormFields";
import { consultationSchema, defaultConsultationValues, type ConsultationFormValues } from "./schema";

export function ClassificationForm() {
  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<ConsultationFormValues>({
    resolver: zodResolver(consultationSchema),
    defaultValues: defaultConsultationValues,
  });
  const { data, error, isLoading, predict } = usePredictClassification();

  const onSubmit = handleSubmit((values) => predict(values));

  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <ConsultationFormFields register={register} errors={errors} />

      <button
        type="submit"
        disabled={isLoading}
        className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-indigo-500 disabled:opacity-50"
      >
        {isLoading ? "Predicting…" : "Predict"}
      </button>

      {error && <p className="text-sm text-red-600">{error}</p>}

      {data && (
        <div className="rounded-lg border border-indigo-100 bg-indigo-50 p-4">
          <p className="text-lg font-semibold text-indigo-900">
            {data.is_long_consultation ? "Long consultation" : "Short consultation"}
          </p>
          <p className="mt-1 text-sm text-indigo-700">
            P(long) = {(data.probability_long * 100).toFixed(1)}% &middot; P(short) ={" "}
            {(data.probability_short * 100).toFixed(1)}%
          </p>
        </div>
      )}
    </form>
  );
}
