import { useCallback, useState } from "react";
import { ApiError, predictRegression } from "../api/client";
import type { ConsultationInput, RegressionOutput } from "../types/api";

interface UsePredictRegressionResult {
  data: RegressionOutput | null;
  error: string | null;
  isLoading: boolean;
  predict: (payload: ConsultationInput) => Promise<void>;
  reset: () => void;
}

export function usePredictRegression(): UsePredictRegressionResult {
  const [data, setData] = useState<RegressionOutput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const predict = useCallback(async (payload: ConsultationInput) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await predictRegression(payload);
      setData(result);
    } catch (err) {
      setData(null);
      setError(err instanceof ApiError ? err.message : "Unable to reach the prediction service.");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const reset = useCallback(() => {
    setData(null);
    setError(null);
  }, []);

  return { data, error, isLoading, predict, reset };
}
