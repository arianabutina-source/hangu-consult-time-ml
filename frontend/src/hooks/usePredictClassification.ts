import { useCallback, useState } from "react";
import { ApiError, predictClassification } from "../api/client";
import type { ClassificationOutput, ConsultationInput } from "../types/api";

interface UsePredictClassificationResult {
  data: ClassificationOutput | null;
  error: string | null;
  isLoading: boolean;
  predict: (payload: ConsultationInput) => Promise<void>;
  reset: () => void;
}

export function usePredictClassification(): UsePredictClassificationResult {
  const [data, setData] = useState<ClassificationOutput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const predict = useCallback(async (payload: ConsultationInput) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await predictClassification(payload);
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
