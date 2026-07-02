import type {
  ApiValidationError,
  ClassificationOutput,
  ConsultationInput,
  RegressionOutput,
} from "../types/api";

const API_BASE_URL: string = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function postJson<T>(path: string, body: ConsultationInput): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    try {
      const errorBody = (await response.json()) as ApiValidationError;
      if (errorBody.detail?.length) {
        message = errorBody.detail
          .map((e) => `${e.loc.at(-1)}: ${e.msg}`)
          .join("; ");
      }
    } catch {
      // response body wasn't JSON; keep the generic message
    }
    throw new ApiError(message, response.status);
  }

  return response.json() as Promise<T>;
}

export function predictClassification(
  payload: ConsultationInput,
): Promise<ClassificationOutput> {
  return postJson<ClassificationOutput>("/api/v1/predict/classification", payload);
}

export function predictRegression(payload: ConsultationInput): Promise<RegressionOutput> {
  return postJson<RegressionOutput>("/api/v1/predict/regression", payload);
}
