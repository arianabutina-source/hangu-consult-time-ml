import { useEffect, useState } from "react";

interface UseJsonDataResult<T> {
  data: T | null;
  error: string | null;
  isLoading: boolean;
}

/** Fetches a static JSON file (served from /public/data, see Milestone 15 export script). */
export function useJsonData<T>(path: string): UseJsonDataResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    setIsLoading(true);
    setError(null);
    fetch(path)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Failed to load ${path} (status ${response.status})`);
        }
        return response.json() as Promise<T>;
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load dashboard data.");
        }
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [path]);

  return { data, error, isLoading };
}
