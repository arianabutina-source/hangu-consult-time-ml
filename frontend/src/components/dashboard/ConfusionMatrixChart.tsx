import { useJsonData } from "../../hooks/useJsonData";
import type { ConfusionMatrixData } from "../../types/dashboard";

interface ConfusionMatrixChartProps {
  dataUrl: string;
}

export function ConfusionMatrixChart({ dataUrl }: ConfusionMatrixChartProps) {
  const { data, error, isLoading } = useJsonData<ConfusionMatrixData>(dataUrl);

  if (isLoading) return <p className="text-sm text-slate-500">Loading confusion matrix…</p>;
  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!data) return null;

  const max = Math.max(...data.matrix.flat());
  const cellStyle = (value: number) => ({
    backgroundColor: `rgba(79, 70, 229, ${0.15 + 0.65 * (value / max)})`,
  });

  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-center text-sm">
        <div />
        {data.labels.map((label) => (
          <div key={`pred-${label}`} className="font-medium text-slate-600">
            Predicted {label}
          </div>
        ))}
        {data.matrix.map((row, rowIndex) => (
          <div key={data.labels[rowIndex]} className="contents">
            <div className="flex items-center justify-end pr-2 font-medium text-slate-600">
              Actual {data.labels[rowIndex]}
            </div>
            {row.map((value, colIndex) => (
              <div
                key={colIndex}
                style={cellStyle(value)}
                className="flex items-center justify-center rounded-md py-6 text-lg font-semibold text-indigo-950"
              >
                {value}
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
