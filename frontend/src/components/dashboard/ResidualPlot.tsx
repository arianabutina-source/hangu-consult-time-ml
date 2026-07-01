import {
  CartesianGrid,
  ReferenceLine,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useJsonData } from "../../hooks/useJsonData";
import type { ResidualPoint } from "../../types/dashboard";

interface ResidualPlotProps {
  dataUrl: string;
}

export function ResidualPlot({ dataUrl }: ResidualPlotProps) {
  const { data, error, isLoading } = useJsonData<ResidualPoint[]>(dataUrl);

  if (isLoading) return <p className="text-sm text-slate-500">Loading residuals…</p>;
  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!data) return null;

  return (
    <ScatterChart width={480} height={320} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis
        type="number"
        dataKey="predicted"
        name="Predicted duration"
        unit=" min"
        label={{ value: "Predicted duration (minutes)", position: "insideBottom", offset: -10 }}
      />
      <YAxis
        type="number"
        dataKey="residual"
        name="Residual"
        unit=" min"
        label={{ value: "Residual (minutes)", angle: -90, position: "insideLeft" }}
      />
      <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="4 4" />
      <Tooltip cursor={{ strokeDasharray: "3 3" }} />
      <Scatter data={data} fill="#4f46e5" fillOpacity={0.5} />
    </ScatterChart>
  );
}
