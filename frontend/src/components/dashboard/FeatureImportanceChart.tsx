import { Bar, BarChart, CartesianGrid, Tooltip, XAxis, YAxis } from "recharts";
import { useJsonData } from "../../hooks/useJsonData";
import type { FeatureImportanceRecord } from "../../types/dashboard";

interface FeatureImportanceChartProps {
  dataUrl: string;
}

export function FeatureImportanceChart({ dataUrl }: FeatureImportanceChartProps) {
  const { data, error, isLoading } = useJsonData<FeatureImportanceRecord[]>(dataUrl);

  if (isLoading) return <p className="text-sm text-slate-500">Loading feature importance…</p>;
  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!data) return null;

  const chartData = [...data].reverse(); // highest importance at the top

  return (
    <BarChart
      width={520}
      height={320}
      data={chartData}
      layout="vertical"
      margin={{ top: 10, right: 20, bottom: 10, left: 40 }}
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis type="number" />
      <YAxis type="category" dataKey="feature" width={160} tick={{ fontSize: 12 }} />
      <Tooltip formatter={(value) => (typeof value === "number" ? value.toFixed(4) : value)} />
      <Bar dataKey="importance" fill="#02c39a" />
    </BarChart>
  );
}
