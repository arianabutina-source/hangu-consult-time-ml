import { CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { useJsonData } from "../../hooks/useJsonData";
import type { RocCurvePoint } from "../../types/dashboard";

interface RocCurveChartProps {
  dataUrl: string;
}

export function RocCurveChart({ dataUrl }: RocCurveChartProps) {
  const { data, error, isLoading } = useJsonData<RocCurvePoint[]>(dataUrl);

  if (isLoading) return <p className="text-sm text-slate-500">Loading ROC curve…</p>;
  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!data) return null;

  const chanceLine = data.map((p) => ({ fpr: p.fpr, chance: p.fpr }));
  const merged = data.map((p, i) => ({ ...p, chance: chanceLine[i].chance }));

  return (
    <LineChart width={480} height={320} data={merged} margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis
        type="number"
        dataKey="fpr"
        domain={[0, 1]}
        label={{ value: "False Positive Rate", position: "insideBottom", offset: -10 }}
      />
      <YAxis type="number" domain={[0, 1]} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft" }} />
      <Tooltip formatter={(value) => (typeof value === "number" ? value.toFixed(3) : value)} />
      <Legend verticalAlign="top" height={24} />
      <Line type="monotone" dataKey="tpr" name="ROC curve" stroke="#4f46e5" dot={false} strokeWidth={2} />
      <Line type="monotone" dataKey="chance" name="Chance" stroke="#94a3b8" dot={false} strokeDasharray="4 4" />
    </LineChart>
  );
}
