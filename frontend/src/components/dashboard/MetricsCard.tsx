interface MetricsCardProps {
  label: string;
  value: string;
  hint?: string;
}

export function MetricsCard({ label, value, hint }: MetricsCardProps) {
  return (
    <div className="rounded-xl border border-espresso/5 bg-white p-4 shadow-sm">
      <p className="text-sm font-medium text-espresso-light">{label}</p>
      <p className="mt-1 font-serif text-2xl font-medium text-terracotta-dark">{value}</p>
      {hint && <p className="mt-1 text-xs text-espresso-light/70">{hint}</p>}
    </div>
  );
}
