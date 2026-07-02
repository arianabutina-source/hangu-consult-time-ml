import { Link } from "react-router-dom";
import { Logo } from "./Logo";

const pages = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Model Results" },
  { to: "/predict/classification", label: "Live Prediction" },
  { to: "/scheduling-script", label: "Scheduling Script" },
  { to: "/pricelist", label: "Pricelist" },
  { to: "/team", label: "Team" },
];

const stack = [
  "FastAPI + scikit-learn backend",
  "Random Forest classification & regression",
  "Grouped, leakage-safe cross-validation",
  "Vite + React + Recharts",
];

export function Footer() {
  return (
    <footer className="bg-espresso text-cream/70">
      <div className="mx-auto grid max-w-6xl grid-cols-1 gap-10 px-6 py-14 sm:grid-cols-3">
        <div>
          <div className="flex items-center gap-2">
            <Logo />
            <span className="font-serif text-lg font-medium text-cream">
              Optimised Scheduling Tool
            </span>
          </div>
          <p className="mt-3 max-w-xs text-sm">
            Outpatient Consultation Duration Predictor — supporting optimised outpatient
            scheduling.
          </p>
        </div>
        <div>
          <h3 className="font-serif text-sm font-medium text-cream">Pages</h3>
          <ul className="mt-3 space-y-2 text-sm">
            {pages.map((page) => (
              <li key={page.to}>
                <Link to={page.to} className="transition-colors hover:text-terracotta">
                  {page.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="font-serif text-sm font-medium text-cream">Stack</h3>
          <ul className="mt-3 space-y-2 text-sm">
            {stack.map((item) => (
              <li key={item} className="text-terracotta-light">
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="border-t border-cream/10">
        <div className="mx-auto max-w-6xl px-6 py-6 text-xs text-cream/50">
          <p>
            <span className="font-medium text-cream/70">Data source:</span> Feng, H., Jia,
            Y., Zhou, S., Chen, H., &amp; Huang, T. (2023). "A Dataset of Service Time and
            Related Patient Characteristics from an Outpatient Clinic." <em>Data</em>, 8(3),
            47.{" "}
            <a
              href="https://doi.org/10.5281/zenodo.7484205"
              target="_blank"
              rel="noreferrer"
              className="underline transition-colors hover:text-terracotta"
            >
              doi.org/10.5281/zenodo.7484205
            </a>
            . Hangu Open Data, licensed CC BY-NC-SA 4.0.
          </p>
          <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <span>© 2026 Hangu outpatient oncology clinic — academic capstone project</span>
            <span>Predictions are estimates from historical data, not clinical guidance.</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
