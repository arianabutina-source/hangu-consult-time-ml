import { NavLink } from "react-router-dom";
import clsx from "clsx";
import { Logo } from "./Logo";

const links = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Appointment Length Calculator" },
  { to: "/predict/classification", label: "Long Visit Predictor" },
  { to: "/predict/regression", label: "Duration Estimate" },
  { to: "/scheduling-script", label: "Scheduling Script" },
  { to: "/pricelist", label: "Pricelist" },
  { to: "/team", label: "Team" },
];

export function Navbar() {
  return (
    <header className="border-b border-espresso/10 bg-cream/90 backdrop-blur">
      <nav className="mx-auto grid max-w-7xl grid-cols-[1fr_auto_1fr] items-center px-6 py-4">
        <NavLink to="/" className="flex items-center gap-2 justify-self-start">
          <Logo />
          <span className="font-serif text-lg font-medium text-espresso">
            Optimised Scheduling Tool
          </span>
        </NavLink>
        <div className="hidden items-center justify-self-center sm:flex">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                clsx(
                  "rounded-md px-2.5 py-2 text-center text-sm font-medium whitespace-nowrap transition-colors",
                  isActive
                    ? "text-terracotta"
                    : "text-espresso-light hover:text-espresso",
                )
              }
            >
              {link.label}
            </NavLink>
          ))}
        </div>
        <div className="justify-self-end" />
      </nav>
    </header>
  );
}
