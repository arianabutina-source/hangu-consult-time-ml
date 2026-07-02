import { NavLink } from "react-router-dom";
import clsx from "clsx";
import { Logo } from "./Logo";

const links = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Appointment Length Calculator" },
  { to: "/predict/classification", label: "Live Prediction" },
  { to: "/scheduling-script", label: "Scheduling Script" },
  { to: "/pricelist", label: "Pricelist" },
  { to: "/team", label: "Team" },
];

export function Navbar() {
  return (
    <header className="border-b border-espresso/10 bg-cream/90 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <NavLink to="/" className="flex items-center gap-2">
          <Logo />
          <span className="font-serif text-lg font-medium text-espresso">
            Optimised Scheduling Tool
          </span>
        </NavLink>
        <div className="hidden items-center gap-1 sm:flex">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.to === "/"}
              className={({ isActive }) =>
                clsx(
                  "rounded-md px-3 py-2 text-sm font-medium transition-colors",
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
      </nav>
    </header>
  );
}
