import { Link } from "react-router-dom";
import { NurseIllustration } from "../components/home/NurseIllustration";

const benefits = [
  {
    icon: "📅",
    title: "Smarter Allocation",
    text: "Optimize appointment allocation and resource utilization.",
    badge: "bg-terracotta-light",
    tilt: "-rotate-2",
  },
  {
    icon: "⏱️",
    title: "Accurate Timing",
    text: "Predict consultation duration for more accurate scheduling.",
    badge: "bg-[#dbe9f1]",
    tilt: "rotate-2",
  },
  {
    icon: "🎯",
    title: "Fewer No-Shows",
    text: "Reduce patient waiting times, no-shows, and scheduling conflicts.",
    badge: "bg-cream-dark",
    tilt: "-rotate-1",
  },
  {
    icon: "📈",
    title: "Higher Productivity",
    text: "Improve staff productivity and overall clinic efficiency.",
    badge: "bg-[#dbe9f1]",
    tilt: "rotate-1",
  },
  {
    icon: "💛",
    title: "Happier Patients",
    text: "Enhance the patient experience through smoother, data-driven scheduling.",
    badge: "bg-terracotta-light",
    tilt: "-rotate-2",
  },
];

export function HomePage() {
  return (
    <div>
      <section className="bg-gradient-to-b from-cream-dark to-cream px-6 py-20">
        <div className="mx-auto max-w-3xl text-center">
          <span className="inline-flex items-center gap-2 rounded-full bg-terracotta-light px-4 py-1.5 text-xs font-semibold tracking-wide text-terracotta-dark uppercase">
            ML-Powered Clinic Intelligence
          </span>
          <h1 className="mt-6 font-serif text-5xl font-medium text-espresso sm:text-6xl">
            Smarter outpatient
            <br />
            scheduling decisions
          </h1>
          <p className="mt-4 font-serif text-xl text-terracotta-dark italic">
            Predict duration. Plan capacity. Cost it accurately.
          </p>
          <p className="mx-auto mt-5 max-w-xl text-espresso-light">
            Efficient scheduling is the foundation of every successful outpatient clinic.
            This tool combines a classification model and a regression model, trained on
            6,637 historical outpatient consultations, to optimize appointment scheduling
            and clinic operations.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <Link
              to="/predict/classification"
              className="rounded-full bg-terracotta px-6 py-3 text-sm font-semibold text-cream shadow-sm transition-colors hover:bg-terracotta-dark"
            >
              Try Live Prediction →
            </Link>
          </div>

          <div className="mt-12">
            <NurseIllustration />
          </div>
        </div>
      </section>

      <section className="px-6 py-20">
        <div className="mx-auto max-w-5xl text-center">
          <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
            Why Teams Love It
          </span>
          <h2 className="mt-3 font-serif text-3xl font-medium text-espresso sm:text-4xl">
            With this tool, you'll be able to:
          </h2>

          <div className="mt-12 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {benefits.map((benefit) => (
              <div
                key={benefit.title}
                className={`group rounded-2xl border border-espresso/5 bg-white p-6 text-left shadow-sm transition-all duration-200 ${benefit.tilt} hover:-translate-y-1 hover:rotate-0 hover:shadow-lg`}
              >
                <div
                  className={`flex h-14 w-14 items-center justify-center rounded-full text-2xl transition-transform duration-200 group-hover:scale-110 ${benefit.badge}`}
                >
                  {benefit.icon}
                </div>
                <h3 className="mt-4 font-serif text-lg font-medium text-espresso">
                  {benefit.title}
                </h3>
                <p className="mt-1 text-sm text-espresso-light">{benefit.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
