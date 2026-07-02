const steps = [
  {
    title: "Ask why they are seeking care.",
    description:
      "Listen actively, understand their concerns, and build rapport. Use this information to explain how your clinic can help.",
  },
  {
    title: "Reassure them that you can help.",
    description:
      "Briefly explain why your clinic is the right choice, highlighting relevant expertise or experience without sounding overly promotional.",
  },
  {
    title: "Offer appointment options.",
    description:
      "Provide two available appointment times instead of asking them to choose freely. Explain reminder options and briefly mention the cancellation policy.",
  },
  {
    title: "Invite questions.",
    description:
      "Ask if they have any questions about the appointment, treatment, parking, preparation, or anything else. Address any concerns and direct them to helpful resources if needed.",
  },
  {
    title: "Collect insurance and patient information.",
    description:
      "Confirm insurance details, verify coverage, gather any required demographic information, and remind the patient to complete any necessary forms before their visit.",
  },
  {
    title: "Ask how they heard about the clinic.",
    description:
      "Record the referral source to help track marketing effectiveness and patient referrals.",
  },
];

export function SchedulingScriptPage() {
  return (
    <div className="mx-auto max-w-3xl px-6 py-14">
      <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
        Communication with a Patient
      </span>
      <h1 className="mt-2 font-serif text-3xl font-medium text-espresso sm:text-4xl">
        How to schedule an appointment?
      </h1>

      <ol className="mt-8 space-y-5">
        {steps.map((step, index) => (
          <li
            key={step.title}
            className="flex gap-4 rounded-xl border border-espresso/5 bg-white p-6 shadow-sm"
          >
            <span className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-terracotta font-serif text-sm font-semibold text-cream">
              {index + 1}
            </span>
            <div>
              <h3 className="font-serif text-lg font-medium text-espresso">{step.title}</h3>
              <p className="mt-1 text-sm text-espresso-light">{step.description}</p>
            </div>
          </li>
        ))}
      </ol>

      <div className="mt-6 rounded-xl border border-terracotta/20 bg-terracotta-light p-6">
        <h3 className="font-serif text-lg font-medium text-terracotta-dark">Throughout</h3>
        <p className="mt-1 text-sm text-espresso-light">
          Be warm, empathetic, and professional. Focus on making the patient feel heard,
          reassured, and confident that they are in good hands.
        </p>
      </div>
    </div>
  );
}
