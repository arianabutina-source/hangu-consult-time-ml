const doctors = [
  {
    name: "Dr. Elena Rossi, MD",
    specialty: "Dermatology",
    bio: "Italian board-certified dermatologist with 14+ years of experience. Specializes in medical dermatology, acne, rosacea and laser therapies. Passionate about skin health and natural-looking results.",
  },
  {
    name: "Dr. James Thornton, MD",
    specialty: "Dermatology",
    bio: "British dermatologist with a focus on skin cancer prevention and early detection. Expert in mole mapping, dermoscopy and clinical research. Known for his thorough approach and patient education.",
  },
  {
    name: "Dr. Mei Ling Tan, MD",
    specialty: "Dermatology",
    bio: "Singapore-trained dermatologist specializing in aesthetic dermatology, pigmentation disorders and anti-aging treatments. Expert in injectables and skin rejuvenation technologies. Committed to enhancing confidence and well-being.",
  },
  {
    name: "Dr. Alejandro Ruiz, MD, PhD",
    specialty: "Plastic Surgery",
    bio: "Spanish board-certified plastic surgeon with 15+ years of experience. Specializes in facial surgery and body contouring. Known for artistic precision, safety and personalized surgical planning.",
  },
  {
    name: "Dr. Sarah Williams, MD",
    specialty: "Plastic Surgery",
    bio: "Australian plastic surgeon specializing in breast surgery and post-weight loss body contouring. Focused on natural, harmonious results and holistic patient care.",
  },
  {
    name: "Dr. Marc Dubois, MD",
    specialty: "General Surgery",
    bio: "French general surgeon with 20+ years of experience. Expert in minimally invasive surgery and hernia repair. Dedicated to excellence and compassionate patient care.",
  },
  {
    name: "Dr. Aisha Rahman, MD",
    specialty: "General Surgery",
    bio: "Canadian general surgeon specializing in abdominal surgery and gastrointestinal disorders. Advocate for patient safety, clear communication and health education.",
  },
];

const cosmeticSpecialists = [
  {
    name: "Gabriella Moretti",
    title: "Lead Cosmetic Specialist",
    bio: "Italian aesthetic expert with 10+ years of experience in non-surgical facial rejuvenation. Specialist in injectables, skin boosters and facial harmonization. Known for subtle, elegant results.",
  },
  {
    name: "Sophie Laurent",
    title: "Cosmetic Specialist",
    bio: "French specialist in advanced skin therapies and laser treatments. Expert in chemical peels, microneedling and anti-aging protocols. Passionate about skin health and confidence.",
  },
  {
    name: "Lukas Schneider",
    title: "Cosmetic Specialist",
    bio: "German cosmetic specialist with expertise in body contouring technologies and regenerative treatments. Focused on innovation, safety and long-term results.",
  },
];

const nurses = [
  "Olivia Bennett",
  "Amelia Carter",
  "Yuki Nakamura",
  "Isabella Conti",
  "Maria Gonzalez",
  "Nora Lindberg",
  "Fatima Al-Mansoori",
  "Elena Popescu",
  "Sara Kowalski",
];

function initials(fullName: string): string {
  const cleaned = fullName.replace(/^Dr\.\s*/, "").split(",")[0].trim();
  return cleaned
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

const avatarSizes = {
  large: "aspect-square w-full text-3xl",
  medium: "h-16 w-16 flex-shrink-0 text-lg",
  small: "aspect-square w-full text-xl",
};

function Avatar({
  name,
  size = "large",
}: {
  name: string;
  size?: keyof typeof avatarSizes;
}) {
  return (
    <div
      className={
        "flex items-center justify-center rounded-xl bg-terracotta-light font-serif font-medium text-terracotta-dark " +
        avatarSizes[size]
      }
    >
      {initials(name)}
    </div>
  );
}

function SectionHeading({ children }: { children: string }) {
  return (
    <div className="mb-6 flex items-center gap-4">
      <div className="h-px flex-1 bg-espresso/10" />
      <h2 className="font-serif text-2xl font-medium tracking-wide text-espresso uppercase">
        {children}
      </h2>
      <div className="h-px flex-1 bg-espresso/10" />
    </div>
  );
}

export function TeamPage() {
  return (
    <div className="mx-auto max-w-6xl px-6 py-14">
      <span className="text-xs font-semibold tracking-wide text-terracotta uppercase">
        Our Team
      </span>
      <h1 className="mt-2 mb-12 font-serif text-3xl font-medium text-espresso sm:text-4xl">
        Meet the clinicians behind your care
      </h1>

      <section className="mb-16">
        <SectionHeading>Doctors</SectionHeading>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {doctors.map((doctor) => (
            <div
              key={doctor.name}
              className="rounded-xl border border-espresso/5 bg-white p-4 shadow-sm"
            >
              <Avatar name={doctor.name} />
              <h3 className="mt-3 font-serif text-lg font-medium text-espresso">
                {doctor.name}
              </h3>
              <p className="text-xs font-semibold tracking-wide text-terracotta-dark uppercase">
                {doctor.specialty}
              </p>
              <p className="mt-2 text-sm text-espresso-light">{doctor.bio}</p>
            </div>
          ))}
        </div>
      </section>

      <div className="grid grid-cols-1 gap-16 lg:grid-cols-[1fr_1.4fr]">
        <section>
          <SectionHeading>Cosmetic Specialists</SectionHeading>
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-3 lg:grid-cols-1">
            {cosmeticSpecialists.map((specialist) => (
              <div
                key={specialist.name}
                className="flex gap-4 rounded-xl border border-espresso/5 bg-white p-4 shadow-sm"
              >
                <Avatar name={specialist.name} size="medium" />
                <div>
                  <h3 className="font-serif text-lg font-medium text-espresso">
                    {specialist.name}
                  </h3>
                  <p className="text-xs font-semibold tracking-wide text-terracotta-dark uppercase">
                    {specialist.title}
                  </p>
                  <p className="mt-2 text-sm text-espresso-light">{specialist.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <SectionHeading>Nurses</SectionHeading>
          <div className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-5">
            {nurses.map((nurse) => (
              <div key={nurse} className="text-center">
                <Avatar name={nurse} size="small" />
                <h3 className="mt-2 text-sm font-medium text-espresso">{nurse}</h3>
                <p className="text-xs font-semibold tracking-wide text-terracotta-dark uppercase">
                  Registered Nurse
                </p>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
