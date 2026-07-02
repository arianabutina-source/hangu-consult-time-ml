/** Simple line-art sketch of a nurse on the phone while looking at a screen. */
export function NurseIllustration() {
  return (
    <svg
      viewBox="0 0 400 300"
      className="mx-auto w-full max-w-md"
      fill="none"
      role="img"
      aria-label="Sketch of a nurse talking on the phone while looking at a computer screen"
    >
      {/* Desk */}
      <line x1="20" y1="222" x2="380" y2="222" stroke="var(--color-espresso)" strokeWidth="2.5" strokeLinecap="round" />

      {/* Monitor */}
      <rect x="220" y="82" width="120" height="92" rx="6" stroke="var(--color-espresso)" strokeWidth="2.5" />
      <line x1="255" y1="192" x2="305" y2="192" stroke="var(--color-espresso)" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="280" y1="174" x2="280" y2="192" stroke="var(--color-espresso)" strokeWidth="2.5" />
      {/* Screen content: a small bar chart + trend line, echoing the dashboard */}
      <rect x="236" y="140" width="12" height="24" fill="var(--color-terracotta)" opacity="0.5" />
      <rect x="256" y="128" width="12" height="36" fill="var(--color-terracotta)" opacity="0.7" />
      <rect x="276" y="118" width="12" height="46" fill="var(--color-terracotta)" />
      <rect x="296" y="132" width="12" height="32" fill="var(--color-terracotta)" opacity="0.6" />
      <path
        d="M234 108 L254 118 L274 98 L294 106 L314 96"
        stroke="var(--color-espresso)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Keyboard on the desk */}
      <rect x="150" y="204" width="60" height="18" rx="3" stroke="var(--color-espresso)" strokeWidth="2" />

      {/* Nurse torso */}
      <path
        d="M90 222 C88 170 96 138 130 138 C164 138 172 170 170 222"
        stroke="var(--color-espresso)"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />

      {/* Left arm resting toward the keyboard */}
      <path
        d="M108 150 C120 175 132 190 152 205"
        stroke="var(--color-espresso)"
        strokeWidth="2.5"
        strokeLinecap="round"
      />

      {/* Right arm holding the phone to her ear */}
      <path
        d="M150 145 C168 150 176 120 168 100"
        stroke="var(--color-espresso)"
        strokeWidth="2.5"
        strokeLinecap="round"
      />

      {/* Head */}
      <circle cx="128" cy="104" r="30" stroke="var(--color-espresso)" strokeWidth="2.5" />

      {/* Nurse cap */}
      <path
        d="M104 82 C110 68 146 68 152 82"
        stroke="var(--color-espresso)"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      <line x1="124" y1="74" x2="132" y2="74" stroke="var(--color-terracotta)" strokeWidth="2.5" strokeLinecap="round" />
      <line x1="128" y1="70" x2="128" y2="78" stroke="var(--color-terracotta)" strokeWidth="2.5" strokeLinecap="round" />

      {/* Phone at her ear */}
      <rect
        x="160"
        y="86"
        width="16"
        height="28"
        rx="4"
        stroke="var(--color-terracotta)"
        strokeWidth="2.5"
        fill="var(--color-terracotta-light)"
      />

      {/* Call sound waves */}
      <path
        d="M184 92 C190 98 190 104 184 110"
        stroke="var(--color-terracotta)"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M192 86 C202 96 202 106 192 116"
        stroke="var(--color-terracotta)"
        strokeWidth="2"
        strokeLinecap="round"
        opacity="0.6"
      />

      {/* Simple facial detail: a friendly smile */}
      <path d="M116 112 Q128 120 140 112" stroke="var(--color-espresso)" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
