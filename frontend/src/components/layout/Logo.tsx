export function Logo() {
  return (
    <svg
      width="28"
      height="28"
      viewBox="0 0 28 28"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      <circle cx="14" cy="14" r="14" fill="var(--color-terracotta)" />
      <circle cx="14" cy="14" r="8.5" stroke="var(--color-cream)" strokeWidth="1.4" />
      <path
        d="M14 9.5V14L17 16.5"
        stroke="var(--color-cream)"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
