export default function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="9" stroke="#00d4a4" strokeWidth="1.4" />
      <path d="M12 3v3M12 18v3M3 12h3M18 12h3" stroke="#00d4a4" strokeWidth="1.4" strokeLinecap="round" />
      <circle cx="12" cy="12" r="2.5" stroke="#00d4a4" strokeWidth="1.4" />
    </svg>
  );
}
