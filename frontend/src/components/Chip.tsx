export default function Chip({
  children,
  dot,
  tone,
}: {
  children: React.ReactNode;
  dot?: boolean;
  tone?: "neutral";
}) {
  const isNeutral = tone === "neutral";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "5px 12px",
        borderRadius: "var(--radius-pill)",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        whiteSpace: "nowrap",
        background: isNeutral ? "var(--surface-inset)" : "var(--fill-accent-wash)",
        border: `1px solid ${isNeutral ? "var(--border-default)" : "var(--border-accent-soft)"}`,
        color: isNeutral ? "var(--text-muted)" : "var(--text-accent)",
      }}
    >
      {dot && (
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            background: "var(--tva-accent)",
            animation: "tvaPulse 2s ease-in-out infinite",
          }}
        />
      )}
      {children}
    </span>
  );
}
