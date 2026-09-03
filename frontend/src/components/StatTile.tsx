export default function StatTile({
  label,
  value,
  sub,
}: {
  label: string;
  value: number | string;
  sub?: string;
}) {
  return (
    <div>
      <div style={{ fontFamily: "var(--font-display)", fontSize: 32, fontWeight: 700, color: "var(--text-accent)" }}>
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{sub}</div>
      )}
      <div
        style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          letterSpacing: "var(--tracking-widest)",
          textTransform: "uppercase",
          color: "var(--text-muted)",
          marginTop: 2,
        }}
      >
        {label}
      </div>
    </div>
  );
}
