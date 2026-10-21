// One measure over a coach's sessions, with the benchmark band behind it.
// Single series, so no legend: the card's title names it.

import type { Band, MetricDef } from "@/lib/metrics";
import { judge } from "@/lib/metrics";

export interface Point { date: string; value: number | null; title: string; href: string }

export function Trend({ def, points, band }: { def: MetricDef; points: Point[]; band: Band | null }) {
  const W = 320, H = 120, L = 34, R = 8, T = 10, B = 22;
  const vals = points.map((p) => p.value).filter((v): v is number => v != null);
  const pctScale = def.format(0.5).endsWith("%");
  const max = pctScale ? 1 : Math.max(...vals, band?.high ?? 0) * 1.15 || 1;
  const x = (i: number) => L + (points.length <= 1 ? (W - L - R) / 2 : (i * (W - L - R)) / (points.length - 1));
  const y = (v: number) => T + (H - T - B) * (1 - Math.min(1, Math.max(0, v / max)));
  const line = points.map((p, i) => (p.value == null ? null : `${x(i)},${y(p.value)}`)).filter(Boolean).join(" ");
  const ticks = [0, max / 2, max];
  return (
    <svg className="trend" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${def.label} by session`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={L} x2={W - R} y1={y(t)} y2={y(t)} stroke="var(--rule)" strokeWidth={1} />
          <text x={L - 6} y={y(t) + 3.5} textAnchor="end">{def.format(t)}</text>
        </g>
      ))}
      {band && <rect x={L} width={W - L - R} y={y(band.high)} height={Math.max(1, y(band.low) - y(band.high))} fill="var(--band)" opacity={0.85} />}
      <polyline points={line} fill="none" stroke="var(--ink-2)" strokeWidth={2} strokeLinejoin="round" />
      {points.map((p, i) => p.value == null ? null : (
        <a key={i} href={p.href}>
          <circle cx={x(i)} cy={y(p.value)} r={10} fill="transparent" />
          <circle cx={x(i)} cy={y(p.value)} r={4.5} stroke="var(--surface)" strokeWidth={2}
            fill={{ good: "var(--good)", within: "var(--ink-3)", short: "var(--look)", none: "var(--ink-2)" }[judge(def, p.value, band)]}>
            <title>{`${p.title}: ${def.format(p.value)}`}</title>
          </circle>
        </a>
      ))}
      {points.length > 0 && <>
        <text x={x(0)} y={H - 6} textAnchor="start">{points[0].date.slice(5)}</text>
        {points.length > 1 && <text x={x(points.length - 1)} y={H - 6} textAnchor="end">{points[points.length - 1].date.slice(5)}</text>}
      </>}
    </svg>
  );
}
