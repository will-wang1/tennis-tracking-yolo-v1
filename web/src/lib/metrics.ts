// The measures a head coach compares across coaches and sessions, how each
// is read from a session report, and how a value is judged against a
// benchmark.

export type Better = "higher" | "lower" | null; // null: a style choice, shown but not judged

export interface MetricDef {
  key: string;
  label: string;
  short: string;
  better: Better;
  format: (v: number) => string;
  help: string;
  // Fallback band until the benchmark has exemplary sessions - only where the
  // backend already has a stated target (src/session/insights.py Targets).
  fallback?: [number, number];
  // Smallest half-width a learned band may have: a few exemplary sessions
  // agreeing closely is chance, not a standard to hold every session to.
  tol: number;
}

const pct = (v: number) => `${Math.round(100 * v)}%`;
const one = (v: number) => v.toFixed(1);

export const METRICS: MetricDef[] = [
  { key: "ball_in_play", tol: 0.05, label: "Ball in play", short: "In play", better: "higher", format: pct, fallback: [0.7, 1],
    help: "Share of analysed time a ball was in play." },
  { key: "waiting", tol: 0.03, label: "Time waiting for a turn", short: "Waiting", better: "lower", format: pct, fallback: [0, 0.1],
    help: "Typical player's share of drill time standing off court waiting." },
  { key: "moving", tol: 0.05, label: "Time moving", short: "Moving", better: "higher", format: pct, fallback: [0.6, 1],
    help: "Typical player's share of time on court spent moving." },
  { key: "balls_per_min", tol: 0.8, label: "Balls hit per player per minute", short: "Balls/min", better: "higher", format: one,
    help: "Typical player's detected swings per minute on camera in drills." },
  { key: "shots_per_rally", tol: 0.3, label: "Shots per rally", short: "Rally", better: "higher", format: one,
    help: "Average rally length in shots." },
  { key: "early_errors", tol: 2, label: "Errors on the 1st-2nd ball, per 100 shots", short: "Early errors", better: "lower", format: one,
    help: "Errors (out or into the net) on the first two balls of a rally, per 100 shots hit." },
  { key: "deep_share", tol: 0.06, label: "Shots landing deep", short: "Deep", better: "higher", format: pct,
    help: "Share of in-court bounces landing past the service line." },
  { key: "live_share", tol: 0.1, label: "Live play (vs feeding)", short: "Live", better: null, format: pct,
    help: "Share of play time in rallies players started (the rest fed by the coach). A style choice." },
  { key: "net_time", tol: 0.04, label: "Time at the net", short: "Net", better: null, format: pct,
    help: "Typical player's share of drill time inside the service line." },
];

export const METRIC = Object.fromEntries(METRICS.map((m) => [m.key, m])) as Record<string, MetricDef>;

// ---- extraction from a backend report -----------------------------------

type J = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

function median(xs: number[]): number | null {
  const v = xs.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const m = Math.floor(v.length / 2);
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

export function extractMetrics(report: J): Record<string, number | null> {
  const coach = report?.summary?.coach_player ?? null;
  const players: J[] = (report.players ?? []).filter((p: J) => p.player !== coach);
  const typical = (f: (p: J) => number | null | undefined) =>
    median(players.map(f).filter((x): x is number => x != null));
  const m = report.metrics ?? {};
  const val = (k: string) => (m[k]?.value ?? null) as number | null;

  let early: number | null = null;
  const pp = report.outcomes?.per_player as Record<string, J> | undefined;
  if (pp) {
    const shots = Object.values(pp).reduce((a, v) => a + (v.shots ?? 0), 0);
    const errs = Object.values(pp).reduce((a, v) => a + (v.errors_first_two_balls ?? 0), 0);
    early = shots ? (100 * errs) / shots : null;
  }
  return {
    ball_in_play: val("ball_in_play"),
    waiting: typical((p) => p.queue_share),
    moving: typical((p) => p.moving_share),
    balls_per_min: typical((p) => p.shots_per_minute),
    shots_per_rally: val("shots_per_rally"),
    early_errors: early,
    deep_share: val("deep_share"),
    live_share: report.feeding?.play_time?.shares?.live ?? null,
    net_time: typical((p) => p.zone_shares?.net),
  };
}

// ---- benchmarks ----------------------------------------------------------

export interface Band {
  low: number;
  high: number;
  median: number | null;
  n: number; // exemplary sessions it was learned from; 0 = the fallback target
}

function quantile(sorted: number[], q: number): number {
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i), hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

/** The benchmark band for one measure, from exemplary sessions' values:
 * centred on their median, half as wide as their middle half (their range,
 * with fewer than three) but never narrower than the measure's tolerance.
 * With no exemplary sessions, the stated default target. */
export function bandFor(def: MetricDef, exemplaryValues: (number | null)[]): Band | null {
  const v = exemplaryValues.filter((x): x is number => x != null && Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return def.fallback ? { low: def.fallback[0], high: def.fallback[1], median: null, n: 0 } : null;
  const median = quantile(v, 0.5);
  const spread = v.length >= 3 ? (quantile(v, 0.75) - quantile(v, 0.25)) / 2 : (v[v.length - 1] - v[0]) / 2;
  const half = Math.max(def.tol, spread);
  const isShare = def.format(0.5).endsWith("%");
  return { low: Math.max(0, median - half), high: isShare ? Math.min(1, median + half) : median + half, median, n: v.length };
}

export type Verdict = "good" | "within" | "short" | "none";

/** Against the band: "good" beyond it the better way, "within", "short" the
 * worse way; "none" when there is no band, no value, or the measure is a
 * style choice. */
export function judge(def: MetricDef, value: number | null, band: Band | null): Verdict {
  if (value == null || !band || !def.better) return "none";
  // Judge at the precision shown, so "79% against 79-89%" is never "short".
  const r = (x: number) => Number(def.format(x).replace("%", ""));
  if (r(value) >= r(band.low) && r(value) <= r(band.high)) return "within";
  const above = r(value) > r(band.high);
  return (above && def.better === "higher") || (!above && def.better === "lower") ? "good" : "short";
}

export const VERDICT_LABEL: Record<Verdict, string> = {
  good: "better than benchmark",
  within: "within benchmark",
  short: "short of benchmark",
  none: "",
};

export function mean(xs: (number | null)[]): number | null {
  const v = xs.filter((x): x is number => x != null && Number.isFinite(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}
