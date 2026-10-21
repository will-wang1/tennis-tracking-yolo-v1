// Coaches compared with the benchmark and with each other.

import type { SessionRow, SessionType } from "./db";
import type { Benchmark } from "./benchmarks";
import { METRICS, METRIC, judge, mean } from "./metrics";

export const JUDGED = METRICS.filter((m) => m.better);

/** Share of a set of sessions' judged measures that met the benchmark
 * (within it or better). Null when nothing could be judged. */
export function onBenchmark(sessions: SessionRow[], bench: Benchmark): { share: number | null; met: number; judged: number } {
  let met = 0, judged = 0;
  for (const s of sessions) {
    for (const def of JUDGED) {
      const v = judge(def, s.metrics[def.key] ?? null, bench.bands[def.key]);
      if (v === "none") continue;
      judged++;
      if (v !== "short") met++;
    }
  }
  return { share: judged ? met / judged : null, met, judged };
}

export function averages(sessions: SessionRow[]): Record<string, number | null> {
  return Object.fromEntries(METRICS.map((d) => [d.key, mean(sessions.map((s) => s.metrics[d.key] ?? null))]));
}

export interface CoachLine {
  coachId: number;
  name: string;
  isDemo: boolean;
  sessions: SessionRow[];
  avg: Record<string, number | null>;
  score: ReturnType<typeof onBenchmark>;
  recentScore: ReturnType<typeof onBenchmark>; // last 4 sessions
  lastDate: string | null;
}

export function coachLines(all: SessionRow[], type: SessionType, bench: Benchmark): CoachLine[] {
  const byCoach = new Map<number, SessionRow[]>();
  for (const s of all) if (s.session_type === type) byCoach.set(s.coach_id, [...(byCoach.get(s.coach_id) ?? []), s]);
  return [...byCoach.values()].map((ss) => ({
    coachId: ss[0].coach_id,
    name: ss[0].coach_name,
    isDemo: ss.every((s) => s.is_demo),
    sessions: ss,
    avg: averages(ss),
    score: onBenchmark(ss, bench),
    recentScore: onBenchmark(ss.slice(-4), bench),
    lastDate: ss.at(-1)?.date ?? null,
  })).sort((a, b) => (b.score.share ?? -1) - (a.score.share ?? -1));
}

export interface Flag { coach: string; coachId: number; recent: number; items: string[] }

/** Per coach, the measures where their recent average (last 4 sessions) is
 * short of the benchmark - what a head coach should look at first. */
export function flags(lines: CoachLine[], bench: Benchmark): Flag[] {
  const out: Flag[] = [];
  for (const line of lines) {
    const recent = line.sessions.slice(-4);
    const avg = averages(recent);
    const items: string[] = [];
    for (const def of JUDGED) {
      const band = bench.bands[def.key];
      const v = avg[def.key];
      if (v == null || !band || judge(def, v, band) !== "short") continue;
      items.push(`${def.label.toLowerCase()} ${def.format(v)} (benchmark ${def.format(band.low)}–${def.format(band.high)})`);
    }
    if (items.length) out.push({ coach: line.name, coachId: line.coachId, recent: recent.length, items });
  }
  return out.sort((a, b) => b.items.length - a.items.length);
}

/** The report's own findings that keep coming back across a coach's
 * sessions, with player numbers taken out ("Player 5 waited..." and
 * "Player 2 waited..." are the same finding). */
export function recurringFindings(reports: { summary?: { went_well?: { title: string }[]; to_improve?: { title: string }[] } }[]) {
  const norm = (t: string) => t.replace(/\bPlayer \d+\b/g, "A player").replace(/\s+/g, " ").trim();
  const count = (key: "went_well" | "to_improve") => {
    const m = new Map<string, number>();
    for (const r of reports) {
      const seen = new Set((r.summary?.[key] ?? []).map((f) => norm(f.title)));
      for (const t of seen) m.set(t, (m.get(t) ?? 0) + 1);
    }
    return [...m.entries()].filter(([, n]) => n >= 2).sort((a, b) => b[1] - a[1]).map(([title, n]) => ({ title, n }));
  };
  return { strengths: count("went_well"), issues: count("to_improve") };
}

export function typeLabel(t: SessionType) {
  return t === "squad" ? "Squad sessions" : "Private lessons";
}

export { METRIC };
