// Benchmarks: per session type, learned from the sessions the head coach
// marked exemplary. Computed from the current marks every time, so marking
// or unmarking a session moves the benchmark at once.

import { listSessions, type SessionType, type SessionRow } from "./db";
import { METRICS, bandFor, type Band } from "./metrics";

export interface Benchmark {
  type: SessionType;
  sessions: SessionRow[]; // the exemplary sessions it is learned from
  bands: Record<string, Band | null>;
}

export function benchmarkFor(type: SessionType): Benchmark {
  const sessions = listSessions({ type, exemplary: true });
  const bands: Record<string, Band | null> = {};
  for (const def of METRICS) bands[def.key] = bandFor(def, sessions.map((s) => s.metrics[def.key] ?? null));
  return { type, sessions, bands };
}

export function allBenchmarks(): Record<SessionType, Benchmark> {
  return { squad: benchmarkFor("squad"), private: benchmarkFor("private") };
}
