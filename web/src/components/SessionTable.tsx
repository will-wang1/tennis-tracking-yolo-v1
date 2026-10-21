import Link from "next/link";
import type { SessionRow } from "@/lib/db";
import type { Benchmark } from "@/lib/benchmarks";
import { JUDGED } from "@/lib/compare";
import { Value } from "./Bullet";
import { ExemplaryToggle } from "./ExemplaryToggle";

export function SessionTable({ sessions, benches, showCoach = true, showToggle = false, showType = true }: {
  sessions: SessionRow[];
  benches: Record<string, Benchmark>;
  showCoach?: boolean;
  showToggle?: boolean;
  showType?: boolean;
}) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Session</th>
            {showCoach && <th>Coach</th>}
            {showType && <th>Type</th>}
            {JUDGED.map((d) => <th key={d.key} title={d.help}>{d.short}</th>)}
            {showToggle && <th>Benchmark</th>}
          </tr>
        </thead>
        <tbody>
          {sessions.map((s) => (
            <tr key={s.id}>
              <td className="n">{s.date}</td>
              <td className="title">
                <Link href={`/sessions/${s.id}`}>{s.title}</Link>{" "}
                {s.exemplary && <span className="chip ex">exemplary</span>}{" "}
                {s.is_demo && <span className="chip demo">demo</span>}
              </td>
              {showCoach && <td className="nw">{s.coach_name}</td>}
              {showType && <td><span className="chip type">{s.session_type}</span></td>}
              {JUDGED.map((d) => <td key={d.key} className="n"><Value def={d} value={s.metrics[d.key] ?? null} band={benches[s.session_type].bands[d.key]} /></td>)}
              {showToggle && <td className="nw"><ExemplaryToggle id={s.id} exemplary={s.exemplary} compact /></td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
