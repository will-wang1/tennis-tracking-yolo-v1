import { listSessions } from "@/lib/db";
import { allBenchmarks } from "@/lib/benchmarks";
import { SessionTable } from "@/components/SessionTable";
import { DemoBanner } from "@/components/DemoBanner";

export const dynamic = "force-dynamic";

export default function SessionsPage() {
  const sessions = listSessions().reverse();
  return (
    <>
      <DemoBanner />
      <div className="head">
        <div>
          <p className="eyebrow">All coaches</p>
          <h1>Sessions</h1>
          <p className="lede">Every analysed session, newest first. Each measure is judged against the benchmark for its session type.</p>
        </div>
      </div>
      {sessions.length ? <SessionTable sessions={sessions} benches={allBenchmarks()} showToggle /> : (
        <div className="panel muted">No sessions yet. Import one with <span className="mono">scripts/publish_session.py</span>.</div>
      )}
    </>
  );
}
