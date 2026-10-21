import { listSessions } from "@/lib/db";
import { allBenchmarks } from "@/lib/benchmarks";
import { SessionTable } from "@/components/SessionTable";
import { DemoBanner } from "@/components/DemoBanner";
import { requireUser, isHead } from "@/lib/auth";

export const dynamic = "force-dynamic";

export default async function SessionsPage() {
  const user = await requireUser();
  const head = isHead(user);
  const sessions = (head ? listSessions() : user.coach_id ? listSessions({ coachId: user.coach_id }) : []).reverse();
  return (
    <>
      {head && <DemoBanner />}
      <div className="head">
        <div>
          <p className="eyebrow">{head ? "All coaches" : user.coach_name}</p>
          <h1>{head ? "Sessions" : "My sessions"}</h1>
          <p className="lede">Every analysed session, newest first. Each measure is judged against the benchmark for its session type.</p>
        </div>
      </div>
      {sessions.length ? <SessionTable sessions={sessions} benches={allBenchmarks()} showToggle={head} showCoach={head} /> : (
        <div className="panel muted">No sessions yet. <a href="/upload">Upload a session</a>.</div>
      )}
    </>
  );
}
