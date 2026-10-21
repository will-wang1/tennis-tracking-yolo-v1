import Link from "next/link";
import { notFound } from "next/navigation";
import { getCoach, getSession, listSessions } from "@/lib/db";
import { allBenchmarks } from "@/lib/benchmarks";
import { averages, onBenchmark, recurringFindings, JUDGED, typeLabel } from "@/lib/compare";
import { METRICS } from "@/lib/metrics";
import { MetricCard } from "@/components/Bullet";
import { Trend } from "@/components/Trend";
import { SessionTable } from "@/components/SessionTable";
import { TypeTabs, parseType } from "@/components/TypeTabs";

export const dynamic = "force-dynamic";

export default async function CoachPage({ params, searchParams }: { params: Promise<{ id: string }>; searchParams: Promise<{ type?: string }> }) {
  const coach = getCoach(Number((await params).id));
  if (!coach) notFound();
  const mine = listSessions({ coachId: coach.id });
  const sp = (await searchParams).type;
  // Default to the type this coach runs most.
  const type = sp ? parseType(sp) : (mine.filter((s) => s.session_type === "private").length > mine.length / 2 ? "private" : "squad");
  const benches = allBenchmarks();
  const bench = benches[type];
  const sessions = mine.filter((s) => s.session_type === type);
  const avg = averages(sessions);
  const club = averages(listSessions({ type }));
  const score = onBenchmark(sessions, bench);
  const found = recurringFindings(sessions.map((s) => getSession(s.id)!.report as never));
  const counts = { squad: mine.filter((s) => s.session_type === "squad").length, private: mine.filter((s) => s.session_type === "private").length };

  return (
    <>
      <div className="head">
        <div>
          <p className="eyebrow"><Link href={`/coaches?type=${type}`}>Coaches</Link> · {typeLabel(type)}</p>
          <h1>{coach.name} {coach.is_demo ? <span className="chip demo">demo</span> : null}</h1>
          <div className="meta">
            <span><b>{sessions.length}</b> {typeLabel(type).toLowerCase()}</span>
            {sessions.length > 0 && <span>{sessions[0].date} → {sessions.at(-1)!.date}</span>}
            {score.share != null && <span>On benchmark <b>{Math.round(100 * score.share)}%</b> ({score.met} of {score.judged} measures)</span>}
          </div>
        </div>
        <TypeTabs base={`/coaches/${coach.id}`} current={type} counts={counts} />
      </div>

      {sessions.length === 0 ? (
        <div className="panel muted">{coach.name} has no {typeLabel(type).toLowerCase()} yet.</div>
      ) : (
        <>
          <section>
            <h2>Against the benchmark</h2>
            <div className="grid cards">
              {METRICS.map((d) => <MetricCard key={d.key} def={d} value={avg[d.key]} band={bench.bands[d.key]} club={club[d.key]} />)}
            </div>
            <div className="legend">
              <span>Dot: {coach.name}&apos;s average</span><span>Shaded: benchmark</span><span>Line: club average</span>
            </div>
          </section>

          <section className="section">
            <h2>Session by session</h2>
            <div className="grid cards">
              {JUDGED.map((d) => (
                <div key={d.key} className="mcard">
                  <div className="label">{d.label}</div>
                  <Trend def={d} band={bench.bands[d.key]}
                    points={sessions.map((s) => ({ date: s.date, value: s.metrics[d.key] ?? null, title: `${s.date} ${s.title}`, href: `/sessions/${s.id}` }))} />
                </div>
              ))}
            </div>
          </section>

          <section className="section two">
            <div className="panel">
              <h3>Keeps going well</h3>
              {found.strengths.length ? (
                <ul className="findings">{found.strengths.map((f) => <li key={f.title}><span>{f.title}</span><span className="count">{f.n} of {sessions.length} sessions</span></li>)}</ul>
              ) : <p className="muted">Nothing the reports call out in more than one session.</p>}
            </div>
            <div className="panel">
              <h3>Keeps coming up</h3>
              {found.issues.length ? (
                <ul className="findings">{found.issues.map((f) => <li key={f.title}><span>{f.title}</span><span className="count">{f.n} of {sessions.length} sessions</span></li>)}</ul>
              ) : <p className="muted">Nothing the reports call out in more than one session.</p>}
            </div>
          </section>

          <section className="section">
            <h2>Sessions</h2>
            <SessionTable sessions={[...sessions].reverse()} benches={benches} showCoach={false} showType={false} />
          </section>
        </>
      )}
    </>
  );
}
