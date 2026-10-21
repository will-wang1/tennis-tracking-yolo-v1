import Link from "next/link";
import { notFound } from "next/navigation";
import { getSession, listSessions, SESSION_TYPES } from "@/lib/db";
import { benchmarkFor } from "@/lib/benchmarks";
import { onBenchmark, averages } from "@/lib/compare";
import { METRICS } from "@/lib/metrics";
import { MetricCard } from "@/components/Bullet";
import { ExemplaryToggle } from "@/components/ExemplaryToggle";
import { SessionVideo, type VideoEvent } from "@/components/SessionVideo";
import { requireUser, canSeeCoach, isHead } from "@/lib/auth";

export const dynamic = "force-dynamic";

type J = Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

function rallyEvents(report: J): VideoEvent[] {
  const rallies: J[] = report.outcomes?.rallies ?? [];
  return rallies.map((r) => {
    const how = r.outcome === "out" || r.outcome === "net"
      ? `error by Player ${r.error_by} (${r.outcome === "net" ? "into the net" : "out"}, ball ${r.ball_number})`
      : r.outcome === "not returned" ? `not returned (last shot Player ${r.last_hitter})` : "ending unclear";
    return {
      t: r.start_s,
      kind: r.outcome === "out" || r.outcome === "net" ? "error" : "other",
      label: `${r.fed ? "Fed" : "Live"} rally, ${r.shots} shot${r.shots === 1 ? "" : "s"} - ${how}`,
    };
  });
}

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const user = await requireUser();
  const s = getSession(Number((await params).id));
  if (!s || !canSeeCoach(user, s.coach_id)) notFound();
  const r = s.report as J;
  const bench = benchmarkFor(s.session_type);
  const score = onBenchmark([s], bench);
  const club = averages(listSessions({ type: s.session_type }));
  const summary = r.summary ?? {};
  const fb = s.feedback as J | null;
  const coachPlayer = summary.coach_player;
  const players: J[] = (summary.players ?? []).filter((p: J) => p.role !== "coach");

  return (
    <>
      {s.is_demo && <div className="banner demo"><b>Demo session.</b> Its numbers are generated from the Dingles session; there is no video.</div>}
      <div className="head">
        <div>
          <p className="eyebrow"><Link href={`/coaches/${s.coach_id}?type=${s.session_type}`}>{s.coach_name}</Link> · {SESSION_TYPES.find((t) => t.key === s.session_type)?.label}</p>
          <h1>{s.title}</h1>
          <div className="meta">
            <span><b>{s.date}</b></span>
            {s.court_name && <span>{s.court_name}</span>}
            {s.duration_s != null && <span>{Math.round(s.duration_s / 60) || "<1"} min analysed</span>}
            {s.themes && <span>Themes: <b>{s.themes.split(",").join(", ")}</b></span>}
            {score.share != null && <span>On benchmark <b>{Math.round(100 * score.share)}%</b> ({score.met} of {score.judged})</span>}
          </div>
        </div>
        {isHead(user) ? <ExemplaryToggle id={s.id} exemplary={s.exemplary} /> : s.exemplary ? <span className="chip ex">exemplary</span> : null}
      </div>

      {summary.headline && <p className="lede" style={{ fontSize: 17, color: "var(--ink)", marginBottom: 22 }}>{summary.headline}</p>}

      <div className="session-grid">
        <div>
          <SessionVideo src={s.video_path ? `/api/sessions/${s.id}/video` : null} events={rallyEvents(r)} />
          <p className="muted" style={{ fontSize: 13, marginTop: 8 }}>Click a rally to watch it. Errors are in orange.</p>
        </div>
        <div className="grid">
          <div className="panel">
            <h3>Went well</h3>
            <ul className="verdict-list good">
              {(summary.went_well ?? []).map((f: J, i: number) => <li key={i}><b>{f.title}</b>{f.detail}</li>)}
              {!(summary.went_well ?? []).length && <li className="muted">Nothing stood out.</li>}
            </ul>
          </div>
          <div className="panel">
            <h3>Worth a look</h3>
            <ul className="verdict-list look">
              {(summary.to_improve ?? []).map((f: J, i: number) => <li key={i}><b>{f.title}</b>{f.detail}</li>)}
              {!(summary.to_improve ?? []).length && <li className="muted">Nothing stood out.</li>}
            </ul>
          </div>
        </div>
      </div>

      <section className="section">
        <h2>Against the benchmark</h2>
        <div className="grid cards">
          {METRICS.map((d) => <MetricCard key={d.key} def={d} value={s.metrics[d.key] ?? null} band={bench.bands[d.key]} club={club[d.key]} />)}
        </div>
        <div className="legend"><span>Dot: this session</span><span>Shaded: benchmark</span><span>Line: club average</span></div>
      </section>

      {(r.themes ?? []).length > 0 && (
        <section className="section">
          <h2>Lesson themes</h2>
          <div className="grid cards">
            {(r.themes as J[]).map((t) => (
              <div key={t.theme} className="panel">
                <h3>{t.name}</h3>
                <p><b>{t.headline}</b></p>
                {(t.measures ?? []).map((m: J) => <p key={m.name} className="muted" style={{ fontSize: 13.5 }}>{m.detail}</p>)}
              </div>
            ))}
          </div>
        </section>
      )}

      {players.length > 0 && (
        <section className="section">
          <h2>Players</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Player</th><th>Summary</th></tr></thead>
              <tbody>
                {players.map((p) => <tr key={p.player}><td className="n">Player {p.player}</td><td>{p.line}</td></tr>)}
              </tbody>
            </table>
          </div>
          {coachPlayer != null && <p className="muted" style={{ fontSize: 13, marginTop: 8 }}>Player {coachPlayer} is taken to be the coach and left out.</p>}
        </section>
      )}

      {fb && (
        <section className="section">
          <h2>AI commentary</h2>
          <div className="panel">
            {fb.headline && <p><b>{fb.headline}</b></p>}
            <ul className="verdict-list">
              {(fb.observations ?? []).map((o: J, i: number) => <li key={i}><b>{o.point}</b>{o.evidence}</li>)}
            </ul>
            {(fb.questions_for_the_coach ?? []).length > 0 && <>
              <h3 style={{ marginTop: 16 }}>Questions for the coach</h3>
              <ul>{fb.questions_for_the_coach.map((q: string, i: number) => <li key={i}>{q}</li>)}</ul>
            </>}
            {fb.model && <p className="muted mono" style={{ marginTop: 10 }}>Written by {fb.model} from the measured numbers.</p>}
          </div>
        </section>
      )}
    </>
  );
}
