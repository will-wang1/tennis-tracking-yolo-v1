import Link from "next/link";
import { listSessions } from "@/lib/db";
import { benchmarkFor } from "@/lib/benchmarks";
import { coachLines, flags, averages, JUDGED, typeLabel } from "@/lib/compare";
import { Value } from "@/components/Bullet";
import { TypeTabs, parseType } from "@/components/TypeTabs";
import { DemoBanner } from "@/components/DemoBanner";

export const dynamic = "force-dynamic";

function pct(x: number | null) {
  return x == null ? "–" : `${Math.round(100 * x)}%`;
}

export default async function CoachesPage({ searchParams }: { searchParams: Promise<{ type?: string }> }) {
  const type = parseType((await searchParams).type);
  const all = listSessions();
  const bench = benchmarkFor(type);
  const lines = coachLines(all, type, bench);
  const club = averages(all.filter((s) => s.session_type === type));
  const flagged = flags(lines, bench);
  const counts = { squad: all.filter((s) => s.session_type === "squad").length, private: all.filter((s) => s.session_type === "private").length };

  return (
    <>
      <DemoBanner />
      <div className="head">
        <div>
          <p className="eyebrow">Head coach · {typeLabel(type)}</p>
          <h1>Coaches</h1>
          <p className="lede">
            How each coach&apos;s sessions measure up against the benchmark
            {bench.sessions.length ? <> learned from {bench.sessions.length} exemplary session{bench.sessions.length === 1 ? "" : "s"}</> : <> (default targets until sessions are marked exemplary)</>}.
          </p>
        </div>
        <TypeTabs base="/coaches" current={type} counts={counts} />
      </div>

      {lines.length === 0 ? (
        <div className="panel muted">No {typeLabel(type).toLowerCase()} yet.</div>
      ) : (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Coach</th>
                  <th title="Share of the judged measures, across all their sessions, that met the benchmark (within it or better)">On benchmark</th>
                  <th title="The same, over their last 4 sessions">Recent</th>
                  <th>Sessions</th>
                  {JUDGED.map((d) => <th key={d.key} title={d.help}>{d.short}</th>)}
                </tr>
              </thead>
              <tbody>
                {lines.map((l) => (
                  <tr key={l.coachId}>
                    <td>
                      <Link href={`/coaches/${l.coachId}?type=${type}`}><b>{l.name}</b></Link>{" "}
                      {l.isDemo && <span className="chip demo">demo</span>}
                    </td>
                    <td className="n score">{pct(l.score.share)}</td>
                    <td className="n score">
                      {pct(l.recentScore.share)}
                      {l.recentScore.share != null && l.score.share != null && Math.abs(l.recentScore.share - l.score.share) >= 0.05 &&
                        <span className="muted"> {l.recentScore.share > l.score.share ? "↑" : "↓"}</span>}
                    </td>
                    <td className="n">{l.sessions.length}</td>
                    {JUDGED.map((d) => <td key={d.key} className="n"><Value def={d} value={l.avg[d.key]} band={bench.bands[d.key]} /></td>)}
                  </tr>
                ))}
                <tr>
                  <td className="muted">Club average</td>
                  <td /><td /><td className="n muted">{lines.reduce((a, l) => a + l.sessions.length, 0)}</td>
                  {JUDGED.map((d) => <td key={d.key} className="n muted">{club[d.key] == null ? "–" : d.format(club[d.key]!)}</td>)}
                </tr>
              </tbody>
            </table>
          </div>
          <div className="legend">
            <span className="v good"><i />better than benchmark</span>
            <span className="v within"><i />within benchmark</span>
            <span className="v short"><i />short of benchmark</span>
            <span className="v none"><i />no benchmark</span>
            <span>Averages over all the coach&apos;s {typeLabel(type).toLowerCase()}.</span>
          </div>

          <section className="section">
            <h2>Worth a look</h2>
            {flagged.length === 0 ? (
              <p className="muted">Every coach&apos;s recent sessions are within the benchmark or better.</p>
            ) : (
              <ul className="flags">
                {flagged.map((f, i) => (
                  <li key={i}>
                    <Link href={`/coaches/${f.coachId}?type=${type}`}><b>{f.coach}</b></Link> - short of the benchmark over
                    the last {f.recent === 1 ? "session" : `${f.recent} sessions`} on {f.items.join("; ")}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </>
      )}
    </>
  );
}
