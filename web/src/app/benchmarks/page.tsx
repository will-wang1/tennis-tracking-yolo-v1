import Link from "next/link";
import { listSessions } from "@/lib/db";
import { allBenchmarks, benchmarkFor } from "@/lib/benchmarks";
import { typeLabel } from "@/lib/compare";
import { METRICS } from "@/lib/metrics";
import { SessionTable } from "@/components/SessionTable";
import { TypeTabs, parseType } from "@/components/TypeTabs";

export const dynamic = "force-dynamic";

export default async function BenchmarksPage({ searchParams }: { searchParams: Promise<{ type?: string }> }) {
  const type = parseType((await searchParams).type);
  const bench = benchmarkFor(type);
  const benches = allBenchmarks();
  const candidates = listSessions({ type, exemplary: false }).reverse();
  return (
    <>
      <div className="head">
        <div>
          <p className="eyebrow">{typeLabel(type)}</p>
          <h1>Benchmark</h1>
          <p className="lede">
            What a well-run {type === "squad" ? "squad session" : "private lesson"} looks like here, learned from the sessions you mark as exemplary:
            a band around their typical value, as wide as they vary but never narrower than a set tolerance per measure. Marking or unmarking a session updates it at once.
          </p>
        </div>
        <TypeTabs base="/benchmarks" current={type} />
      </div>

      {bench.sessions.length === 0 && (
        <div className="banner info">No sessions are marked exemplary yet, so the default targets are used where there is one. Mark sessions below.</div>
      )}

      <div className="table-wrap">
        <table>
          <thead><tr><th>Measure</th><th>Benchmark</th><th>Typical</th><th>Better</th><th>From</th></tr></thead>
          <tbody>
            {METRICS.map((d) => {
              const b = bench.bands[d.key];
              return (
                <tr key={d.key}>
                  <td title={d.help}>{d.label}</td>
                  <td className="n">{b ? `${d.format(b.low)} – ${d.format(b.high)}` : "–"}</td>
                  <td className="n">{b?.median != null ? d.format(b.median) : "–"}</td>
                  <td>{d.better ? d.better : <span className="muted">style choice, not judged</span>}</td>
                  <td className="muted">{b ? (b.n ? `${b.n} exemplary session${b.n === 1 ? "" : "s"}` : "default target") : "no data"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <section className="section">
        <h2>Exemplary sessions ({bench.sessions.length})</h2>
        {bench.sessions.length ? <SessionTable sessions={bench.sessions} benches={benches} showToggle showType={false} /> : <p className="muted">None yet.</p>}
      </section>

      <section className="section">
        <h2>Other {typeLabel(type).toLowerCase()}</h2>
        <p className="muted">Mark the ones that show how sessions should be run. Watch them first - <Link href="/sessions">all sessions</Link>.</p>
        {candidates.length ? <SessionTable sessions={candidates} benches={benches} showToggle showType={false} /> : <p className="muted">None.</p>}
      </section>
    </>
  );
}
