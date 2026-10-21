// One measure against its benchmark: the band (shaded), the club average
// (thin line) and this value (the dot, coloured by the verdict).

import { judge, VERDICT_LABEL, type Band, type MetricDef } from "@/lib/metrics";

function domainMax(def: MetricDef, xs: (number | null | undefined)[]): number {
  if (def.format(0.5).endsWith("%")) return 1;
  const top = Math.max(0, ...xs.filter((x): x is number => x != null && Number.isFinite(x)));
  return top > 0 ? top * 1.15 : 1;
}

export function Bullet({ def, value, band, club }: { def: MetricDef; value: number | null; band: Band | null; club?: number | null }) {
  const max = domainMax(def, [value, band?.high, band?.low, club]);
  const at = (x: number) => `${Math.max(0, Math.min(100, (100 * x) / max))}%`;
  const verdict = judge(def, value, band);
  return (
    <div className="bullet" role="img"
      aria-label={`${def.label}: ${value == null ? "not measured" : def.format(value)}${band ? `, benchmark ${def.format(band.low)} to ${def.format(band.high)}` : ""}${club != null ? `, club ${def.format(club)}` : ""}`}>
      <div className="track" />
      {band && <div className="band" style={{ left: at(band.low), width: `calc(${at(band.high)} - ${at(band.low)})` }} title={`Benchmark ${def.format(band.low)}–${def.format(band.high)}`} />}
      {club != null && <div className="club" style={{ left: at(club) }} title={`Club average ${def.format(club)}`} />}
      {value != null && <div className={`me ${verdict}`} style={{ left: at(value) }} title={`${def.format(value)}${VERDICT_LABEL[verdict] ? " - " + VERDICT_LABEL[verdict] : ""}`} />}
    </div>
  );
}

export function MetricCard({ def, value, band, club, note }: { def: MetricDef; value: number | null; band: Band | null; club?: number | null; note?: string }) {
  const verdict = judge(def, value, band);
  return (
    <div className="mcard">
      <div className="top">
        <span className="label" title={def.help}>{def.label}</span>
        <span className="big">{value == null ? "–" : def.format(value)}</span>
      </div>
      <Bullet def={def} value={value} band={band} club={club} />
      <div className="sub">
        {band ? <>Benchmark {def.format(band.low)}–{def.format(band.high)}{band.n === 0 ? " (default target)" : ""}</> : def.better ? "No benchmark yet" : "Style choice - not judged"}
        {club != null && <> · club {def.format(club)}</>}
        {verdict !== "none" && <> · <b className={`vt ${verdict}`}>{VERDICT_LABEL[verdict]}</b></>}
        {note && <> · {note}</>}
      </div>
    </div>
  );
}

export function Value({ def, value, band }: { def: MetricDef; value: number | null; band: Band | null }) {
  const verdict = judge(def, value, band);
  return (
    <span className={`v ${verdict}`} title={VERDICT_LABEL[verdict] || undefined}>
      <i />{value == null ? "–" : def.format(value)}
    </span>
  );
}
