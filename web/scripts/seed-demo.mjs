// Demo data: made-up coaches and sessions, generated from the one real
// session's report (Dingles), so the head-coach views have something to
// compare while real sessions are collected. Every session is imported
// through the app's own import endpoint with is_demo set, and
// `npm run demo:clear` removes them all.
//
//   npm run dev            (in one terminal)
//   npm run demo:seed      (in another)

import fs from "node:fs";
import path from "node:path";

const URL = process.env.APP_URL ?? "http://localhost:3000";
const TOKEN = process.env.IMPORT_TOKEN ?? "dev-import-token";
const SOURCE = path.resolve(import.meta.dirname, "../../outputs/dingles_serve_volley/session_report.json");

// Deterministic, so the demo looks the same every time it is seeded.
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) - 1) / 2147483646;
const noise = (sd) => (rand() + rand() + rand() - 1.5) * sd * 1.4;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Each demo coach's typical session; `to` makes a coach change over the weeks.
const COACHES = [
  { name: "Coach A", squad: { inPlay: 0.82, waiting: 0.04, moving: 0.69, bpm: 7.6, rally: 2.5, early: 6, deep: 0.56, live: 0.72, net: 0.16 },
    priv: { inPlay: 0.8, moving: 0.72, bpm: 11, rally: 3.1, early: 5, deep: 0.6, live: 0.55, net: 0.12 }, exemplary: [1, 3, 5] },
  { name: "Coach B", squad: { inPlay: 0.72, waiting: 0.17, moving: 0.6, bpm: 4.9, rally: 2.0, early: 10, deep: 0.47, live: 0.66, net: 0.1 } },
  { name: "Coach C", squad: { inPlay: 0.63, waiting: 0.08, moving: 0.63, bpm: 6.1, rally: 1.5, early: 8, deep: 0.5, live: 0.28, net: 0.2 },
    priv: { inPlay: 0.66, moving: 0.66, bpm: 9.5, rally: 2.2, early: 9, deep: 0.5, live: 0.2, net: 0.08 } },
  { name: "Coach D", squad: { inPlay: 0.66, waiting: 0.15, moving: 0.58, bpm: 5.0, rally: 1.7, early: 12, deep: 0.44, live: 0.6, net: 0.12 },
    to: { inPlay: 0.8, waiting: 0.06, moving: 0.67, bpm: 7.0, rally: 2.3, early: 7, deep: 0.53 } },
];

const SQUAD_DATES = ["2026-08-25", "2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22", "2026-09-29", "2026-10-06", "2026-10-13"];
const PRIVATE_DATES = ["2026-08-28", "2026-09-11", "2026-09-25", "2026-10-09"];

function findings(v, isSquad) {
  const went = [], look = [];
  if (v.inPlay >= 0.75) went.push({ title: "Very little dead time", detail: `A ball was in play ${Math.round(100 * v.inPlay)}% of the time.` });
  if (v.inPlay < 0.68) look.push({ title: "A lot of dead time", detail: `A ball was in play only ${Math.round(100 * v.inPlay)}% of the time.` });
  if (v.moving >= 0.65) went.push({ title: "Players kept moving", detail: `The typical player was on the move ${Math.round(100 * v.moving)}% of the time.` });
  if (isSquad && v.waiting >= 0.12) look.push({ title: `Player ${2 + Math.floor(rand() * 5)} waited much longer than the others`, detail: `The typical player spent ${Math.round(100 * v.waiting)}% of the drill waiting off court.` });
  if (v.early >= 10) look.push({ title: "Many errors on the first two balls", detail: `${v.early.toFixed(0)} errors on the 1st-2nd ball per 100 shots.` });
  if (v.rally >= 2.3) went.push({ title: "Rallies held up", detail: `Rallies averaged ${v.rally.toFixed(1)} shots.` });
  if (v.live < 0.35) look.push({ title: "Mostly fed balls", detail: `Only ${Math.round(100 * v.live)}% of play time was live rallies.` });
  return { went, look };
}

function makeReport(base, v, isSquad) {
  const r = structuredClone(base);
  const coach = r.summary.coach_player;
  const set = (k, x) => { if (r.metrics[k]) r.metrics[k].value = x; else r.metrics[k] = { value: x }; };
  set("ball_in_play", v.inPlay); set("shots_per_rally", v.rally); set("deep_share", v.deep);
  r.feeding.play_time.shares = { fed: 1 - v.live, live: v.live, mixed: 0 };
  let players = r.players.filter((p) => p.player !== coach);
  if (!isSquad) players = players.slice(0, 1);
  for (const p of players) {
    p.queue_share = isSquad ? clamp(v.waiting + noise(0.02), 0, 0.5) : 0;
    p.moving_share = clamp(v.moving + noise(0.04), 0.2, 0.95);
    p.shots_per_minute = Math.max(0.5, v.bpm + noise(0.8));
    p.zone_shares = { ...p.zone_shares, net: clamp(v.net + noise(0.04), 0, 0.8) };
  }
  r.players = [...r.players.filter((p) => p.player === coach), ...players];
  r.outcomes.per_player = { all: { shots: 200, errors_first_two_balls: Math.round(v.early * 2) } };
  r.outcomes.rallies = [];
  r.themes = [];
  const f = findings(v, isSquad);
  r.summary = {
    ...r.summary,
    headline: `Demo session: ${Math.round(100 * v.inPlay)}% ball in play, ${v.bpm.toFixed(1)} balls a minute for the typical player, rallies of ${v.rally.toFixed(1)} shots.`,
    went_well: f.went, to_improve: f.look, notes: [], players: [],
  };
  return r;
}

function vary(p, t) {
  const v = {};
  for (const k of Object.keys(p)) {
    const target = p.to?.[k] ?? p[k];
    v[k] = typeof p[k] === "number" ? p[k] + (target - p[k]) * t : p[k];
  }
  return {
    inPlay: clamp(v.inPlay + noise(0.04), 0.4, 0.95),
    waiting: clamp((v.waiting ?? 0) + noise(0.025), 0, 0.4),
    moving: clamp(v.moving + noise(0.03), 0.3, 0.9),
    bpm: Math.max(1, v.bpm + noise(0.6)),
    rally: Math.max(1, v.rally + noise(0.25)),
    early: Math.max(0, v.early + noise(2)),
    deep: clamp(v.deep + noise(0.05), 0.1, 0.9),
    live: clamp(v.live + noise(0.06), 0, 1),
    net: clamp(v.net + noise(0.03), 0, 0.6),
  };
}

async function post(body) {
  const res = await fetch(`${URL}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${TOKEN}` },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`import failed ${res.status}: ${await res.text()}`);
  return (await res.json()).id;
}

const base = JSON.parse(fs.readFileSync(SOURCE, "utf8"));
let n = 0;
for (const c of COACHES) {
  for (const [i, date] of SQUAD_DATES.entries()) {
    const v = vary({ ...c.squad, to: c.to }, i / (SQUAD_DATES.length - 1));
    await post({
      title: i % 2 ? "Squad - consistency" : "Squad - net play", coach: c.name, date, session_type: "squad", court: `Court ${1 + (i % 3)}`,
      themes: i % 2 ? ["consistency"] : ["netplay"], report: makeReport(base, v, true),
      exemplary: (c.exemplary ?? []).includes(i), is_demo: true,
    });
    n++;
  }
  if (!c.priv) continue;
  for (const [i, date] of PRIVATE_DATES.entries()) {
    const v = vary(c.priv, 0);
    await post({
      title: "Private lesson", coach: c.name, date, session_type: "private", court: "Court 4",
      report: makeReport(base, v, false), exemplary: c.name === "Coach A" && i < 2, is_demo: true,
    });
    n++;
  }
}
console.log(`Seeded ${n} demo sessions for ${COACHES.length} demo coaches at ${URL}`);
