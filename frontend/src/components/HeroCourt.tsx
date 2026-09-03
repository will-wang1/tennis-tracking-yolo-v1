import { useEffect, useRef, useState } from "react";

// Animated hero court - a ball tracked around a top-down court with bounce
// speed readouts. One straight flight per shot: the ball only changes
// direction where a player contacts it near a baseline, never mid-court or
// at the net.
const COURT_WAYPOINTS: { t: number; x: number; y: number; bounce?: number }[] = [
  { t: 0.0, x: 196, y: 470 },
  { t: 1.3, x: 344, y: 96, bounce: 203 },
  { t: 1.72, x: 361, y: 52 },
  { t: 2.3, x: 361, y: 52 },
  { t: 3.42, x: 150, y: 418, bounce: 187 },
  { t: 3.68, x: 135, y: 466 },
  { t: 4.3, x: 135, y: 466 },
  { t: 5.42, x: 322, y: 88, bounce: 218 },
  { t: 5.62, x: 328, y: 56 },
  { t: 6.05, x: 328, y: 56 },
  { t: 6.6, x: 196, y: 470 },
];
const CYCLE = 6.6;
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

function getBallPos(elapsed: number): { x: number; y: number } {
  const t = ((elapsed % CYCLE) + CYCLE) % CYCLE;
  for (let i = 0; i < COURT_WAYPOINTS.length - 1; i++) {
    const a = COURT_WAYPOINTS[i];
    const b = COURT_WAYPOINTS[i + 1];
    if (t >= a.t && t < b.t) {
      const f = (t - a.t) / (b.t - a.t);
      return { x: lerp(a.x, b.x, f), y: lerp(a.y, b.y, f) };
    }
  }
  return { x: COURT_WAYPOINTS[0].x, y: COURT_WAYPOINTS[0].y };
}

interface Bounce {
  x: number;
  y: number;
  speed: number;
  born: number;
}

export default function HeroCourt() {
  const [frameState, setFrameState] = useState<{
    ball: { x: number; y: number };
    trail: { x: number; y: number }[];
    bounces: Bounce[];
    tick: number;
  }>({ ball: { x: 200, y: 455 }, trail: [], bounces: [], tick: 0 });
  const start = useRef(0);
  const lastWp = useRef(-1);
  const raf = useRef(0);

  useEffect(() => {
    start.current = performance.now();
    function frame(now: number) {
      const elapsed = (now - start.current) / 1000;
      const t = ((elapsed % CYCLE) + CYCLE) % CYCLE;
      const pos = getBallPos(elapsed);
      let hit: Bounce | null = null;
      COURT_WAYPOINTS.forEach((wp, i) => {
        if (!wp.bounce) return;
        if (Math.abs(t - wp.t) < 0.06 && lastWp.current !== i) {
          lastWp.current = i;
          hit = { x: wp.x, y: wp.y, speed: wp.bounce, born: now };
        }
      });
      setFrameState((s) => ({
        ball: pos,
        trail: [...s.trail.slice(-28), pos],
        bounces: hit ? [...s.bounces.filter((b) => now - b.born < 2200), hit] : s.bounces,
        tick: now,
      }));
      raf.current = requestAnimationFrame(frame);
    }
    raf.current = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf.current);
  }, []);

  const { ball, trail, bounces, tick } = frameState;

  const cx = 40;
  const cy = 20;
  const cw = 400;
  const ch = 480;
  const netY = cy + ch / 2;
  const sOff = cw * 0.125;
  const svcD = (ch / 2) * 0.538;
  const svcNY = netY + svcD;
  const svcFY = netY - svcD;
  const midX = cx + cw / 2;

  return (
    <svg viewBox="0 0 480 520" style={{ width: "100%", height: "100%", display: "block" }}>
      <defs>
        <filter id="hglow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id="cg" cx="50%" cy="50%" r="55%">
          <stop offset="0%" stopColor="#1e5c40" />
          <stop offset="100%" stopColor="#12402c" />
        </radialGradient>
      </defs>
      <rect x={cx} y={cy} width={cw} height={ch} fill="url(#cg)" rx={3} />
      <rect x={cx} y={cy} width={cw} height={ch} fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth={1.5} rx={3} />
      <line x1={cx + sOff} y1={cy} x2={cx + sOff} y2={cy + ch} stroke="rgba(255,255,255,0.45)" strokeWidth={1} />
      <line x1={cx + cw - sOff} y1={cy} x2={cx + cw - sOff} y2={cy + ch} stroke="rgba(255,255,255,0.45)" strokeWidth={1} />
      <line x1={cx + sOff} y1={svcNY} x2={cx + cw - sOff} y2={svcNY} stroke="rgba(255,255,255,0.45)" strokeWidth={1} />
      <line x1={cx + sOff} y1={svcFY} x2={cx + cw - sOff} y2={svcFY} stroke="rgba(255,255,255,0.45)" strokeWidth={1} />
      <line x1={midX} y1={svcFY} x2={midX} y2={svcNY} stroke="rgba(255,255,255,0.45)" strokeWidth={1} />
      <rect x={cx} y={netY - 2} width={cw} height={4} fill="rgba(255,255,255,0.75)" rx={2} />
      <rect x={cx - 5} y={netY - 10} width={5} height={20} fill="white" rx={1} />
      <rect x={cx + cw} y={netY - 10} width={5} height={20} fill="white" rx={1} />
      {trail.map((pt, i) => (
        <circle key={i} cx={pt.x} cy={pt.y} r={2} fill="#00d4a4" opacity={(i / trail.length) * 0.45} />
      ))}
      {bounces
        .filter((b) => tick - b.born < 1800)
        .map((b, i) => {
          const age = (tick - b.born) / 1800;
          return (
            <g key={i}>
              <circle cx={b.x} cy={b.y} r={4 + age * 22} fill="none" stroke="#00d4a4" strokeWidth={1.5} opacity={0.7 * (1 - age)} />
              <circle cx={b.x} cy={b.y} r={2 + age * 10} fill="none" stroke="#00d4a4" strokeWidth={1} opacity={0.3 * (1 - age)} />
              {age < 0.85 && (
                <g>
                  <rect x={b.x + 12} y={b.y - 14} width={66} height={16} rx={4} fill="#080c10" fillOpacity={0.94} stroke="#00d4a4" strokeWidth={0.8} />
                  <text x={b.x + 45} y={b.y - 3} textAnchor="middle" fill="#00d4a4" fontSize={8.5} fontFamily="JetBrains Mono, monospace" fontWeight={500}>
                    {b.speed} km/h
                  </text>
                </g>
              )}
            </g>
          );
        })}
      <circle cx={ball.x} cy={ball.y} r={8} fill="rgba(0,212,164,0.18)" />
      <circle cx={ball.x} cy={ball.y} r={6} fill="#d4f07a" filter="url(#hglow)" />
      <circle cx={ball.x} cy={ball.y} r={4.5} fill="#e8f59a" />
    </svg>
  );
}
