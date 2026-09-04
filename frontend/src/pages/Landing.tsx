import { useEffect, useRef, useState, type RefObject } from "react";
import { Link, useNavigate } from "react-router-dom";
import Logo from "../components/Logo";
import Chip from "../components/Chip";
import HeroCourt from "../components/HeroCourt";
import StatTile from "../components/StatTile";
import ShotSpeedChart from "../components/ShotSpeedChart";
import ShotStatsTable from "../components/ShotStatsTable";
import BounceHeatmap from "../components/BounceHeatmap";
import type { ShotSpeedStat } from "../api/types";

// ── Sample data (matches the reference design's fixture data exactly) ──

const SHOTS_RAW = [
  { frames: "001–018", speed: 187, method: "Optical flow" },
  { frames: "045–063", speed: 142, method: "Keypoint" },
  { frames: "098–112", speed: 203, method: "Optical flow" },
  { frames: "156–171", speed: 168, method: "Keypoint" },
  { frames: "212–229", speed: 195, method: "Optical flow" },
  { frames: "278–294", speed: 134, method: "Keypoint" },
  { frames: "331–347", speed: 178, method: "Optical flow" },
  { frames: "401–418", speed: 221, method: "Optical flow" },
];

const SAMPLE_SHOTS: ShotSpeedStat[] = SHOTS_RAW.map((s) => {
  const [start, end] = s.frames.split("–").map((n) => parseInt(n, 10));
  return { start_frame: start, end_frame: end, peak_speed: s.speed, unit: "km/h", method: s.method };
});

const BOUNCES_PCT: [number, number][] = [
  [25, 87], [45, 91], [68, 85], [38, 89], [72, 93], [15, 82], [58, 88], [82, 86],
  [30, 8], [55, 12], [72, 7], [42, 15], [20, 10],
  [35, 68], [62, 71], [50, 65], [28, 74],
  [40, 32], [65, 28], [25, 35], [55, 38],
  [45, 52], [38, 48], [60, 55],
  [8, 70], [92, 25], [50, 78], [33, 20],
];
const COURT_WIDTH_M = 10.97;
const COURT_LENGTH_M = 23.77;
const SAMPLE_BOUNCES: [number, number][] = BOUNCES_PCT.map(([x, y]) => [
  (x / 100) * COURT_WIDTH_M,
  (y / 100) * COURT_LENGTH_M,
]);

// ── Scroll-driven motion (ports the reference design's pinned/reveal math) ──

function clamp(v: number, a: number, b: number) {
  return v < a ? a : v > b ? b : v;
}
function ease(t: number) {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

function useLandingMotion() {
  const passRef = useRef<HTMLDivElement>(null);
  const panelsRef = useRef<HTMLDivElement>(null);
  const statsRef = useRef<HTMLDivElement>(null);
  const [vals, setVals] = useState({ pass: 0, panels: 0, statsIn: 0, navAlpha: 0 });
  const lastKey = useRef<number | null>(null);
  const raf = useRef(0);

  useEffect(() => {
    const pinned = (ref: RefObject<HTMLDivElement | null>) => {
      const el = ref.current;
      if (!el) return 0;
      const vh = window.innerHeight || 800;
      const r = el.getBoundingClientRect();
      const total = r.height - vh;
      return total <= 0 ? 0 : clamp(-r.top / total, 0, 1);
    };
    const reveal = (ref: RefObject<HTMLDivElement | null>) => {
      const el = ref.current;
      if (!el) return 0;
      const vh = window.innerHeight || 800;
      const r = el.getBoundingClientRect();
      return clamp((vh - r.top) / (vh * 0.65), 0, 1);
    };
    const measure = () => {
      setVals({
        pass: pinned(passRef),
        panels: pinned(panelsRef),
        statsIn: reveal(statsRef),
        navAlpha: clamp((window.scrollY || 0) / 120, 0, 0.95),
      });
    };

    let alive = true;
    const tick = () => {
      if (!alive) return;
      const el = passRef.current;
      const key = el ? Math.round(el.getBoundingClientRect().top) : null;
      if (key !== lastKey.current) {
        lastKey.current = key;
        measure();
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    measure();
    window.addEventListener("resize", measure);
    return () => {
      alive = false;
      cancelAnimationFrame(raf.current);
      window.removeEventListener("resize", measure);
    };
  }, []);

  return { passRef, panelsRef, statsRef, ...vals };
}

function VideoFramePlaceholder({ fileName }: { fileName: string }) {
  return (
    <div
      style={{
        borderRadius: "var(--radius-md)",
        background: "var(--tva-black)",
        border: "1px solid var(--border-default)",
        padding: "40px 16px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          width: 44,
          height: 44,
          margin: "0 auto 14px",
          borderRadius: "50%",
          background: "var(--fill-accent-wash)",
          color: "var(--text-accent)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width="16" height="16" viewBox="0 0 14 14" fill="none">
          <path d="M4 2.5v9l8-4.5-8-4.5Z" fill="currentColor" />
        </svg>
      </div>
      <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text-body)" }}>{fileName}</div>
      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>1080p · H.264</div>
    </div>
  );
}

export default function Landing() {
  const navigate = useNavigate();
  const { passRef, panelsRef, statsRef, pass, panels, statsIn, navAlpha } = useLandingMotion();

  const seg = (a: number, b: number) => clamp((pass - a) / (b - a), 0, 1);
  const boxIn = seg(0.02, 0.14);
  const trail = seg(0.16, 0.42);
  const hud = seg(0.44, 0.6);
  const rot = seg(0.64, 0.96);
  const stageIndex = pass < 0.16 ? 0 : pass < 0.44 ? 1 : pass < 0.64 ? 2 : 3;
  const eyebrows = ["01 — DETECTION", "02 — TRACKING", "03 — SPEED", "04 — PROJECTION"];
  const headlines = [
    "Find the ball, frame by frame",
    "Reject the noise, fill the gaps",
    "Read the speed at the net",
    "Flatten the court to real metres",
  ];
  const copies = [
    "A YOLO checkpoint fine-tuned on ball-only footage runs every frame at low confidence — recall over precision, because a frame never flagged can't be recovered later.",
    "Implausible jumps are discarded, lock-ons onto fixed screen objects are thrown out, and short gaps are filled by interpolation. Longer gaps are left unfilled rather than guessed at.",
    "Each shot's headline reading is taken where its fitted flight crosses the net — the lowest point between two strikes, and so the least distorted by a ground-plane homography.",
    "Fourteen court keypoints lock the homography. Every landing moves from pixels into metres, and the match becomes a to-scale map.",
  ];

  const boxOpacity = boxIn * (1 - rot);
  const trailOpacity = clamp(trail * 3, 0, 1) * (1 - rot * 0.8);
  const trailOffset = 1 - trail;
  const hudOpacity = hud * (1 - rot);
  const netSpeed = Math.round(hud * 203);
  const keyOpacity = seg(0.6, 0.72) * (1 - rot * 0.9);
  const persRotate = -16 * rot;
  const persScale = 1 - 0.08 * rot;
  const persOpacity = 1 - rot;
  const topRotate = 66 * (1 - rot);
  const topScale = 0.86 + 0.14 * rot;
  const topOpacity = clamp(rot * 1.6 - 0.15, 0, 1);
  const visibleBounces = SAMPLE_BOUNCES.slice(0, Math.round(rot * SAMPLE_BOUNCES.length));
  const showHud = pass > 0.01 && pass < 0.99;
  const hudFrame = String(400 + Math.round(pass * 418)).padStart(5, "0");
  const hudTrack = (0.62 + boxIn * 0.32).toFixed(2);
  const hudSpeed = Math.round(hud * 203) || Math.round(trail * 148);
  const hudBounces = Math.round(rot * 2841).toLocaleString("en-US");
  const hudStage = ["DETECT", "TRACK", "SPEED", "PROJECT"][stageIndex];
  const railFill = Math.round(pass * 100);
  const railLabel = "PIPELINE " + Math.round(pass * 100) + "%";

  const panelAt = (i: number) => {
    const d = panels * 3 - i;
    const o = clamp((0.52 - Math.abs(d)) / 0.06, 0, 1);
    return { o, y: Math.round(-30 * clamp(d, -1, 1)) };
  };
  const pa = [panelAt(0), panelAt(1), panelAt(2), panelAt(3)];

  const st = ease(statsIn);
  const stat0 = Math.round(st * 2841).toLocaleString("en-US");
  const stat1 = Math.round(st * 221);
  const stat2 = Math.round(st * 1398).toLocaleString("en-US");
  const stat3 = (st * 4.2).toFixed(1);

  return (
    <div style={{ background: "var(--tva-black)", minHeight: "100vh" }}>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 60,
          transition: "background 300ms cubic-bezier(0,0,0.2,1), border-color 300ms cubic-bezier(0,0,0.2,1)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid transparent",
        }}
      >
        <div style={{ background: `rgba(8,12,16,${navAlpha})`, borderBottom: `1px solid rgba(24,34,48,${navAlpha})` }}>
          <div
            style={{
              maxWidth: 1152,
              margin: "0 auto",
              padding: "0 24px",
              height: 56,
              display: "flex",
              alignItems: "center",
              gap: 24,
            }}
          >
            <Link to="/" className="brand" style={{ flex: 1, display: "flex" }}>
              <Logo size={20} />
              <span className="brand-wordmark">
                Tennis<span className="accent">VA</span>
              </span>
            </Link>
            <nav style={{ display: "flex", gap: 24 }}>
              <a href="#pass" style={{ fontSize: 12, color: "var(--text-muted)", letterSpacing: "0.025em" }}>
                Pipeline
              </a>
              <a href="#capabilities" style={{ fontSize: 12, color: "var(--text-muted)", letterSpacing: "0.025em" }}>
                Capabilities
              </a>
              <a href="#results" style={{ fontSize: 12, color: "var(--text-muted)", letterSpacing: "0.025em" }}>
                Sample output
              </a>
            </nav>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginLeft: "auto" }}>
              <button className="btn-ghost" onClick={() => navigate("/login")}>
                Sign in
              </button>
              <button className="btn btn-primary" onClick={() => navigate("/register")}>
                Get started
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* ── Hero ── */}
      <section style={{ position: "relative", minHeight: "calc(100vh - 56px)", display: "flex", alignItems: "center", overflow: "hidden" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            pointerEvents: "none",
            backgroundImage: "radial-gradient(circle, rgba(0,212,164,0.07) 1px, transparent 1px)",
            backgroundSize: "36px 36px",
          }}
        />
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: 160,
            background: "linear-gradient(to top, #080c10, transparent)",
            pointerEvents: "none",
          }}
        />
        <div
          style={{
            position: "relative",
            maxWidth: 1152,
            margin: "0 auto",
            padding: "64px 24px",
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 48,
            alignItems: "center",
            width: "100%",
          }}
        >
          <div>
            <div style={{ marginBottom: 32, whiteSpace: "nowrap" }}>
              <Chip dot>Now in public beta</Chip>
            </div>
            <h1
              style={{
                margin: "0 0 24px",
                fontFamily: "var(--font-display)",
                fontWeight: 700,
                fontSize: "clamp(3rem, 6vw, 5.5rem)",
                lineHeight: 0.95,
                color: "var(--text-body)",
              }}
            >
              See every shot.
              <br />
              <span style={{ color: "var(--text-accent)" }}>Miss nothing.</span>
            </h1>
            <p style={{ margin: "0 0 32px", fontSize: 16, lineHeight: 1.65, color: "var(--text-muted)", maxWidth: 448 }}>
              Upload any tennis match video and receive an annotated output with shot speeds, bounce heatmaps, and
              rally statistics — in minutes.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 12 }}>
              <button className="btn btn-primary" style={{ padding: "12px 22px", fontSize: 15 }} onClick={() => navigate("/register")}>
                Start analysing free
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path
                    d="M3 7h8M7 3l4 4-4 4"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </button>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>No credit card · Free tier available</span>
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, marginTop: 32, whiteSpace: "nowrap" }}>
              <Chip tone="neutral">221 km/h peak speed</Chip>
              <Chip tone="neutral">&lt; 5 min analysis</Chip>
              <Chip tone="neutral">1080p export</Chip>
            </div>
          </div>
          <div style={{ position: "relative", width: "100%", maxWidth: 480, margin: "0 auto", height: "min(520px, 62vh)" }}>
            <div
              style={{
                position: "absolute",
                inset: 32,
                borderRadius: "50%",
                background: "radial-gradient(circle, rgba(0,212,164,0.08), transparent 70%)",
                pointerEvents: "none",
              }}
            />
            <HeroCourt />
          </div>
        </div>
      </section>

      {/* ── Pipeline showcase (scroll-pinned) ── */}
      <div id="pass" ref={passRef} style={{ position: "relative", height: "420vh" }}>
        <div
          style={{
            position: "sticky",
            top: 0,
            height: "100vh",
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 28,
            padding: "72px 24px 48px",
            boxSizing: "border-box",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              backgroundImage: "radial-gradient(circle, rgba(0,212,164,0.05) 1px, transparent 1px)",
              backgroundSize: "36px 36px",
            }}
          />

          <div style={{ position: "relative", textAlign: "center", flex: "none" }}>
            <div
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                letterSpacing: "0.2em",
                color: "var(--text-accent)",
                textTransform: "uppercase",
              }}
            >
              {eyebrows[stageIndex]}
            </div>
            <h2
              style={{
                margin: "12px 0 0",
                fontFamily: "var(--font-display)",
                fontWeight: 700,
                fontSize: "clamp(2rem, 4vw, 3rem)",
                lineHeight: 1.15,
                color: "var(--text-body)",
              }}
            >
              {headlines[stageIndex]}
            </h2>
            <p style={{ margin: "8px auto 0", maxWidth: 520, fontSize: 14, lineHeight: 1.65, color: "var(--text-muted)" }}>
              {copies[stageIndex]}
            </p>
          </div>

          <div
            style={{
              position: "relative",
              flex: 1,
              minHeight: 0,
              width: "min(760px, 80vw)",
              aspectRatio: "640/430",
              maxHeight: "52vh",
              perspective: 1400,
            }}
          >
            <div style={{ position: "absolute", inset: 0, transformOrigin: "50% 80%" }}>
              <div
                style={{
                  transform: `rotateX(${persRotate.toFixed(1)}deg) scale(${persScale.toFixed(3)})`,
                  opacity: persOpacity.toFixed(2),
                  transformOrigin: "50% 80%",
                }}
              >
                <div style={{ position: "relative", border: "1px solid var(--border-default)", borderRadius: 12, overflow: "hidden", background: "#172b22" }}>
                  <svg viewBox="0 0 640 430" preserveAspectRatio="xMidYMid meet" style={{ width: "100%", height: "100%", display: "block" }}>
                    <rect width="640" height="430" fill="#10201a" />
                    <polygon points="70,395 570,395 480,42 160,42" fill="#1e5c40" />
                    <polygon points="70,395 570,395 480,42 160,42" fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth="2" />
                    <line x1="148" y1="395" x2="214" y2="42" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
                    <line x1="492" y1="395" x2="426" y2="42" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
                    <line x1="70" y1="218" x2="570" y2="218" stroke="rgba(255,255,255,0.75)" strokeWidth="2.5" />
                    <line x1="136" y1="307" x2="504" y2="307" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
                    <line x1="186" y1="129" x2="454" y2="129" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />
                    <line x1="320" y1="129" x2="320" y2="307" stroke="rgba(255,255,255,0.4)" strokeWidth="1.5" />

                    <g opacity={trailOpacity.toFixed(2)}>
                      <path
                        d="M 316 386 L 344 70 L 176 372"
                        pathLength={1}
                        fill="none"
                        stroke="#00d4a4"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeDasharray="1 1"
                        strokeDashoffset={trailOffset.toFixed(3)}
                        opacity="0.85"
                      />
                      <path
                        d="M 316 386 L 344 70 L 176 372"
                        pathLength={1}
                        fill="none"
                        stroke="#00d4a4"
                        strokeWidth="7"
                        strokeLinecap="round"
                        strokeDasharray="1 1"
                        strokeDashoffset={trailOffset.toFixed(3)}
                        opacity="0.12"
                      />
                    </g>

                    <g opacity={boxOpacity.toFixed(2)}>
                      <circle cx="318" cy="384" r="5" fill="#d4f07a" />
                      <circle cx="318" cy="384" r="13" fill="none" stroke="#00d4a4" strokeWidth="1.2" opacity="0.55" />
                    </g>

                    <g opacity={hudOpacity.toFixed(2)}>
                      <line x1="70" y1="218" x2="570" y2="218" stroke="#00d4a4" strokeWidth="2.5" opacity="0.8" />
                      <circle cx="360" cy="218" r="6" fill="none" stroke="#00d4a4" strokeWidth="1.4" />
                      <text x="378" y="234" fill="rgba(220,228,237,0.5)" fontSize="8" fontFamily="JetBrains Mono, monospace" letterSpacing="0.1em">
                        NET CROSSING
                      </text>
                    </g>

                    <g opacity={keyOpacity.toFixed(2)}>
                      <circle cx="70" cy="395" r="4" fill="#00d4a4" />
                      <circle cx="570" cy="395" r="4" fill="#00d4a4" />
                      <circle cx="480" cy="42" r="4" fill="#00d4a4" />
                      <circle cx="160" cy="42" r="4" fill="#00d4a4" />
                      <circle cx="148" cy="395" r="3" fill="#00d4a4" />
                      <circle cx="492" cy="395" r="3" fill="#00d4a4" />
                      <circle cx="214" cy="42" r="3" fill="#00d4a4" />
                      <circle cx="426" cy="42" r="3" fill="#00d4a4" />
                      <circle cx="136" cy="307" r="3" fill="#00d4a4" />
                      <circle cx="504" cy="307" r="3" fill="#00d4a4" />
                      <circle cx="186" cy="129" r="3" fill="#00d4a4" />
                      <circle cx="454" cy="129" r="3" fill="#00d4a4" />
                      <circle cx="320" cy="395" r="3" fill="#00d4a4" />
                      <circle cx="320" cy="42" r="3" fill="#00d4a4" />
                      <text x="320" y="416" textAnchor="middle" fill="#00d4a4" fontSize="8" fontFamily="JetBrains Mono, monospace" letterSpacing="0.1em">
                        14 KEYPOINTS · HOMOGRAPHY LOCKED
                      </text>
                    </g>
                  </svg>
                  <div style={{ position: "absolute", top: 12, left: 16, fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.1em", color: "rgba(220,228,237,0.45)" }}>
                    RAW FRAME · 1920×1080
                  </div>
                  <div
                    style={{
                      position: "absolute",
                      left: "59%",
                      top: "44.5%",
                      opacity: hudOpacity.toFixed(2),
                      padding: "3px 9px",
                      borderRadius: 4,
                      background: "rgba(8,12,16,0.94)",
                      border: "1px solid #00d4a4",
                      fontFamily: "var(--font-mono)",
                      fontSize: 11,
                      fontWeight: 500,
                      color: "#00d4a4",
                      whiteSpace: "nowrap",
                      pointerEvents: "none",
                    }}
                  >
                    {netSpeed} km/h
                  </div>
                </div>
              </div>
            </div>

            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none" }}>
              <div
                style={{
                  transform: `rotateX(${topRotate.toFixed(1)}deg) scale(${topScale.toFixed(3)})`,
                  opacity: topOpacity.toFixed(2),
                  transformOrigin: "50% 50%",
                }}
              >
                <BounceHeatmap locations={visibleBounces} widthPx={180} />
              </div>
            </div>
          </div>

          {showHud && (
            <div
              style={{
                position: "absolute",
                left: 32,
                bottom: 32,
                border: "1px solid var(--border-default)",
                background: "rgba(13,21,32,0.85)",
                backdropFilter: "blur(12px)",
                borderRadius: 12,
                padding: "14px 16px",
                display: "grid",
                gap: 8,
                minWidth: 210,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#00d4a4", animation: "tvaPulse 2s ease-in-out infinite" }} />
                <span style={{ fontFamily: "var(--font-mono)", fontSize: 9, letterSpacing: "0.2em", color: "var(--text-muted)" }}>TELEMETRY</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: 11 }}>
                <span style={{ color: "var(--text-muted)" }}>FRAME</span>
                <span style={{ color: "var(--text-body)" }}>{hudFrame}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: 11 }}>
                <span style={{ color: "var(--text-muted)" }}>TRACK</span>
                <span style={{ color: "var(--text-body)" }}>{hudTrack}</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: 11 }}>
                <span style={{ color: "var(--text-muted)" }}>PEAK</span>
                <span style={{ color: "var(--text-accent)" }}>{hudSpeed} km/h</span>
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: 11 }}>
                <span style={{ color: "var(--text-muted)" }}>BOUNCES</span>
                <span style={{ color: "var(--text-accent)" }}>{hudBounces}</span>
              </div>
              <div style={{ height: 1, background: "var(--border-default)" }} />
              <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "var(--font-mono)", fontSize: 10 }}>
                <span style={{ color: "var(--text-muted)" }}>STAGE</span>
                <span style={{ color: "var(--text-body)" }}>{hudStage}</span>
              </div>
            </div>
          )}

          <div style={{ position: "absolute", right: 32, top: "50%", display: "grid", gap: 10 }}>
            <div style={{ width: 2, height: 44, background: "var(--border-default)", position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", left: 0, top: 0, width: 2, background: "#00d4a4", height: `${railFill}%` }} />
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 9, letterSpacing: "0.1em", color: "var(--text-muted)", writingMode: "vertical-rl" }}>
              {railLabel}
            </div>
          </div>
        </div>
      </div>

      {/* ── Capabilities (scroll cross-fade) ── */}
      <div id="capabilities" ref={panelsRef} style={{ position: "relative", height: "400vh" }}>
        <div style={{ position: "sticky", top: 0, height: "100vh", overflow: "hidden", display: "flex", alignItems: "center" }}>
          <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: "var(--border-default)" }} />
          <div style={{ position: "relative", width: "100%", maxWidth: 1152, margin: "0 auto", padding: "0 24px" }}>
            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", padding: "0 24px" }}>
              <div
                style={{
                  width: "100%",
                  opacity: pa[0].o.toFixed(2),
                  transform: `translateY(${pa[0].y}px)`,
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 56,
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.2em", color: "var(--text-accent)" }}>
                    01 — SHOT DETECTION
                  </div>
                  <h3
                    style={{
                      margin: "16px 0 0",
                      fontFamily: "var(--font-display)",
                      fontWeight: 700,
                      fontSize: "clamp(2rem, 3.4vw, 2.6rem)",
                      lineHeight: 1.15,
                      color: "var(--text-body)",
                    }}
                  >
                    Measure every shot's speed
                  </h3>
                  <p style={{ margin: "14px 0 0", fontSize: 15, lineHeight: 1.65, color: "var(--text-muted)", maxWidth: 420 }}>
                    Peak ball speed per detected shot, read where the fitted flight crosses the net — the lowest,
                    most accurate point in a monocular view. Readings above 300 km/h are discarded as detector
                    noise.
                  </p>
                </div>
                <div style={{ border: "1px solid var(--border-default)", borderRadius: 16, background: "var(--tva-panel)", padding: 24 }}>
                  <ShotSpeedChart shots={SAMPLE_SHOTS} />
                </div>
              </div>
            </div>

            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", padding: "0 24px" }}>
              <div
                style={{
                  width: "100%",
                  opacity: pa[1].o.toFixed(2),
                  transform: `translateY(${pa[1].y}px)`,
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 56,
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.2em", color: "var(--text-accent)" }}>
                    02 — BOUNCE ANALYSIS
                  </div>
                  <h3
                    style={{
                      margin: "16px 0 0",
                      fontFamily: "var(--font-display)",
                      fontWeight: 700,
                      fontSize: "clamp(2rem, 3.4vw, 2.6rem)",
                      lineHeight: 1.15,
                      color: "var(--text-body)",
                    }}
                  >
                    Map every bounce
                  </h3>
                  <p style={{ margin: "14px 0 0", fontSize: 15, lineHeight: 1.65, color: "var(--text-muted)", maxWidth: 420 }}>
                    A parabolic fit either side of each impact keeps only transitions a court surface could have
                    produced — so a racket contact is never mistaken for a landing. 2,841 bounces per match, placed
                    in metres.
                  </p>
                </div>
                <div style={{ display: "flex", justifyContent: "center" }}>
                  <BounceHeatmap locations={SAMPLE_BOUNCES} widthPx={220} />
                </div>
              </div>
            </div>

            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", padding: "0 24px" }}>
              <div
                style={{
                  width: "100%",
                  opacity: pa[2].o.toFixed(2),
                  transform: `translateY(${pa[2].y}px)`,
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 56,
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.2em", color: "var(--text-accent)" }}>
                    03 — CALIBRATION
                  </div>
                  <h3
                    style={{
                      margin: "16px 0 0",
                      fontFamily: "var(--font-display)",
                      fontWeight: 700,
                      fontSize: "clamp(2rem, 3.4vw, 2.6rem)",
                      lineHeight: 1.15,
                      color: "var(--text-body)",
                    }}
                  >
                    Calibrate in seconds
                  </h3>
                  <p style={{ margin: "14px 0 0", fontSize: 15, lineHeight: 1.65, color: "var(--text-muted)", maxWidth: 420 }}>
                    Click the four near-camera court corners, in order. That one-time 4-point calibration turns
                    px/s into real km/h and puts every landing on a to-scale court.
                  </p>
                </div>
                <div style={{ border: "1px solid var(--border-default)", borderRadius: 16, overflow: "hidden", background: "#172b22" }}>
                  <svg viewBox="0 0 640 430" style={{ width: "100%", display: "block" }}>
                    <rect width="640" height="430" fill="#10201a" />
                    <polygon points="70,395 570,395 480,42 160,42" fill="#1e5c40" />
                    <polygon points="70,395 570,395 480,42 160,42" fill="none" stroke="white" strokeWidth="2" />
                    <line x1="148" y1="395" x2="214" y2="42" stroke="white" strokeWidth="1.5" />
                    <line x1="492" y1="395" x2="426" y2="42" stroke="white" strokeWidth="1.5" />
                    <line x1="70" y1="218" x2="570" y2="218" stroke="#e0e0e0" strokeWidth="2.5" />
                    <line x1="136" y1="307" x2="504" y2="307" stroke="white" strokeWidth="1.5" />
                    <line x1="186" y1="129" x2="454" y2="129" stroke="white" strokeWidth="1.5" />
                    <line x1="320" y1="129" x2="320" y2="307" stroke="white" strokeWidth="1.5" />
                    <g>
                      <circle cx="82" cy="368" r="13" fill="#00d4a4" opacity="0.15" />
                      <circle cx="82" cy="368" r="8" fill="#00d4a4" />
                      <circle cx="558" cy="368" r="13" fill="#00d4a4" opacity="0.15" />
                      <circle cx="558" cy="368" r="8" fill="#00d4a4" />
                      <circle cx="178" cy="168" r="13" fill="#00d4a4" opacity="0.15" />
                      <circle cx="178" cy="168" r="8" fill="#00d4a4" />
                      <circle cx="462" cy="168" r="13" fill="#00d4a4" opacity="0.15" />
                      <circle cx="462" cy="168" r="8" fill="#00d4a4" />
                    </g>
                  </svg>
                </div>
              </div>
            </div>

            <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", padding: "0 24px" }}>
              <div
                style={{
                  width: "100%",
                  opacity: pa[3].o.toFixed(2),
                  transform: `translateY(${pa[3].y}px)`,
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 56,
                  alignItems: "center",
                }}
              >
                <div>
                  <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.2em", color: "var(--text-accent)" }}>
                    04 — EXPORT
                  </div>
                  <h3
                    style={{
                      margin: "16px 0 0",
                      fontFamily: "var(--font-display)",
                      fontWeight: 700,
                      fontSize: "clamp(2rem, 3.4vw, 2.6rem)",
                      lineHeight: 1.15,
                      color: "var(--text-body)",
                    }}
                  >
                    Take the whole match with you
                  </h3>
                  <p style={{ margin: "14px 0 0", fontSize: 15, lineHeight: 1.65, color: "var(--text-muted)", maxWidth: 420 }}>
                    An annotated 1080p · H.264 render with trail, landing markers and an optional sidebar readout —
                    plus the raw per-shot table behind it.
                  </p>
                </div>
                <div style={{ border: "1px solid var(--border-default)", borderRadius: 16, background: "var(--tva-panel)", padding: 24 }}>
                  <VideoFramePlaceholder fileName="federer_nadal_ao2017_annotated.mp4" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Stats band ── */}
      <div ref={statsRef} style={{ borderTop: "1px solid var(--border-default)", borderBottom: "1px solid var(--border-default)", background: "#0a1017" }}>
        <div style={{ maxWidth: 1152, margin: "0 auto", padding: "64px 24px", display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 32 }}>
          <div>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "3.5rem", lineHeight: 1, color: "var(--text-accent)" }}>
              {stat0}
            </div>
            <div style={{ marginTop: 8, fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.2em", color: "var(--text-muted)" }}>
              BOUNCES PER MATCH
            </div>
          </div>
          <div>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "3.5rem", lineHeight: 1, color: "var(--text-accent)" }}>
              {stat1}
            </div>
            <div style={{ marginTop: 8, fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.2em", color: "var(--text-muted)" }}>
              PEAK SPEED KM/H
            </div>
          </div>
          <div>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "3.5rem", lineHeight: 1, color: "var(--text-accent)" }}>
              {stat2}
            </div>
            <div style={{ marginTop: 8, fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.2em", color: "var(--text-muted)" }}>
              SHOTS SEGMENTED
            </div>
          </div>
          <div>
            <div style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: "3.5rem", lineHeight: 1, color: "var(--text-accent)" }}>
              {stat3}
            </div>
            <div style={{ marginTop: 8, fontFamily: "var(--font-mono)", fontSize: 10, letterSpacing: "0.2em", color: "var(--text-muted)" }}>
              MINUTES TO RESULT
            </div>
          </div>
        </div>
      </div>

      {/* ── Sample output ── */}
      <section id="results" style={{ maxWidth: 1152, margin: "0 auto", padding: "112px 24px" }}>
        <div style={{ textAlign: "center", marginBottom: 48 }}>
          <div style={{ fontFamily: "var(--font-mono)", fontSize: 11, letterSpacing: "0.2em", color: "var(--text-accent)" }}>
            SAMPLE OUTPUT
          </div>
          <h2
            style={{
              margin: "12px 0 0",
              fontFamily: "var(--font-display)",
              fontWeight: 700,
              fontSize: "clamp(2rem, 4vw, 3rem)",
              lineHeight: 1.15,
              color: "var(--text-body)",
            }}
          >
            What comes back
          </h2>
          <p style={{ margin: "10px auto 0", maxWidth: 520, fontSize: 14, lineHeight: 1.65, color: "var(--text-muted)" }}>
            federer_nadal_ao2017.mp4 — 3h 22m, analysed in 4m 11s.
          </p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1.4fr 1fr", gap: 20, alignItems: "start" }}>
          <div className="card">
            <h3 className="card-title">Shot speeds</h3>
            <p className="card-subtitle">Peak ball speed per detected shot (km/h) — hover for value</p>
            <div style={{ marginTop: 20 }}>
              <ShotSpeedChart shots={SAMPLE_SHOTS} />
            </div>
            <div style={{ marginTop: 24 }}>
              <ShotStatsTable shots={SAMPLE_SHOTS} />
            </div>
          </div>
          <div style={{ display: "grid", gap: 20 }}>
            <div className="card">
              <h3 className="card-title">Match summary</h3>
              <div style={{ marginTop: 20, display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
                <StatTile value="142" sub="detected" label="Rallies" />
                <StatTile value="2,841" sub="detected" label="Bounces" />
                <StatTile value="1,398" sub="detected" label="Contacts" />
              </div>
            </div>
            <div className="card">
              <h3 className="card-title">Bounce landing heatmap</h3>
              <div style={{ marginTop: 24, display: "flex", justifyContent: "center" }}>
                <BounceHeatmap locations={SAMPLE_BOUNCES} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section style={{ position: "relative", borderTop: "1px solid var(--border-default)", overflow: "hidden" }}>
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "radial-gradient(ellipse 60% 40% at 50% 50%, rgba(0,212,164,0.06), transparent)",
            pointerEvents: "none",
          }}
        />
        <div style={{ position: "relative", maxWidth: 1152, margin: "0 auto", padding: "112px 24px", textAlign: "center" }}>
          <h2
            style={{
              margin: 0,
              fontFamily: "var(--font-display)",
              fontWeight: 700,
              fontSize: "clamp(2rem, 4vw, 3rem)",
              lineHeight: 1.15,
              color: "var(--text-body)",
            }}
          >
            See your game differently
          </h2>
          <p style={{ margin: "14px auto 32px", maxWidth: 460, fontSize: 15, lineHeight: 1.65, color: "var(--text-muted)" }}>
            Upload a match, pick your toggles, and get the numbers back before you have left the court.
          </p>
          <div style={{ display: "flex", justifyContent: "center", gap: 12, flexWrap: "wrap" }}>
            <button className="btn btn-primary" style={{ padding: "12px 22px", fontSize: 15 }} onClick={() => navigate("/register")}>
              Start analysing free
            </button>
            <button className="btn btn-secondary" style={{ padding: "12px 22px", fontSize: 15 }} onClick={() => navigate("/login")}>
              Sign in
            </button>
          </div>
        </div>
      </section>

      <footer style={{ borderTop: "1px solid var(--border-default)" }}>
        <div
          style={{
            maxWidth: 1152,
            margin: "0 auto",
            padding: "32px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 24,
            flexWrap: "wrap",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Logo size={18} />
            <span className="brand-wordmark" style={{ fontSize: 15 }}>
              Tennis<span className="accent">VA</span>
            </span>
          </span>
        </div>
      </footer>
    </div>
  );
}
