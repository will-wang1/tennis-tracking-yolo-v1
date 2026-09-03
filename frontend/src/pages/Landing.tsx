import { Link } from "react-router-dom";
import HeroCourt from "../components/HeroCourt";

const CAPABILITIES = [
  {
    eyebrow: "01 — SHOT DETECTION",
    title: "Measure every shot's speed",
    body: "Peak ball speed per detected shot, read where the fitted flight crosses the net — the lowest, most accurate point in a monocular view. Reported in km/h once a video is calibrated, px/s otherwise.",
  },
  {
    eyebrow: "02 — BOUNCE ANALYSIS",
    title: "Map every bounce",
    body: "A parabolic-arc fit separates real court bounces from racket contacts and detector noise — the ball descending in, rising out, giving back less than it received. Every landing spot plotted on a to-scale court diagram.",
  },
  {
    eyebrow: "03 — COURT CALIBRATION",
    title: "Calibrate in seconds",
    body: "Click the four near-camera court corners once, in the browser — no separate tool, no manual pixel-reading. That single 4-point homography is what turns raw pixels into real-world speed and landing coordinates.",
  },
  {
    eyebrow: "04 — RESULTS",
    title: "See the whole match at a glance",
    body: "The annotated video, a shot-speed chart and table, rally/bounce/contact totals, and a bounce landing heatmap — one results page per upload, ready to review or download.",
  },
];

export default function Landing() {
  return (
    <div>
      <section
        style={{
          maxWidth: "var(--container-marketing, 1152px)",
          margin: "0 auto",
          padding: "64px 20px 48px",
          display: "grid",
          gridTemplateColumns: "1.1fr 1fr",
          gap: 48,
          alignItems: "center",
        }}
      >
        <div>
          <div
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              letterSpacing: "var(--tracking-eyebrow)",
              textTransform: "uppercase",
              color: "var(--text-accent)",
            }}
          >
            Match video analysis
          </div>
          <h1
            style={{
              fontSize: "clamp(2.5rem, 5vw, 4rem)",
              lineHeight: "var(--leading-hero, 0.95)",
              margin: "12px 0 20px",
            }}
          >
            See every shot.
            <br />
            <span className="accent" style={{ color: "var(--text-accent)" }}>
              Miss nothing.
            </span>
          </h1>
          <p style={{ fontSize: 16, lineHeight: 1.6, color: "var(--text-body)", maxWidth: 480 }}>
            Upload a match video. A computer-vision pipeline tracks the ball, detects bounces,
            measures shot speed, and maps every landing spot on the court — no manual tagging,
            no stopwatch.
          </p>
          <div style={{ display: "flex", gap: 12, marginTop: 28 }}>
            <Link to="/register" className="btn btn-primary">
              Get started
            </Link>
            <Link to="/login" className="btn btn-secondary">
              Log in
            </Link>
          </div>
        </div>

        <div
          style={{
            position: "relative",
            background: "var(--surface-card)",
            border: "1px solid var(--border-default)",
            borderRadius: "var(--radius-xl)",
            padding: 16,
          }}
        >
          <HeroCourt />
          <div
            style={{
              position: "absolute",
              top: 28,
              left: 28,
              background: "rgba(8,12,16,0.85)",
              border: "1px solid var(--border-default)",
              borderRadius: "var(--radius-md)",
              padding: "10px 14px",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              display: "grid",
              gridTemplateColumns: "auto auto",
              gap: "4px 16px",
              color: "var(--text-muted)",
            }}
          >
            <span>TRACK</span>
            <span style={{ color: "var(--text-body)", textAlign: "right" }}>0.94</span>
            <span>PEAK</span>
            <span style={{ color: "var(--text-accent)", textAlign: "right" }}>192 km/h</span>
            <span>BOUNCES</span>
            <span style={{ color: "var(--text-body)", textAlign: "right" }}>3</span>
            <span>STAGE</span>
            <span style={{ color: "var(--text-body)", textAlign: "right" }}>SPEED</span>
          </div>
        </div>
      </section>

      <section
        style={{
          borderTop: "1px solid var(--border-default)",
          borderBottom: "1px solid var(--border-default)",
        }}
      >
        <div
          style={{
            maxWidth: "var(--container-marketing, 1152px)",
            margin: "0 auto",
            padding: "56px 20px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
            gap: 32,
          }}
        >
          {CAPABILITIES.map((c) => (
            <div key={c.eyebrow}>
              <div
                style={{
                  fontFamily: "var(--font-mono)",
                  fontSize: 11,
                  letterSpacing: "var(--tracking-eyebrow)",
                  color: "var(--text-accent)",
                }}
              >
                {c.eyebrow}
              </div>
              <h3 style={{ fontSize: 22, margin: "8px 0 8px" }}>{c.title}</h3>
              <p style={{ fontSize: 14, lineHeight: 1.6, color: "var(--text-muted)", margin: 0 }}>
                {c.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      <section style={{ textAlign: "center", padding: "64px 20px" }}>
        <h2 style={{ fontSize: 28, margin: "0 0 12px" }}>Built for serious analysis</h2>
        <p style={{ color: "var(--text-muted)", margin: "0 0 24px" }}>
          Free to try — upload a clip and see what it finds.
        </p>
        <Link to="/register" className="btn btn-primary">
          Get started
        </Link>
      </section>
    </div>
  );
}
