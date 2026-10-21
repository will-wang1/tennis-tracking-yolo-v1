"use client";

import { useRef } from "react";

export interface VideoEvent { t: number; label: string; kind: string }

function clock(t: number) {
  const m = Math.floor(t / 60), s = Math.floor(t % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function SessionVideo({ src, events }: { src: string | null; events: VideoEvent[] }) {
  const ref = useRef<HTMLVideoElement>(null);
  function jump(t: number) {
    const v = ref.current;
    if (!v) return;
    v.currentTime = Math.max(0, t - 1);
    v.play().catch(() => {});
  }
  return (
    <div>
      {src ? <video ref={ref} src={src} controls preload="metadata" playsInline /> : <div className="panel muted">No video for this session.</div>}
      {events.length > 0 && (
        <div className="events" aria-label="Rallies - click to watch">
          {events.map((e, i) => (
            <button key={i} onClick={() => jump(e.t)} disabled={!src}>
              <span className="t">{clock(e.t)}</span>
              <span className={`ek ${e.kind}`}>{e.label}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
