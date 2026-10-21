"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { Upload } from "@/lib/uploads";

// Stored in UTC (SQLite datetime('now')); shown in the viewer's time.
function local(utc: string) {
  return new Date(utc.replace(" ", "T") + "Z").toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

const LABEL: Record<Upload["status"], string> = {
  uploading: "uploading", queued: "waiting", processing: "analysing", done: "done", failed: "failed",
};

export function UploadList({ initial, totalSteps, showCoach }: { initial: Upload[]; totalSteps: number; showCoach: boolean }) {
  const [uploads, setUploads] = useState(initial);
  const [open, setOpen] = useState<number | null>(null);
  const active = uploads.some((u) => u.status === "queued" || u.status === "processing" || u.status === "uploading");

  async function refresh() {
    const res = await fetch("/api/uploads");
    if (res.ok) setUploads(await res.json());
  }
  useEffect(() => {
    if (!active) return;
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [active]);

  async function act(id: number, how: "retry" | "delete") {
    if (how === "delete" && !confirm("Delete this upload and its video?")) return;
    const stuck = uploads.find((u) => u.id === id)?.status === "processing";
    if (how === "retry" && stuck && !confirm("Only do this if the analysis computer has stopped - otherwise the clip is analysed twice. Start it again?")) return;
    await fetch(how === "retry" ? `/api/uploads/${id}/retry` : `/api/uploads/${id}`, { method: how === "retry" ? "POST" : "DELETE" });
    refresh();
  }

  if (!uploads.length) return <div className="panel muted">Nothing uploaded yet. <Link href="/upload">Upload a session</Link>.</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Uploaded</th><th>Session</th>{showCoach && <th>Coach</th>}<th>Status</th><th>Progress</th><th /></tr></thead>
        <tbody>
          {uploads.map((u) => (
            <tr key={u.id}>
              <td className="n nw" suppressHydrationWarning>{local(u.created_at)}</td>
              <td className="title">
                {u.session_id ? <Link href={`/sessions/${u.session_id}`}>{u.title}</Link> : u.title}
                <div className="muted small">{u.date}{u.court ? ` · ${u.court}` : ""} · {u.original_name}</div>
              </td>
              {showCoach && <td className="nw">{u.coach_name}</td>}
              <td><span className={`status ${u.status}`}>{LABEL[u.status]}</span></td>
              <td>
                {u.status === "processing" && <>
                  <div className="progress"><i style={{ width: `${Math.min(95, (100 * u.steps_done) / totalSteps)}%` }} /></div>
                  <div className="muted small">{u.step ?? "starting"}</div>
                </>}
                {u.status === "queued" && <span className="muted small">waiting for the analysis computer</span>}
                {u.status === "uploading" && <span className="muted small">video not received</span>}
                {u.status === "done" && u.session_id && <Link href={`/sessions/${u.session_id}`}>Open report →</Link>}
                {u.status === "failed" && <button className="link-btn" onClick={() => setOpen(open === u.id ? null : u.id)}>{open === u.id ? "hide" : "what went wrong"}</button>}
                {open === u.id && u.error && <pre className="err">{u.error}</pre>}
              </td>
              <td className="nw">
                {(u.status === "failed" || u.status === "processing") && <button className="btn" onClick={() => act(u.id, "retry")}
                  title={u.status === "processing" ? "Only if the analysis computer stopped - it starts again" : undefined}>Retry</button>}{" "}
                {u.status !== "processing" && u.status !== "done" && <button className="btn danger" onClick={() => act(u.id, "delete")}>Delete</button>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
