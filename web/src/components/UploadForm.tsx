"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

const THEMES = [
  { key: "netplay", label: "Net play" },
  { key: "consistency", label: "Consistency" },
  { key: "adjustment", label: "Adjustment (slice)" },
  { key: "around_backhand", label: "Around the backhand" },
];

function size(bytes: number) {
  return bytes > 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

// Today where the viewer is, not on the server.
function localToday() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export function UploadForm({ coaches, fixedCoach }: {
  coaches: { id: number; name: string }[] | null; // null: a coach uploading their own
  fixedCoach: string | null;
}) {
  const [today] = useState(localToday);
  const router = useRouter();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [over, setOver] = useState(false);
  const [coach, setCoach] = useState(coaches?.[0]?.id ? String(coaches[0].id) : "new");
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState<number | null>(null); // bytes sent, while uploading

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    if (!file) return setError("Choose the session video.");
    const f = new FormData(e.currentTarget);
    const meta = {
      title: f.get("title"), date: f.get("date"), court: f.get("court"), session_type: f.get("session_type"),
      themes: f.getAll("themes"), file_name: file.name,
      ...(coaches ? (coach === "new" ? { coach_name: f.get("coach_name") } : { coach_id: Number(coach) }) : {}),
    };
    const res = await fetch("/api/uploads", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(meta) });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) return setError(body.error ?? "Could not start the upload.");

    // The video goes up in one streamed PUT; XHR because fetch can't report upload progress.
    setSent(0);
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", `/api/uploads/${body.id}/file`);
    xhr.upload.onprogress = (ev) => setSent(ev.loaded);
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) router.push("/uploads");
      else { setSent(null); setError(JSON.parse(xhr.responseText || "{}").error ?? `Upload failed (${xhr.status}).`); }
    };
    xhr.onerror = () => { setSent(null); setError("The connection dropped. The upload is listed under Uploads - delete it and try again."); };
    xhr.send(file);
  }

  const busy = sent != null;
  return (
    <form className="form" onSubmit={submit}>
      <div
        className={`drop${over ? " over" : ""}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); }}
        role="button" tabIndex={0}
        onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") input.current?.click(); }}
      >
        {file ? <><b>{file.name}</b><br /><span className="muted">{size(file.size)} · click to choose another</span></>
              : <><b>Drop the session video here</b><br /><span className="muted">or click to choose - .mp4 or .mov from the court camera</span></>}
        <input ref={input} type="file" accept="video/*,.mp4,.mov,.m4v,.mkv,.avi" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>

      {coaches ? (
        <div className="row">
          <label><span>Coach who ran it</span>
            <select value={coach} onChange={(e) => setCoach(e.target.value)}>
              {coaches.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              <option value="new">Someone else…</option>
            </select>
          </label>
          {coach === "new" && <label><span>Their name</span><input name="coach_name" required /></label>}
        </div>
      ) : <p className="muted small">Filed under your sessions ({fixedCoach}).</p>}

      <div className="row">
        <label><span>Date</span><input name="date" type="date" defaultValue={today} required suppressHydrationWarning /></label>
        <label><span>Session type</span>
          <select name="session_type" defaultValue="squad">
            <option value="squad">Squad session</option>
            <option value="private">Private lesson</option>
          </select>
        </label>
      </div>
      <div className="row">
        <label><span>Court</span><input name="court" placeholder="e.g. Court 3" /></label>
        <label><span>Title (optional)</span><input name="title" placeholder="e.g. Year 9 squad" /></label>
      </div>
      <fieldset>
        <legend>What the session was working on</legend>
        {THEMES.map((t) => <label key={t.key} className="check"><input type="checkbox" name="themes" value={t.key} /> {t.label}</label>)}
      </fieldset>

      {error && <p className="form-error" role="alert">{error}</p>}
      {busy && file && (
        <div>
          <div className="progress"><i style={{ width: `${(100 * (sent ?? 0)) / file.size}%` }} /></div>
          <p className="muted small">Uploading {size(sent ?? 0)} of {size(file.size)} - keep this page open.</p>
        </div>
      )}
      <div><button className="btn primary" disabled={busy}>{busy ? "Uploading…" : "Upload and analyse"}</button></div>
    </form>
  );
}
