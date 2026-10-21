// Uploaded clips and their way through the processing worker:
// uploading -> queued -> processing -> done (a session) | failed.

import { getDb, type SessionType } from "./db";

export type UploadStatus = "uploading" | "queued" | "processing" | "done" | "failed";

export interface Upload {
  id: number;
  created_by: number | null;
  created_by_name: string | null;
  coach_id: number;
  coach_name: string;
  title: string;
  date: string;
  court: string | null;
  session_type: SessionType;
  themes: string[];
  original_name: string | null;
  file_path: string | null;
  size_bytes: number | null;
  status: UploadStatus;
  step: string | null;
  steps_done: number;
  error: string | null;
  worker: string | null;
  session_id: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

type Raw = Record<string, unknown>;
const SELECT = `SELECT up.*, c.name AS coach_name, u.name AS created_by_name
  FROM uploads up JOIN coaches c ON c.id = up.coach_id LEFT JOIN users u ON u.id = up.created_by`;

function toUpload(r: Raw): Upload {
  return {
    ...(r as unknown as Upload),
    id: Number(r.id),
    coach_id: Number(r.coach_id),
    created_by: r.created_by == null ? null : Number(r.created_by),
    size_bytes: r.size_bytes == null ? null : Number(r.size_bytes),
    steps_done: Number(r.steps_done),
    session_id: r.session_id == null ? null : Number(r.session_id),
    themes: String(r.themes ?? "").split(",").filter(Boolean),
  };
}

export function createUpload(u: {
  createdBy: number; coachId: number; title: string; date: string; court: string | null;
  sessionType: SessionType; themes: string[]; originalName: string;
}): number {
  const res = getDb().prepare(`INSERT INTO uploads (created_by, coach_id, title, date, court, session_type, themes, original_name)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)`).run(u.createdBy, u.coachId, u.title, u.date, u.court, u.sessionType, u.themes.join(","), u.originalName);
  return Number(res.lastInsertRowid);
}

export function getUpload(id: number): Upload | null {
  const r = getDb().prepare(`${SELECT} WHERE up.id = ?`).get(id) as Raw | undefined;
  return r ? toUpload(r) : null;
}

export function listUploads(coachId?: number): Upload[] {
  const rows = coachId == null
    ? getDb().prepare(`${SELECT} ORDER BY up.id DESC`).all()
    : getDb().prepare(`${SELECT} WHERE up.coach_id = ? ORDER BY up.id DESC`).all(coachId);
  return (rows as Raw[]).map(toUpload);
}

export function fileReceived(id: number, filePath: string, size: number): void {
  getDb().prepare("UPDATE uploads SET file_path = ?, size_bytes = ?, status = 'queued' WHERE id = ?").run(filePath, size, id);
}

/** The oldest queued upload, now marked as this worker's - atomically, so
 * two workers never take the same clip. */
export function claimNext(worker: string): Upload | null {
  const r = getDb().prepare(`UPDATE uploads SET status = 'processing', worker = ?, started_at = datetime('now'),
      step = NULL, steps_done = 0, error = NULL
    WHERE id = (SELECT id FROM uploads WHERE status = 'queued' ORDER BY id LIMIT 1) RETURNING id`).get(worker) as { id: number } | undefined;
  return r ? getUpload(Number(r.id)) : null;
}

export function reportProgress(id: number, step: string, stepsDone: number): void {
  getDb().prepare("UPDATE uploads SET step = ?, steps_done = ? WHERE id = ? AND status = 'processing'").run(step, stepsDone, id);
}

export function markFailed(id: number, error: string): void {
  getDb().prepare("UPDATE uploads SET status = 'failed', error = ?, finished_at = datetime('now') WHERE id = ?").run(error.slice(-4000), id);
}

export function markDone(id: number, sessionId: number): void {
  getDb().prepare("UPDATE uploads SET status = 'done', session_id = ?, step = NULL, finished_at = datetime('now') WHERE id = ?").run(sessionId, id);
}

export function requeue(id: number): void {
  getDb().prepare("UPDATE uploads SET status = 'queued', error = NULL, step = NULL, steps_done = 0, worker = NULL WHERE id = ? AND file_path IS NOT NULL").run(id);
}

export function deleteUpload(id: number): void {
  getDb().prepare("DELETE FROM uploads WHERE id = ?").run(id);
}

/** Steps the backend runs (scripts/run_session.py) - for the progress bar. */
export const TOTAL_STEPS = 13;
