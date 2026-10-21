// The app's data: coaches, courts, sessions and their reports.
//
// Locally this is a SQLite file (Node's built-in node:sqlite - nothing to
// install or compile). Every query is plain SQL in src/lib (this file,
// users.ts, uploads.ts), so moving to Postgres for the cloud means changing
// those only.

import { DatabaseSync } from "node:sqlite";
import fs from "node:fs";
import path from "node:path";

// turbopackIgnore: these are runtime data paths, not files to bundle.
export const APP_ROOT = /* turbopackIgnore: true */ process.cwd(); // web/
export const REPO_ROOT = path.resolve(APP_ROOT, "..");
const DB_PATH = process.env.APP_DB ?? path.join(APP_ROOT, "data", "app.db");
// Files the app stores itself (uploaded clips, processed videos). Paths in
// the database under "media/" are relative to this; anything else is
// relative to the repository (sessions published from outputs/).
export const MEDIA_DIR = process.env.APP_MEDIA ?? path.join(APP_ROOT, "data", "media");

export function resolveMedia(stored: string): string {
  if (path.isAbsolute(stored)) return stored;
  if (stored.startsWith("media/")) return path.join(/* turbopackIgnore: true */ MEDIA_DIR, stored.slice("media/".length));
  return path.join(/* turbopackIgnore: true */ REPO_ROOT, stored);
}

const SCHEMA = `
CREATE TABLE IF NOT EXISTS coaches (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  is_demo INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS courts (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  date TEXT NOT NULL,                 -- ISO date the session was played
  coach_id INTEGER NOT NULL REFERENCES coaches(id),
  court_id INTEGER REFERENCES courts(id),
  session_type TEXT NOT NULL,         -- 'squad' | 'private'
  themes TEXT NOT NULL DEFAULT '',    -- comma-separated lesson themes
  duration_s REAL,
  video_path TEXT,                    -- path to the video served on the report page
  report_json TEXT NOT NULL,          -- the backend's session_report.json
  feedback_json TEXT,                 -- session_feedback.json, if any
  metrics_json TEXT NOT NULL,         -- the comparable metrics, extracted at import
  exemplary INTEGER NOT NULL DEFAULT 0,  -- marked by the head coach: feeds the benchmark
  is_demo INTEGER NOT NULL DEFAULT 0,
  imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS sessions_coach ON sessions(coach_id, date);
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY,
  email TEXT NOT NULL UNIQUE COLLATE NOCASE,
  name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL,                 -- 'head_coach' | 'coach'
  coach_id INTEGER REFERENCES coaches(id),  -- a coach account sees this coach's sessions
  must_change_password INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS logins (
  token_hash TEXT PRIMARY KEY,        -- sha256 of the cookie's token; the token itself is never stored
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS uploads (
  id INTEGER PRIMARY KEY,
  created_by INTEGER REFERENCES users(id),
  coach_id INTEGER NOT NULL REFERENCES coaches(id),
  title TEXT NOT NULL,
  date TEXT NOT NULL,
  court TEXT,
  session_type TEXT NOT NULL,
  themes TEXT NOT NULL DEFAULT '',
  original_name TEXT,
  file_path TEXT,                     -- the clip, under media/
  size_bytes INTEGER,
  status TEXT NOT NULL DEFAULT 'uploading',  -- uploading | queued | processing | done | failed
  step TEXT,                          -- the backend step running now
  steps_done INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  worker TEXT,
  session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  started_at TEXT,
  finished_at TEXT
);
`;

let db: DatabaseSync | null = null;

export function getDb(): DatabaseSync {
  if (!db) {
    fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
    db = new DatabaseSync(DB_PATH);
    db.exec("PRAGMA foreign_keys = ON;");
    db.exec(SCHEMA);
  }
  return db;
}

// ---- types -------------------------------------------------------------

export type SessionType = "squad" | "private";
export const SESSION_TYPES: { key: SessionType; label: string }[] = [
  { key: "squad", label: "Squad session" },
  { key: "private", label: "Private lesson" },
];

export interface Coach {
  id: number;
  name: string;
  is_demo: number;
}

export interface SessionRow {
  id: number;
  title: string;
  date: string;
  coach_id: number;
  coach_name: string;
  court_name: string | null;
  session_type: SessionType;
  themes: string;
  duration_s: number | null;
  video_path: string | null;
  metrics: Record<string, number | null>;
  exemplary: boolean;
  is_demo: boolean;
}

export interface SessionFull extends SessionRow {
  report: Record<string, unknown>;
  feedback: Record<string, unknown> | null;
}

type Raw = Record<string, unknown>;

function toRow(r: Raw): SessionRow {
  return {
    id: Number(r.id),
    title: String(r.title),
    date: String(r.date),
    coach_id: Number(r.coach_id),
    coach_name: String(r.coach_name),
    court_name: (r.court_name as string) ?? null,
    session_type: r.session_type as SessionType,
    themes: String(r.themes ?? ""),
    duration_s: r.duration_s == null ? null : Number(r.duration_s),
    video_path: (r.video_path as string) ?? null,
    metrics: JSON.parse(String(r.metrics_json)),
    exemplary: Number(r.exemplary) === 1,
    is_demo: Number(r.is_demo) === 1,
  };
}

const SESSION_SELECT = `
  SELECT s.id, s.title, s.date, s.coach_id, c.name AS coach_name, ct.name AS court_name,
         s.session_type, s.themes, s.duration_s, s.video_path, s.metrics_json, s.exemplary, s.is_demo
  FROM sessions s JOIN coaches c ON c.id = s.coach_id LEFT JOIN courts ct ON ct.id = s.court_id`;

// ---- reads -------------------------------------------------------------

export function listCoaches(): Coach[] {
  return getDb().prepare("SELECT id, name, is_demo FROM coaches ORDER BY name").all() as unknown as Coach[];
}

export function getCoach(id: number): Coach | null {
  return (getDb().prepare("SELECT id, name, is_demo FROM coaches WHERE id = ?").get(id) as unknown as Coach) ?? null;
}

export function coachIdByName(name: string, isDemo = false): number {
  return idFor("coaches", name, isDemo);
}

export function listSessions(filter: { coachId?: number; type?: SessionType; exemplary?: boolean } = {}): SessionRow[] {
  const where: string[] = [];
  const args: (number | string)[] = [];
  if (filter.coachId != null) { where.push("s.coach_id = ?"); args.push(filter.coachId); }
  if (filter.type) { where.push("s.session_type = ?"); args.push(filter.type); }
  if (filter.exemplary != null) { where.push("s.exemplary = ?"); args.push(filter.exemplary ? 1 : 0); }
  const sql = `${SESSION_SELECT} ${where.length ? "WHERE " + where.join(" AND ") : ""} ORDER BY s.date, s.id`;
  return (getDb().prepare(sql).all(...args) as Raw[]).map(toRow);
}

export function getSession(id: number): SessionFull | null {
  const r = getDb().prepare(`${SESSION_SELECT.replace("s.metrics_json,", "s.metrics_json, s.report_json, s.feedback_json,")} WHERE s.id = ?`).get(id) as Raw | undefined;
  if (!r) return null;
  return {
    ...toRow(r),
    report: JSON.parse(String(r.report_json)),
    feedback: r.feedback_json ? JSON.parse(String(r.feedback_json)) : null,
  };
}

// ---- writes ------------------------------------------------------------

function idFor(table: "coaches" | "courts", name: string, isDemo = false): number {
  const d = getDb();
  const found = d.prepare(`SELECT id FROM ${table} WHERE name = ?`).get(name) as { id: number } | undefined;
  if (found) return Number(found.id);
  const res = table === "coaches"
    ? d.prepare("INSERT INTO coaches (name, is_demo) VALUES (?, ?)").run(name, isDemo ? 1 : 0)
    : d.prepare("INSERT INTO courts (name) VALUES (?)").run(name);
  return Number(res.lastInsertRowid);
}

export interface NewSession {
  title: string;
  date: string;
  coach: string;
  court?: string | null;
  session_type: SessionType;
  themes?: string[];
  duration_s?: number | null;
  video_path?: string | null;
  report: Record<string, unknown>;
  feedback?: Record<string, unknown> | null;
  metrics: Record<string, number | null>;
  exemplary?: boolean;
  is_demo?: boolean;
}

export function insertSession(s: NewSession): number {
  const coachId = idFor("coaches", s.coach, s.is_demo);
  const courtId = s.court ? idFor("courts", s.court) : null;
  const res = getDb().prepare(`
    INSERT INTO sessions (title, date, coach_id, court_id, session_type, themes, duration_s, video_path,
                          report_json, feedback_json, metrics_json, exemplary, is_demo)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
    s.title, s.date, coachId, courtId, s.session_type, (s.themes ?? []).join(","), s.duration_s ?? null,
    s.video_path ?? null, JSON.stringify(s.report), s.feedback ? JSON.stringify(s.feedback) : null,
    JSON.stringify(s.metrics), s.exemplary ? 1 : 0, s.is_demo ? 1 : 0,
  );
  return Number(res.lastInsertRowid);
}

export function setExemplary(id: number, exemplary: boolean): void {
  getDb().prepare("UPDATE sessions SET exemplary = ? WHERE id = ?").run(exemplary ? 1 : 0, id);
}

export function deleteDemoData(): void {
  const d = getDb();
  d.exec("DELETE FROM sessions WHERE is_demo = 1; DELETE FROM coaches WHERE is_demo = 1 AND id NOT IN (SELECT coach_id FROM sessions);");
}
