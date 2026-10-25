// Nightly upkeep on the server (deploy/backup.sh runs it; safe to run by hand):
//
//   1. back up the database: a consistent copy to $BACKUP_DIR/db/app-<date>.db,
//      keeping the newest $BACKUP_KEEP (default 14)
//   2. delete session videos older than $VIDEO_RETENTION_DAYS (unset: keep
//      them); the session and its report stay, only the video goes
//   3. delete the clips of uploads that failed more than
//      $FAILED_UPLOAD_DAYS ago (default 14) - nobody retried them
//
//   APP_DB=... APP_MEDIA=... BACKUP_DIR=... node scripts/maintenance.mjs

import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

const APP_ROOT = path.resolve(import.meta.dirname, "..");
const DB_PATH = process.env.APP_DB ?? path.join(APP_ROOT, "data", "app.db");
const MEDIA_DIR = process.env.APP_MEDIA ?? path.join(APP_ROOT, "data", "media");
const BACKUP_DIR = process.env.BACKUP_DIR;
const BACKUP_KEEP = Number(process.env.BACKUP_KEEP ?? 14);
const VIDEO_DAYS = process.env.VIDEO_RETENTION_DAYS ? Number(process.env.VIDEO_RETENTION_DAYS) : null;
const FAILED_DAYS = Number(process.env.FAILED_UPLOAD_DAYS ?? 14);

if (!fs.existsSync(DB_PATH)) throw new Error(`No database at ${DB_PATH}`);
const db = new DatabaseSync(DB_PATH);
db.exec("PRAGMA busy_timeout = 5000");

// Only files the app stored itself (media/...) are ever deleted - never a
// path into the repo, like a video published from outputs/ by hand.
const mediaFile = (stored) => stored?.startsWith("media/") ? path.join(MEDIA_DIR, stored.slice("media/".length)) : null;

if (BACKUP_DIR) {
  const dir = path.join(BACKUP_DIR, "db");
  fs.mkdirSync(dir, { recursive: true });
  const target = path.join(dir, `app-${new Date().toISOString().slice(0, 10)}.db`);
  fs.rmSync(target, { force: true });
  db.exec(`VACUUM INTO '${target.replaceAll("'", "''")}'`);
  const old = fs.readdirSync(dir).filter((f) => /^app-\d{4}-\d{2}-\d{2}\.db$/.test(f)).sort().reverse().slice(BACKUP_KEEP);
  for (const f of old) fs.rmSync(path.join(dir, f));
  console.log(`database backed up to ${target}${old.length ? ` (removed ${old.length} old)` : ""}`);
} else {
  console.log("BACKUP_DIR not set - no database backup");
}

if (VIDEO_DAYS != null) {
  const expired = db.prepare(`SELECT id, video_path FROM sessions
    WHERE video_path LIKE 'media/%' AND imported_at < datetime('now', ?)`).all(`-${VIDEO_DAYS} days`);
  for (const s of expired) {
    fs.rmSync(mediaFile(s.video_path), { force: true });
    db.prepare("UPDATE sessions SET video_path = NULL WHERE id = ?").run(s.id);
  }
  console.log(`videos older than ${VIDEO_DAYS} days: removed ${expired.length}`);
}

const stale = db.prepare(`SELECT id, file_path FROM uploads
  WHERE status = 'failed' AND file_path LIKE 'media/%' AND finished_at < datetime('now', ?)`).all(`-${FAILED_DAYS} days`);
for (const u of stale) {
  fs.rmSync(mediaFile(u.file_path), { force: true });
  db.prepare("UPDATE uploads SET file_path = NULL WHERE id = ?").run(u.id);
}
console.log(`clips of uploads failed over ${FAILED_DAYS} days ago: removed ${stale.length}`);
db.close();
