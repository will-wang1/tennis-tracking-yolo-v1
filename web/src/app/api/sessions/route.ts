// Import a processed session: the backend's report plus who, where and what.
// Called by scripts/publish_session.py (and, in the cloud, by the processing
// worker once a clip has been analysed).

import { insertSession, SESSION_TYPES, type SessionType } from "@/lib/db";
import { extractMetrics } from "@/lib/metrics";

const TOKEN = process.env.IMPORT_TOKEN ?? "dev-import-token";

export async function POST(req: Request) {
  if (req.headers.get("authorization") !== `Bearer ${TOKEN}`) {
    return Response.json({ error: "bad or missing import token" }, { status: 401 });
  }
  const body = await req.json().catch(() => null);
  if (!body || typeof body !== "object") return Response.json({ error: "expected a JSON body" }, { status: 400 });
  const { report, coach, date, session_type } = body;
  if (!report || typeof report !== "object") return Response.json({ error: "report is required" }, { status: 400 });
  if (!coach || typeof coach !== "string") return Response.json({ error: "coach is required" }, { status: 400 });
  if (!date || !/^\d{4}-\d{2}-\d{2}/.test(date)) return Response.json({ error: "date must be YYYY-MM-DD" }, { status: 400 });
  if (!SESSION_TYPES.some((t) => t.key === session_type)) {
    return Response.json({ error: `session_type must be one of ${SESSION_TYPES.map((t) => t.key).join(", ")}` }, { status: 400 });
  }
  const id = insertSession({
    title: body.title || `${coach} - ${date}`,
    date: date.slice(0, 10),
    coach,
    court: body.court ?? null,
    session_type: session_type as SessionType,
    themes: Array.isArray(body.themes) ? body.themes : [],
    duration_s: report?.clip?.duration_s ?? null,
    video_path: body.video_path ?? null,
    report,
    feedback: body.feedback ?? null,
    metrics: extractMetrics(report),
    exemplary: Boolean(body.exemplary),
    is_demo: Boolean(body.is_demo),
  });
  return Response.json({ id }, { status: 201 });
}
