// Start an upload: the session's details first, then the clip itself is
// PUT to /api/uploads/[id]/file. GET lists uploads for the status page.

import { coachIdByName, getCoach, SESSION_TYPES, type SessionType } from "@/lib/db";
import { apiUser } from "@/lib/auth";
import { createUpload, listUploads } from "@/lib/uploads";

const THEMES = ["netplay", "consistency", "adjustment", "around_backhand"];

export async function GET() {
  const user = await apiUser();
  if (!user) return Response.json({ error: "sign in" }, { status: 401 });
  if (user.role !== "head_coach" && user.coach_id == null) return Response.json([]);
  return Response.json(listUploads(user.role === "head_coach" ? undefined : user.coach_id!));
}

export async function POST(req: Request) {
  const user = await apiUser();
  if (!user) return Response.json({ error: "sign in" }, { status: 401 });
  const b = await req.json().catch(() => null);
  if (!b || typeof b !== "object") return Response.json({ error: "expected JSON" }, { status: 400 });

  // A coach uploads their own sessions; a head coach picks the coach (or names a new one).
  let coachId: number | null;
  if (user.role === "head_coach") {
    coachId = b.coach_id ? Number(b.coach_id) : b.coach_name?.trim() ? coachIdByName(b.coach_name.trim()) : null;
    if (coachId == null || !getCoach(coachId)) return Response.json({ error: "choose the coach who ran the session" }, { status: 400 });
  } else {
    coachId = user.coach_id;
    if (coachId == null) return Response.json({ error: "your account is not linked to a coach - ask your head coach" }, { status: 403 });
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(b.date ?? "")) return Response.json({ error: "date must be YYYY-MM-DD" }, { status: 400 });
  if (!SESSION_TYPES.some((t) => t.key === b.session_type)) return Response.json({ error: "choose squad or private" }, { status: 400 });
  if (!/\.(mp4|mov|m4v|avi|mkv)$/i.test(b.file_name ?? "")) return Response.json({ error: "upload an .mp4, .mov, .m4v, .avi or .mkv video" }, { status: 400 });
  const themes = (Array.isArray(b.themes) ? b.themes : []).filter((t: string) => THEMES.includes(t));

  const id = createUpload({
    createdBy: user.id, coachId, date: b.date, court: b.court?.trim() || null,
    sessionType: b.session_type as SessionType, themes, originalName: String(b.file_name),
    title: b.title?.trim() || `${b.session_type === "private" ? "Private lesson" : "Squad session"} - ${b.date}`,
  });
  return Response.json({ id }, { status: 201 });
}
