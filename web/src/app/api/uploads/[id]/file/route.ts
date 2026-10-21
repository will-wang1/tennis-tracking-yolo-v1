// The clip itself: PUT by the browser (streamed straight to disk), GET by
// the processing worker.

import path from "node:path";
import { MEDIA_DIR } from "@/lib/db";
import { apiUser, canSeeCoach, workerAuthorised } from "@/lib/auth";
import { fileReceived, getUpload } from "@/lib/uploads";
import { saveBody, streamFile } from "@/lib/files";
import { resolveMedia } from "@/lib/db";

const MAX_BYTES = Number(process.env.MAX_UPLOAD_GB ?? 20) * 1e9;

export async function PUT(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const user = await apiUser();
  const up = getUpload(Number((await ctx.params).id));
  if (!user || !up || !canSeeCoach(user, up.coach_id)) return Response.json({ error: "not found" }, { status: 404 });
  if (up.status !== "uploading") return Response.json({ error: "this upload already has its video" }, { status: 409 });
  const ext = path.extname(up.original_name ?? ".mp4").toLowerCase() || ".mp4";
  const rel = `media/uploads/${up.id}${ext}`;
  try {
    const bytes = await saveBody(req, path.join(MEDIA_DIR, "uploads", `${up.id}${ext}`), MAX_BYTES);
    fileReceived(up.id, rel, bytes);
    return Response.json({ id: up.id, bytes, status: "queued" });
  } catch (e) {
    return Response.json({ error: `upload failed: ${(e as Error).message}` }, { status: 400 });
  }
}

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  if (!workerAuthorised(req)) return new Response("worker token only", { status: 401 });
  const up = getUpload(Number((await ctx.params).id));
  if (!up?.file_path) return new Response("not found", { status: 404 });
  return streamFile(req, resolveMedia(up.file_path), "application/octet-stream");
}
