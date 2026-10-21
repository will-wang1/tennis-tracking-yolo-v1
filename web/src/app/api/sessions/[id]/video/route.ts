// A session's video, with HTTP range support so the player can seek.
// Locally it streams from disk; in the cloud this becomes a signed URL to
// object storage.

import { getSession, resolveMedia } from "@/lib/db";
import { apiUser, canSeeCoach } from "@/lib/auth";
import { streamFile } from "@/lib/files";

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const user = await apiUser();
  if (!user) return new Response("sign in", { status: 401 });
  const session = getSession(Number((await ctx.params).id));
  if (!session || !canSeeCoach(user, session.coach_id)) return new Response("not found", { status: 404 });
  if (!session.video_path) return new Response("no video", { status: 404 });
  return streamFile(req, resolveMedia(session.video_path), "video/mp4");
}
