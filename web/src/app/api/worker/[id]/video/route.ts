// The worker sends back the annotated video, so the app can serve it
// whichever machine did the processing.

import path from "node:path";
import { MEDIA_DIR } from "@/lib/db";
import { workerAuthorised } from "@/lib/auth";
import { getUpload } from "@/lib/uploads";
import { saveBody } from "@/lib/files";

export async function PUT(req: Request, ctx: { params: Promise<{ id: string }> }) {
  if (!workerAuthorised(req)) return Response.json({ error: "worker token only" }, { status: 401 });
  const up = getUpload(Number((await ctx.params).id));
  if (!up) return Response.json({ error: "not found" }, { status: 404 });
  const name = `upload_${up.id}.mp4`;
  const bytes = await saveBody(req, path.join(MEDIA_DIR, "videos", name), 50e9);
  return Response.json({ video_path: `media/videos/${name}`, bytes });
}
