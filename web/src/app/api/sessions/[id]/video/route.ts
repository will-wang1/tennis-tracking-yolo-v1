// A session's video, with HTTP range support so the player can seek.
// Locally it streams from the path recorded at import; in the cloud this
// becomes a signed URL to object storage.

import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { getSession, REPO_ROOT } from "@/lib/db";

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const session = getSession(Number((await ctx.params).id));
  if (!session?.video_path) return new Response("no video", { status: 404 });
  const file = path.isAbsolute(session.video_path) ? session.video_path : path.join(REPO_ROOT, session.video_path);
  if (!fs.existsSync(file)) return new Response("video file missing", { status: 404 });
  const size = fs.statSync(file).size;
  const range = req.headers.get("range");
  const headers: Record<string, string> = { "Content-Type": "video/mp4", "Accept-Ranges": "bytes" };
  if (range) {
    const [startText, endText] = range.replace("bytes=", "").split("-");
    const start = Number(startText);
    const end = endText ? Number(endText) : Math.min(start + 4 * 1024 * 1024, size - 1);
    const stream = fs.createReadStream(file, { start, end });
    return new Response(Readable.toWeb(stream) as ReadableStream, {
      status: 206,
      headers: { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": String(end - start + 1) },
    });
  }
  return new Response(Readable.toWeb(fs.createReadStream(file)) as ReadableStream, {
    headers: { ...headers, "Content-Length": String(size) },
  });
}
