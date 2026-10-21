// Streaming files to and from disk for route handlers.

import fs from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

/** Serve a file, honouring Range requests so a video can seek. */
export function streamFile(req: Request, file: string, contentType: string): Response {
  if (!fs.existsSync(file)) return new Response("file missing", { status: 404 });
  const size = fs.statSync(file).size;
  const headers: Record<string, string> = { "Content-Type": contentType, "Accept-Ranges": "bytes" };
  const range = req.headers.get("range");
  if (range) {
    const [startText, endText] = range.replace("bytes=", "").split("-");
    const start = Number(startText);
    const end = endText ? Math.min(Number(endText), size - 1) : Math.min(start + 4 * 1024 * 1024, size - 1);
    if (!(start >= 0 && start <= end)) return new Response("bad range", { status: 416, headers: { "Content-Range": `bytes */${size}` } });
    return new Response(Readable.toWeb(fs.createReadStream(file, { start, end })) as ReadableStream, {
      status: 206,
      headers: { ...headers, "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": String(end - start + 1) },
    });
  }
  return new Response(Readable.toWeb(fs.createReadStream(file)) as ReadableStream, {
    headers: { ...headers, "Content-Length": String(size) },
  });
}

/** Write a request body to disk as it arrives - a session video can be
 * gigabytes, so it is never held in memory. Returns the bytes written. */
export async function saveBody(req: Request, file: string, maxBytes: number): Promise<number> {
  if (!req.body) throw new Error("empty body");
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const partial = file + ".part";
  let bytes = 0;
  const counted = Readable.fromWeb(req.body as never).on("data", (chunk: Buffer) => {
    bytes += chunk.length;
    if (bytes > maxBytes) counted.destroy(new Error(`larger than ${Math.round(maxBytes / 1e9)} GB`));
  });
  try {
    await pipeline(counted, fs.createWriteStream(partial));
  } catch (e) {
    fs.rmSync(partial, { force: true });
    throw e;
  }
  fs.renameSync(partial, file);
  return bytes;
}
