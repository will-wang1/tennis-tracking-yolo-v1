// The processing worker (scripts/worker.py) asks for the next clip.

import { workerAuthorised } from "@/lib/auth";
import { claimNext } from "@/lib/uploads";

export async function POST(req: Request) {
  if (!workerAuthorised(req)) return Response.json({ error: "worker token only" }, { status: 401 });
  const body = await req.json().catch(() => ({}));
  const job = claimNext(String(body.worker ?? "worker").slice(0, 80));
  if (!job) return new Response(null, { status: 204 });
  return Response.json(job);
}
