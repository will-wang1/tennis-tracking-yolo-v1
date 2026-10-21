import { workerAuthorised } from "@/lib/auth";
import { markFailed } from "@/lib/uploads";

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  if (!workerAuthorised(req)) return Response.json({ error: "worker token only" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  markFailed(Number((await ctx.params).id), String(b.error ?? "unknown error"));
  return Response.json({ ok: true });
}
