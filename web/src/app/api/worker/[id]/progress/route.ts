import { workerAuthorised } from "@/lib/auth";
import { reportProgress } from "@/lib/uploads";

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  if (!workerAuthorised(req)) return Response.json({ error: "worker token only" }, { status: 401 });
  const b = await req.json().catch(() => ({}));
  reportProgress(Number((await ctx.params).id), String(b.step ?? "").slice(0, 200), Number(b.steps_done ?? 0));
  return Response.json({ ok: true });
}
