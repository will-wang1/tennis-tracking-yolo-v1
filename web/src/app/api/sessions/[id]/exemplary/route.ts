// Mark or unmark a session as exemplary - it then feeds (or stops feeding)
// the benchmark for its session type.

import { getSession, setExemplary } from "@/lib/db";

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const id = Number((await ctx.params).id);
  if (!getSession(id)) return Response.json({ error: "no such session" }, { status: 404 });
  const body = await req.json().catch(() => ({}));
  setExemplary(id, Boolean(body.exemplary));
  return Response.json({ id, exemplary: Boolean(body.exemplary) });
}
