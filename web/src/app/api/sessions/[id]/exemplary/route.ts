// Mark or unmark a session as exemplary - it then feeds (or stops feeding)
// the benchmark for its session type.

import { getSession, setExemplary } from "@/lib/db";
import { apiUser } from "@/lib/auth";

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const user = await apiUser();
  if (user?.role !== "head_coach") return Response.json({ error: "only a head coach can set the benchmark" }, { status: 403 });
  const id = Number((await ctx.params).id);
  if (!getSession(id)) return Response.json({ error: "no such session" }, { status: 404 });
  const body = await req.json().catch(() => ({}));
  setExemplary(id, Boolean(body.exemplary));
  return Response.json({ id, exemplary: Boolean(body.exemplary) });
}
