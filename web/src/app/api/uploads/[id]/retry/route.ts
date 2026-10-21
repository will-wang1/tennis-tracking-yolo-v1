import { apiUser, canSeeCoach } from "@/lib/auth";
import { getUpload, requeue } from "@/lib/uploads";

/** Put a failed (or stuck) upload back in the queue. */
export async function POST(_: Request, ctx: { params: Promise<{ id: string }> }) {
  const user = await apiUser();
  const up = getUpload(Number((await ctx.params).id));
  if (!user || !up || !canSeeCoach(user, up.coach_id)) return Response.json({ error: "not found" }, { status: 404 });
  if (up.status === "done" || up.status === "uploading") return Response.json({ error: `it is ${up.status}` }, { status: 409 });
  requeue(up.id);
  return Response.json({ id: up.id, status: "queued" });
}
