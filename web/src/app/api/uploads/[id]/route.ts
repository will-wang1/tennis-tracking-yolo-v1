import fs from "node:fs";
import { resolveMedia } from "@/lib/db";
import { apiUser, canSeeCoach } from "@/lib/auth";
import { deleteUpload, getUpload } from "@/lib/uploads";

/** Remove an upload that has not become a session (or failed). The clip
 * goes too; a finished session stays. */
export async function DELETE(_: Request, ctx: { params: Promise<{ id: string }> }) {
  const user = await apiUser();
  const up = getUpload(Number((await ctx.params).id));
  if (!user || !up || !canSeeCoach(user, up.coach_id)) return Response.json({ error: "not found" }, { status: 404 });
  if (up.status === "processing") return Response.json({ error: "it is being processed - wait for it to finish or fail" }, { status: 409 });
  if (up.file_path) fs.rmSync(resolveMedia(up.file_path), { force: true });
  deleteUpload(up.id);
  return Response.json({ deleted: true });
}
