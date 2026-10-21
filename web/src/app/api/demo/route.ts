// Remove the demo sessions and demo coaches (scripts/seed-demo.mjs) once
// real sessions are in.

import { deleteDemoData } from "@/lib/db";
import { apiUser, workerAuthorised } from "@/lib/auth";

export async function DELETE(req: Request) {
  if (!workerAuthorised(req) && (await apiUser())?.role !== "head_coach") {
    return Response.json({ error: "head coach or worker token only" }, { status: 401 });
  }
  deleteDemoData();
  return Response.json({ deleted: true });
}
