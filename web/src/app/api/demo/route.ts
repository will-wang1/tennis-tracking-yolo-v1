// Remove the demo sessions and demo coaches (scripts/seed-demo.mjs) once
// real sessions are in.

import { deleteDemoData } from "@/lib/db";

const TOKEN = process.env.IMPORT_TOKEN ?? "dev-import-token";

export async function DELETE(req: Request) {
  if (req.headers.get("authorization") !== `Bearer ${TOKEN}`) {
    return Response.json({ error: "bad or missing import token" }, { status: 401 });
  }
  deleteDemoData();
  return Response.json({ deleted: true });
}
