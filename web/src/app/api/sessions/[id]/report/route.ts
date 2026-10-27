// A session's full report page - every chart and heatmap - drawn by the
// backend's own template (src/session/coach_report_template.html, the page
// scripts/render_coach_report.py writes) from the report stored with the
// session. ?embed=1 leaves out what the session page already shows (title,
// verdicts, AI commentary) for the frame on that page.

import fs from "node:fs";
import path from "node:path";
import { getSession, REPO_ROOT } from "@/lib/db";
import { apiUser, canSeeCoach } from "@/lib/auth";

const TEMPLATE = path.join(/* turbopackIgnore: true */ REPO_ROOT, "src", "session", "coach_report_template.html");

// "</" inside the <script> would end it early if any string held
// "</script>"; escaping the slash is inert in JSON (as render_coach_report.py does).
const embed = (data: unknown) => JSON.stringify(data ?? null).replaceAll("</", "<\\/");
const escapeHtml = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

const EMBED_STYLE = `<style>
  header.top, #verdict, #ai-section { display:none !important; }
  .wrap { padding-block:4px 16px; max-width:none; }
</style>`;

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const user = await apiUser();
  if (!user) return new Response("sign in", { status: 401 });
  const session = getSession(Number((await ctx.params).id));
  if (!session || !canSeeCoach(user, session.coach_id)) return new Response("not found", { status: 404 });

  const embedded = new URL(req.url).searchParams.get("embed") === "1";
  const html = fs.readFileSync(TEMPLATE, "utf8")
    .replace("__TITLE__", escapeHtml(session.title))
    .replace("__REPORT__", () => embed(session.report))
    .replace("__FEEDBACK__", () => embed(session.feedback));
  const head = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">${embedded ? EMBED_STYLE : ""}`;
  return new Response(head + html, {
    headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "private, no-store", "X-Frame-Options": "SAMEORIGIN" },
  });
}
