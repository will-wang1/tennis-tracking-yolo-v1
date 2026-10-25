// Secrets the server needs. On a laptop (next dev) they have harmless
// defaults; on the server (NODE_ENV=production) they must be set for real,
// and src/instrumentation.ts refuses to start until they are.

import { createHash, timingSafeEqual } from "node:crypto";

export const PRODUCTION = process.env.NODE_ENV === "production";
const DEV_IMPORT_TOKEN = "dev-import-token";
const MIN_SECRET = 32;

/** The worker's shared secret (Authorization: Bearer ...), or null when the
 * server has none it can trust - then no worker call is accepted. */
export function importToken(): string | null {
  const token = process.env.IMPORT_TOKEN?.trim();
  if (!PRODUCTION) return token || DEV_IMPORT_TOKEN;
  return token && token !== DEV_IMPORT_TOKEN && token.length >= MIN_SECRET ? token : null;
}

/** The code that first-run setup asks for. Required on the server, so
 * whoever finds the site first can't make themselves head coach; optional
 * on a laptop. */
export function setupToken(): string | null {
  return process.env.SETUP_TOKEN?.trim() || null;
}

/** Compare a secret without leaking, through timing, how much matched. */
export function sameSecret(given: string, expected: string): boolean {
  const digest = (s: string) => createHash("sha256").update(s).digest();
  return timingSafeEqual(digest(given), digest(expected));
}

export function configProblems(): string[] {
  if (!PRODUCTION) return [];
  const problems: string[] = [];
  if (!importToken()) problems.push(`IMPORT_TOKEN must be a random value of at least ${MIN_SECRET} characters (not the dev default)`);
  if ((setupToken()?.length ?? MIN_SECRET) < 16) problems.push("SETUP_TOKEN, if set, must be at least 16 characters");
  return problems;
}

/** Exits rather than throwing - a throw leaves every page an error 500 -
 * so the service shows as failed, with the reason in its log. */
export function exitOnBadConfig(): void {
  const problems = configProblems();
  if (!problems.length) return;
  console.error(`Not starting - fix the server's settings file (deploy/coaching.env):\n  ${problems.join("\n  ")}`);
  process.exit(1);
}
