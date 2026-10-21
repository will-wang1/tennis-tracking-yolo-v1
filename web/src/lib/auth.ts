// Who is signed in, and what they may see. Pages and API routes call these;
// src/proxy.ts only sends browsers without a sign-in cookie to /login.
//
//   head coach - everything: every coach, every session, benchmarks,
//                marking exemplary, the team, uploads for any coach
//   coach      - their own sessions and coach page, the benchmark bands,
//                uploads for their own sessions

import { cache } from "react";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { userForToken, type User } from "./users";

export const COOKIE = "cr_login";

export const currentUser = cache(async (): Promise<User | null> => {
  return userForToken((await cookies()).get(COOKIE)?.value);
});

/** For pages: the signed-in user, or off to /login. A user still on a
 * temporary password is sent to change it first. */
export async function requireUser(opts: { allowTemporary?: boolean } = {}): Promise<User> {
  const user = await currentUser();
  if (!user) redirect("/login");
  if (user.must_change_password && !opts.allowTemporary) redirect("/account?first=1");
  return user;
}

export async function requireHeadCoach(): Promise<User> {
  const user = await requireUser();
  if (user.role !== "head_coach") redirect(user.coach_id ? `/coaches/${user.coach_id}` : "/sessions");
  return user;
}

export function isHead(user: User | null): boolean {
  return user?.role === "head_coach";
}

/** May this user see this coach's sessions? */
export function canSeeCoach(user: User, coachId: number): boolean {
  return user.role === "head_coach" || user.coach_id === coachId;
}

/** For API routes: the user, or null (the route answers 401). Accounts on
 * a temporary password are refused until they change it. */
export async function apiUser(): Promise<User | null> {
  const user = await currentUser();
  return user && !user.must_change_password ? user : null;
}

export function workerAuthorised(req: Request): boolean {
  const token = process.env.IMPORT_TOKEN ?? "dev-import-token";
  return req.headers.get("authorization") === `Bearer ${token}`;
}
