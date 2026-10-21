"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { COOKIE, currentUser } from "@/lib/auth";
import { passwordProblem, setPassword, startLogin, verifyLogin } from "@/lib/users";
import type { FormState } from "../login/actions";

export async function changePassword(_: FormState, form: FormData): Promise<FormState> {
  const user = await currentUser();
  if (!user) redirect("/login");
  const now = String(form.get("current") ?? "");
  const next = String(form.get("password") ?? "");
  if ("error" in verifyLogin(user.email, now)) return { error: "Your current password is wrong." };
  const weak = passwordProblem(next);
  if (weak) return { error: weak };
  if (next !== String(form.get("again") ?? "")) return { error: "The new passwords don't match." };
  setPassword(user.id, next, false); // signs out every browser...
  const { token, expires } = startLogin(user.id); // ...so sign this one back in
  (await cookies()).set(COOKIE, token, { httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", path: "/", expires });
  redirect("/");
}
