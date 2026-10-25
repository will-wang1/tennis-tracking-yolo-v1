"use server";

import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { COOKIE } from "@/lib/auth";
import {
  countUsers, createUser, emailTaken, endLogin, passwordProblem, startLogin, verifyLogin,
} from "@/lib/users";
import { coachIdByName } from "@/lib/db";
import { PRODUCTION, sameSecret, setupToken } from "@/lib/config";

export interface FormState { error?: string }

async function signIn(userId: number) {
  const { token, expires } = startLogin(userId);
  (await cookies()).set(COOKIE, token, {
    httpOnly: true, sameSite: "lax", secure: process.env.NODE_ENV === "production", path: "/", expires,
  });
}

export async function login(_: FormState, form: FormData): Promise<FormState> {
  const result = verifyLogin(String(form.get("email") ?? ""), String(form.get("password") ?? ""));
  if ("error" in result) return { error: result.error };
  await signIn(result.user.id);
  redirect(result.user.must_change_password ? "/account?first=1" : "/");
}

/** First run only: the head coach's account, while there are no accounts. */
export async function setup(_: FormState, form: FormData): Promise<FormState> {
  if (countUsers() > 0) redirect("/login");
  const code = setupToken();
  if (PRODUCTION && !code) return { error: "Setup is turned off: set SETUP_TOKEN in the server's settings file." };
  if (code && !sameSecret(String(form.get("setup_token") ?? "").trim(), code)) return { error: "That setup code is wrong." };
  const name = String(form.get("name") ?? "").trim();
  const email = String(form.get("email") ?? "").trim();
  const password = String(form.get("password") ?? "");
  if (!name || !/^\S+@\S+\.\S+$/.test(email)) return { error: "Enter your name and email." };
  const weak = passwordProblem(password);
  if (weak) return { error: weak };
  if (emailTaken(email)) return { error: "That email already has an account." };
  const coaches = form.get("also_coach") ? coachIdByName(name) : null;
  const id = createUser({ email, name, password, role: "head_coach", coachId: coaches, mustChange: false });
  await signIn(id);
  redirect("/");
}

export async function logout() {
  const store = await cookies();
  endLogin(store.get(COOKIE)?.value);
  store.delete(COOKIE);
  redirect("/login");
}
