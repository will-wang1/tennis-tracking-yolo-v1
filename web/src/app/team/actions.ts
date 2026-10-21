"use server";

import { revalidatePath } from "next/cache";
import { requireHeadCoach } from "@/lib/auth";
import { coachIdByName } from "@/lib/db";
import { createUser, deleteUser, emailTaken, getUser, setPassword, temporaryPassword, type Role } from "@/lib/users";

export interface TeamState { error?: string; made?: { name: string; email: string; password: string; reset?: boolean } }

/** Add an account. It gets a temporary password, shown once to the head
 * coach to pass on, and must be changed at first sign-in. */
export async function addMember(_: TeamState, form: FormData): Promise<TeamState> {
  await requireHeadCoach();
  const name = String(form.get("name") ?? "").trim();
  const email = String(form.get("email") ?? "").trim();
  const role = (form.get("role") === "head_coach" ? "head_coach" : "coach") as Role;
  const link = String(form.get("coach") ?? "new");
  if (!name || !/^\S+@\S+\.\S+$/.test(email)) return { error: "Enter a name and an email." };
  if (emailTaken(email)) return { error: "That email already has an account." };
  // A coach account sees one coach's sessions: an existing coach (e.g. one
  // whose sessions were imported before they had an account) or a new one.
  const coachId = link === "none" ? null : link === "new" ? coachIdByName(name) : Number(link);
  if (role === "coach" && coachId == null) return { error: "A coach account needs to be linked to a coach." };
  const password = temporaryPassword();
  createUser({ email, name, password, role, coachId, mustChange: true });
  revalidatePath("/team");
  return { made: { name, email, password } };
}

export async function resetPassword(_: TeamState, form: FormData): Promise<TeamState> {
  await requireHeadCoach();
  const user = getUser(Number(form.get("id")));
  if (!user) return { error: "No such account." };
  const password = temporaryPassword();
  setPassword(user.id, password, true);
  revalidatePath("/team");
  return { made: { name: user.name, email: user.email, password, reset: true } };
}

export async function removeMember(form: FormData): Promise<void> {
  const me = await requireHeadCoach();
  const id = Number(form.get("id"));
  if (id === me.id) return; // never lock the club out of its own head coach account
  deleteUser(id);
  revalidatePath("/team");
}
