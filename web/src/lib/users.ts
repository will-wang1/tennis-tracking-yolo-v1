// Accounts and sign-in. Passwords are hashed with scrypt; a signed-in
// browser holds a random token in an httpOnly cookie, and only its sha256
// is stored, so a copy of the database cannot be used to sign in.

import crypto from "node:crypto";
import { getDb } from "./db";

export type Role = "head_coach" | "coach";
export const ROLE_LABEL: Record<Role, string> = { head_coach: "Head coach", coach: "Coach" };

export interface User {
  id: number;
  email: string;
  name: string;
  role: Role;
  coach_id: number | null;
  coach_name: string | null;
  must_change_password: boolean;
}

const LOGIN_DAYS = 30;

type Raw = Record<string, unknown>;
function toUser(r: Raw): User {
  return {
    id: Number(r.id),
    email: String(r.email),
    name: String(r.name),
    role: r.role as Role,
    coach_id: r.coach_id == null ? null : Number(r.coach_id),
    coach_name: (r.coach_name as string) ?? null,
    must_change_password: Number(r.must_change_password) === 1,
  };
}

const USER_SELECT = `SELECT u.id, u.email, u.name, u.role, u.coach_id, c.name AS coach_name, u.must_change_password
  FROM users u LEFT JOIN coaches c ON c.id = u.coach_id`;

// ---- passwords -----------------------------------------------------------

export function hashPassword(password: string): string {
  const salt = crypto.randomBytes(16);
  const key = crypto.scryptSync(password, salt, 64);
  return `scrypt$${salt.toString("base64")}$${key.toString("base64")}`;
}

function checkPassword(password: string, stored: string): boolean {
  const [scheme, salt, key] = stored.split("$");
  if (scheme !== "scrypt" || !salt || !key) return false;
  const expected = Buffer.from(key, "base64");
  const got = crypto.scryptSync(password, Buffer.from(salt, "base64"), expected.length);
  return crypto.timingSafeEqual(got, expected);
}

export function passwordProblem(password: string): string | null {
  if (password.length < 10) return "Use at least 10 characters.";
  return null;
}

export function temporaryPassword(): string {
  // Readable: no 0/O, 1/l/I.
  const alphabet = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  return Array.from(crypto.randomBytes(14), (b) => alphabet[b % alphabet.length]).join("");
}

// ---- users -----------------------------------------------------------------

export function countUsers(): number {
  return Number((getDb().prepare("SELECT COUNT(*) AS n FROM users").get() as { n: number }).n);
}

export function listUsers(): User[] {
  return (getDb().prepare(`${USER_SELECT} ORDER BY u.role DESC, u.name`).all() as Raw[]).map(toUser);
}

export function getUser(id: number): User | null {
  const r = getDb().prepare(`${USER_SELECT} WHERE u.id = ?`).get(id) as Raw | undefined;
  return r ? toUser(r) : null;
}

export function createUser(u: { email: string; name: string; password: string; role: Role; coachId: number | null; mustChange: boolean }): number {
  const res = getDb().prepare(
    "INSERT INTO users (email, name, password_hash, role, coach_id, must_change_password) VALUES (?, ?, ?, ?, ?, ?)",
  ).run(u.email.trim(), u.name.trim(), hashPassword(u.password), u.role, u.coachId, u.mustChange ? 1 : 0);
  return Number(res.lastInsertRowid);
}

export function setPassword(userId: number, password: string, mustChange: boolean): void {
  getDb().prepare("UPDATE users SET password_hash = ?, must_change_password = ? WHERE id = ?")
    .run(hashPassword(password), mustChange ? 1 : 0, userId);
  // Every other signed-in browser is signed out when the password changes.
  getDb().prepare("DELETE FROM logins WHERE user_id = ?").run(userId);
}

export function deleteUser(userId: number): void {
  getDb().prepare("DELETE FROM users WHERE id = ?").run(userId);
}

export function emailTaken(email: string): boolean {
  return Boolean(getDb().prepare("SELECT 1 FROM users WHERE email = ?").get(email.trim()));
}

// ---- sign-in -------------------------------------------------------------

const failures = new Map<string, { n: number; until: number }>();

/** The user for this email and password, or why not. Five wrong passwords
 * lock the email for 15 minutes. */
export function verifyLogin(email: string, password: string): { user: User } | { error: string } {
  const key = email.trim().toLowerCase();
  const f = failures.get(key);
  if (f && f.n >= 5 && Date.now() < f.until) return { error: "Too many attempts. Try again in 15 minutes." };
  const r = getDb().prepare("SELECT id, password_hash FROM users WHERE email = ?").get(key) as { id: number; password_hash: string } | undefined;
  if (!r || !checkPassword(password, r.password_hash)) {
    failures.set(key, { n: (f && Date.now() < f.until ? f.n : 0) + 1, until: Date.now() + 15 * 60_000 });
    return { error: "Wrong email or password." };
  }
  failures.delete(key);
  return { user: getUser(Number(r.id))! };
}

const sha = (t: string) => crypto.createHash("sha256").update(t).digest("hex");

export function startLogin(userId: number): { token: string; expires: Date } {
  const token = crypto.randomBytes(32).toString("base64url");
  const expires = new Date(Date.now() + LOGIN_DAYS * 86_400_000);
  getDb().prepare("DELETE FROM logins WHERE expires_at < ?").run(new Date().toISOString());
  getDb().prepare("INSERT INTO logins (token_hash, user_id, expires_at) VALUES (?, ?, ?)").run(sha(token), userId, expires.toISOString());
  return { token, expires };
}

export function userForToken(token: string | undefined): User | null {
  if (!token) return null;
  const r = getDb().prepare("SELECT user_id, expires_at FROM logins WHERE token_hash = ?").get(sha(token)) as { user_id: number; expires_at: string } | undefined;
  if (!r || new Date(r.expires_at) < new Date()) return null;
  return getUser(Number(r.user_id));
}

export function endLogin(token: string | undefined): void {
  if (token) getDb().prepare("DELETE FROM logins WHERE token_hash = ?").run(sha(token));
}
