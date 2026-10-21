import { requireUser } from "@/lib/auth";
import { ROLE_LABEL } from "@/lib/users";
import { AuthForm } from "@/components/AuthForm";
import { changePassword } from "./actions";

export const dynamic = "force-dynamic";

export default async function AccountPage({ searchParams }: { searchParams: Promise<{ first?: string }> }) {
  const user = await requireUser({ allowTemporary: true });
  const first = (await searchParams).first || user.must_change_password;
  return (
    <div className="auth">
      <p className="eyebrow">{ROLE_LABEL[user.role]}{user.coach_name ? ` · ${user.coach_name}` : ""}</p>
      <h1>{first ? "Choose your password" : "Account"}</h1>
      <p className="lede">{user.name} · {user.email}</p>
      {first && <p className="banner info">You signed in with a temporary password. Choose your own to continue.</p>}
      <AuthForm action={changePassword} submit="Change password" fields={[
        { name: "current", label: first ? "Temporary password" : "Current password", type: "password", autoComplete: "current-password" },
        { name: "password", label: "New password", type: "password", autoComplete: "new-password", hint: "At least 10 characters." },
        { name: "again", label: "New password again", type: "password", autoComplete: "new-password" },
      ]} />
    </div>
  );
}
