import { redirect } from "next/navigation";
import { countUsers } from "@/lib/users";
import { currentUser } from "@/lib/auth";
import { AuthForm } from "@/components/AuthForm";
import { login } from "./actions";

export const dynamic = "force-dynamic";

export default async function LoginPage() {
  if (countUsers() === 0) redirect("/setup");
  if (await currentUser()) redirect("/");
  return (
    <div className="auth">
      <h1>Sign in</h1>
      <p className="lede">Coaching Review - for the club&apos;s head coach and coaches.</p>
      <AuthForm action={login} submit="Sign in" fields={[
        { name: "email", label: "Email", type: "email", autoComplete: "username" },
        { name: "password", label: "Password", type: "password", autoComplete: "current-password" },
      ]} />
      <p className="muted small">No account? Ask your head coach to add you.</p>
    </div>
  );
}
