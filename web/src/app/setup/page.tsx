import { redirect } from "next/navigation";
import { countUsers } from "@/lib/users";
import { AuthForm } from "@/components/AuthForm";
import { setup } from "../login/actions";
import { PRODUCTION, setupToken } from "@/lib/config";

export const dynamic = "force-dynamic";

export default function SetupPage() {
  if (countUsers() > 0) redirect("/login");
  return (
    <div className="auth">
      <p className="eyebrow">First run</p>
      <h1>Create the head coach account</h1>
      <p className="lede">This account sees every coach and session, sets the benchmarks and adds the coaches. It can only be made once, here.</p>
      <AuthForm action={setup} submit="Create account" fields={[
        ...(PRODUCTION || setupToken() ? [{ name: "setup_token", label: "Setup code", type: "password", autoComplete: "off",
          hint: "SETUP_TOKEN from the server's settings file." }] : []),
        { name: "name", label: "Your name", autoComplete: "name" },
        { name: "email", label: "Email", type: "email", autoComplete: "username" },
        { name: "password", label: "Password", type: "password", autoComplete: "new-password", hint: "At least 10 characters." },
        { name: "also_coach", label: "I also coach sessions myself", checkbox: true },
      ]} />
    </div>
  );
}
