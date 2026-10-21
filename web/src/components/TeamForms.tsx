"use client";

import { useActionState } from "react";
import { addMember, removeMember, resetPassword, type TeamState } from "@/app/team/actions";

function Made({ state }: { state: TeamState }) {
  if (state.error) return <p className="form-error" role="alert">{state.error}</p>;
  if (!state.made) return null;
  const m = state.made;
  return (
    <div className="banner info" role="status">
      {m.reset ? "New temporary password" : "Account made"} for <b>{m.name}</b> ({m.email}):{" "}
      <span className="secret">{m.password}</span>
      <br />Pass it on now - it is not shown again. They choose their own password when they first sign in.
    </div>
  );
}

export function AddMember({ coaches }: { coaches: { id: number; name: string; taken: boolean }[] }) {
  const [state, run, pending] = useActionState(addMember, {});
  return (
    <form action={run} className="form">
      <Made state={state} />
      <div className="row">
        <label><span>Name</span><input name="name" required autoComplete="off" /></label>
        <label><span>Email</span><input name="email" type="email" required autoComplete="off" /></label>
      </div>
      <div className="row">
        <label><span>Role</span>
          <select name="role" defaultValue="coach">
            <option value="coach">Coach - sees their own sessions</option>
            <option value="head_coach">Head coach - sees everything</option>
          </select>
        </label>
        <label><span>Their sessions are filed under</span>
          <select name="coach" defaultValue="new">
            <option value="new">A new coach with this name</option>
            {coaches.filter((c) => !c.taken).map((c) => <option key={c.id} value={c.id}>{c.name} (existing)</option>)}
            <option value="none">No coach (head coach who doesn&apos;t coach)</option>
          </select>
        </label>
      </div>
      <div><button className="btn primary" disabled={pending}>{pending ? "…" : "Add account"}</button></div>
    </form>
  );
}

export function ResetPassword({ id }: { id: number }) {
  const [state, run, pending] = useActionState(resetPassword, {});
  return (
    <form action={run} style={{ display: "inline" }}>
      <input type="hidden" name="id" value={id} />
      <button className="btn" disabled={pending}>Reset password</button>
      {(state.made || state.error) && <div style={{ marginTop: 8 }}><Made state={state} /></div>}
    </form>
  );
}

export function RemoveMember({ id, name }: { id: number; name: string }) {
  return (
    <form action={removeMember} style={{ display: "inline" }}
      onSubmit={(e) => { if (!confirm(`Remove ${name}'s account? Their sessions stay.`)) e.preventDefault(); }}>
      <input type="hidden" name="id" value={id} />
      <button className="btn danger" title="Their sessions stay; only the account goes">Remove</button>
    </form>
  );
}
