"use client";

import { useActionState } from "react";
import type { FormState } from "@/app/login/actions";

export interface Field { name: string; label: string; type?: string; autoComplete?: string; hint?: string; checkbox?: boolean }

export function AuthForm({ action, fields, submit }: {
  action: (state: FormState, form: FormData) => Promise<FormState>;
  fields: Field[];
  submit: string;
}) {
  const [state, run, pending] = useActionState(action, {});
  return (
    <form action={run} className="form">
      {fields.map((f) => f.checkbox ? (
        <label key={f.name} className="check"><input type="checkbox" name={f.name} /> {f.label}</label>
      ) : (
        <label key={f.name}>
          <span>{f.label}</span>
          <input name={f.name} type={f.type ?? "text"} autoComplete={f.autoComplete} required />
          {f.hint && <small className="muted">{f.hint}</small>}
        </label>
      ))}
      {state.error && <p className="form-error" role="alert">{state.error}</p>}
      <button className="btn primary" disabled={pending}>{pending ? "…" : submit}</button>
    </form>
  );
}
