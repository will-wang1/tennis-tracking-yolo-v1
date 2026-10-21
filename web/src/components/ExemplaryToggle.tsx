"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";

export function ExemplaryToggle({ id, exemplary, compact }: { id: number; exemplary: boolean; compact?: boolean }) {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [, startTransition] = useTransition();
  async function toggle() {
    setBusy(true);
    await fetch(`/api/sessions/${id}/exemplary`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ exemplary: !exemplary }),
    });
    setBusy(false);
    startTransition(() => router.refresh());
  }
  const label = exemplary ? (compact ? "Exemplary ✓" : "Marked exemplary ✓") : (compact ? "Mark exemplary" : "Mark as exemplary");
  return (
    <button className={`btn${exemplary ? " on" : ""}`} onClick={toggle} disabled={busy}
      title={exemplary ? "This session sets the benchmark for its session type. Click to remove it." : "Use this session to set the benchmark for its session type."}>
      {label}
    </button>
  );
}
