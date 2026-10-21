import Link from "next/link";
import { SESSION_TYPES, type SessionType } from "@/lib/db";

export function TypeTabs({ base, current, counts }: { base: string; current: SessionType; counts?: Partial<Record<SessionType, number>> }) {
  return (
    <div className="tabs">
      {SESSION_TYPES.map((t) => (
        <Link key={t.key} href={`${base}?type=${t.key}`} className={`btn${t.key === current ? " on" : ""}`}>
          {t.label}{counts?.[t.key] != null ? ` · ${counts[t.key]}` : ""}
        </Link>
      ))}
    </div>
  );
}

export function parseType(v: string | string[] | undefined): SessionType {
  return v === "private" ? "private" : "squad";
}
