"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

export function Nav({ links }: { links: { href: string; label: string }[] }) {
  const path = usePathname();
  // The longest matching link is the current one (/coaches/3 under /coaches).
  const on = links.map((l) => l.href).filter((h) => path === h || path.startsWith(h + "/")).sort((a, b) => b.length - a.length)[0];
  return (
    <nav className="nav">
      {links.map((l) => (
        <Link key={l.href} href={l.href} className={l.href === on ? "on" : ""}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
