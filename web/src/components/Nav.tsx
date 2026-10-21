"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/coaches", label: "Coaches" },
  { href: "/sessions", label: "Sessions" },
  { href: "/benchmarks", label: "Benchmarks" },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="nav">
      {LINKS.map((l) => (
        <Link key={l.href} href={l.href} className={path.startsWith(l.href) ? "on" : ""}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
