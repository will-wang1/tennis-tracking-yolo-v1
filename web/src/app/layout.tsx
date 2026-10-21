import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import { currentUser } from "@/lib/auth";
import { logout } from "./login/actions";
import "./globals.css";

export const metadata: Metadata = {
  title: "Coaching Review",
  description: "How coaching sessions are run, measured from court video.",
};

export default async function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const user = await currentUser();
  const signedIn = user && !user.must_change_password;
  const links = !signedIn ? [] : user.role === "head_coach"
    ? [{ href: "/coaches", label: "Coaches" }, { href: "/sessions", label: "Sessions" }, { href: "/benchmarks", label: "Benchmarks" },
       { href: "/uploads", label: "Uploads" }, { href: "/team", label: "Team" }]
    : [...(user.coach_id ? [{ href: `/coaches/${user.coach_id}`, label: "My coaching" }] : []),
       { href: "/sessions", label: "My sessions" }, { href: "/benchmarks", label: "Benchmark" }, { href: "/uploads", label: "Uploads" }];
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        {/* App Router root layout: loads on every page. */}
        {/* eslint-disable-next-line @next/next/no-page-custom-font */}
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@500;600;700&family=IBM+Plex+Serif:wght@400;600&display=swap"
        />
      </head>
      <body>
        <header className="topbar">
          <div className="topbar-inner">
            <div className="brand">Coaching <span>Review</span></div>
            <Nav links={links} />
            {user && (
              <div className="user">
                <a href="/account">{user.name}</a>
                <form action={logout}><button className="link">Sign out</button></form>
              </div>
            )}
          </div>
        </header>
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
