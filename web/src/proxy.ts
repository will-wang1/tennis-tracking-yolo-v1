// Browsers without a sign-in cookie go to /login. Only the cookie's
// presence is checked here; pages and API routes check it properly
// (src/lib/auth.ts). API routes are left out: they answer 401 themselves,
// and the proxy would buffer upload bodies.

import { NextResponse, type NextRequest } from "next/server";

const OPEN = ["/login", "/setup"];

export function proxy(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (OPEN.some((p) => pathname === p || pathname.startsWith(p + "/"))) return NextResponse.next();
  if (!request.cookies.has("cr_login")) return NextResponse.redirect(new URL("/login", request.url));
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!api/|_next/|favicon.ico).*)"],
};
