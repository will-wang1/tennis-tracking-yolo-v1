import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // The app reads its data fresh on every request (sessions arrive at any
  // time), so the classic dynamic model is used rather than Cache Components.
  turbopack: { root: path.join(__dirname) },
  // Forms post back to the address in the browser. Behind a proxy that
  // doesn't pass that host on, list it here (APP_HOSTS, comma-separated,
  // e.g. "coaching.tail1234.ts.net") or every form is rejected.
  experimental: {
    serverActions: { allowedOrigins: (process.env.APP_HOSTS ?? "").split(",").map((h) => h.trim()).filter(Boolean) },
  },
};

export default nextConfig;
