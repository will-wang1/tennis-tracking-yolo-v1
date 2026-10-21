import type { NextConfig } from "next";
import path from "node:path";

const nextConfig: NextConfig = {
  // The app reads its data fresh on every request (sessions arrive at any
  // time), so the classic dynamic model is used rather than Cache Components.
  turbopack: { root: path.join(__dirname) },
};

export default nextConfig;
