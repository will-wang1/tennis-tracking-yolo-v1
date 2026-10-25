// Runs once as the server starts: refuse to serve with placeholder secrets.

export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  (await import("./lib/config")).exitOnBadConfig();
}
