import { listSessions } from "@/lib/db";

export function DemoBanner() {
  const all = listSessions();
  const demo = all.filter((s) => s.is_demo).length;
  if (!demo) return null;
  return (
    <div className="banner demo">
      <b>Demo data.</b> {demo} of {all.length} sessions are generated variations of the one real session (Dingles), under made-up coaches, so the
      comparisons have something to show. They are marked <span className="chip demo">demo</span> and are removed with <span className="mono">npm run demo:clear</span>.
    </div>
  );
}
