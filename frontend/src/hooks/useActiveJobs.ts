import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Job } from "../api/types";

const POLL_INTERVAL_MS = 3000;

/** Jobs still queued or running, refreshed in the background so progress
 * stays visible from anywhere in the app - not just the job's own page. */
export function useActiveJobs(enabled = true): Job[] {
  const [jobs, setJobs] = useState<Job[]>([]);

  useEffect(() => {
    if (!enabled) {
      setJobs([]);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const active = await api.listJobs(true);
        if (!cancelled) setJobs(active);
      } catch {
        // Keep the last known state - a transient failure shouldn't make a
        // running job's indicator vanish.
      }
      if (!cancelled) timer = setTimeout(poll, POLL_INTERVAL_MS);
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [enabled]);

  return jobs;
}
