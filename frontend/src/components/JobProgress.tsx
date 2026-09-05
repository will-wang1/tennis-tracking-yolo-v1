import { Link } from "react-router-dom";
import type { Job } from "../api/types";

export function isActive(job: Job | null | undefined): boolean {
  return job?.status === "queued" || job?.status === "running";
}

/** A job's live state, compact enough to sit under a row in the video list. */
export default function JobProgress({ job }: { job: Job }) {
  const active = isActive(job);

  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, flexWrap: "wrap" }}>
        <span className={`badge ${job.status}`}>{job.status}</span>
        {active && (
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-accent)" }}>
            {job.progress}%
          </span>
        )}
        {job.status === "queued" && (
          <span style={{ color: "var(--text-muted)" }}>waiting for a worker to pick this up</span>
        )}
        {job.status === "running" && (
          <span style={{ color: "var(--text-muted)" }}>analysing in the background</span>
        )}
        {job.status === "done" && <Link to={`/jobs/${job.id}/results`}>View results</Link>}
      </div>

      {active && (
        <div className="progress-track" style={{ marginTop: 6 }}>
          <div className="progress-fill" style={{ width: `${job.progress}%` }} />
        </div>
      )}

      {job.status === "failed" && job.error_message && (
        <p className="error-text" style={{ margin: "6px 0 0", fontSize: 12 }}>
          {job.error_message}
        </p>
      )}
    </div>
  );
}
