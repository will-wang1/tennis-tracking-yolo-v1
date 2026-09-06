import { useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { Job } from "../api/types";

export function isActive(job: Job | null | undefined): boolean {
  return job?.status === "queued" || job?.status === "running";
}

/** A job's live state, compact enough to sit under a row in the video list. */
export default function JobProgress({
  job,
  onCancelled,
}: {
  job: Job;
  onCancelled?: () => void;
}) {
  const active = isActive(job);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  async function handleCancel() {
    setCancelling(true);
    setCancelError(null);
    try {
      await api.cancelJob(job.id);
      onCancelled?.();
    } catch (err) {
      // 409 means it reached a terminal state between render and click -
      // nothing went wrong, the list just needs refreshing.
      if (err instanceof ApiError && err.status === 409) {
        onCancelled?.();
      } else {
        setCancelError(err instanceof ApiError ? err.message : "Couldn't cancel this job");
      }
    } finally {
      setCancelling(false);
    }
  }

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
        {active && (
          <button
            className="btn-ghost"
            style={{ marginLeft: "auto", padding: "3px 10px", fontSize: 12 }}
            onClick={handleCancel}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling..." : "Cancel"}
          </button>
        )}
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
      {cancelError && (
        <p className="error-text" style={{ margin: "6px 0 0", fontSize: 12 }}>
          {cancelError}
        </p>
      )}
    </div>
  );
}
