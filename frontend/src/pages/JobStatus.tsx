import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api/client";
import type { Job } from "../api/types";

const POLL_INTERVAL_MS = 2000;

export default function JobStatus() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        const latest = await api.getJob(jobId!);
        if (cancelled) return;
        setJob(latest);
        if (latest.status === "done") {
          navigate(`/jobs/${jobId}/results`);
          return;
        }
        if (latest.status !== "failed") {
          timer = setTimeout(poll, POLL_INTERVAL_MS);
        }
      } catch {
        if (!cancelled) timer = setTimeout(poll, POLL_INTERVAL_MS);
      }
    }
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [jobId, navigate]);

  return (
    <div className="card" style={{ marginTop: 32 }}>
      <h3 className="card-title">Processing your video</h3>
      {!job && <p style={{ color: "var(--text-muted)" }}>Loading job status...</p>}
      {job && (
        <>
          <p style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Status</span>
            <span className={`badge ${job.status}`}>{job.status}</span>
          </p>
          <div className="progress-track">
            <div className="progress-fill" style={{ width: `${job.progress}%` }} />
          </div>
          <p style={{ fontFamily: "var(--font-mono)", fontSize: 13, color: "var(--text-accent)" }}>
            {job.progress}%
          </p>
          {job.status === "failed" && (
            <p className="error-text">{job.error_message ?? "Processing failed."}</p>
          )}
          {job.status === "queued" && (
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Waiting for a worker to pick this up...</p>
          )}
        </>
      )}
      <p style={{ marginTop: 16 }}>
        <Link to="/">Back to videos</Link>
      </p>
    </div>
  );
}
