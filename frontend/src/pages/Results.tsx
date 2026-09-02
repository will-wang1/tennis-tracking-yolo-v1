import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { JobResult } from "../api/types";
import ShotStatsTable from "../components/ShotStatsTable";
import ShotSpeedChart from "../components/ShotSpeedChart";
import BounceHeatmap from "../components/BounceHeatmap";
import StatTile from "../components/StatTile";

export default function Results() {
  const { jobId } = useParams<{ jobId: string }>();
  const [result, setResult] = useState<JobResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    api
      .getJobResult(jobId)
      .then(setResult)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load results"));
  }, [jobId]);

  if (error) return <p className="error-text">{error}</p>;
  if (!result) return <p>Loading results...</p>;
  if (result.status !== "done") {
    return (
      <div className="card">
        <p>
          This job is <span className={`badge ${result.status}`}>{result.status}</span>, not finished
          yet.
        </p>
        <Link to={`/jobs/${jobId}`}>View progress</Link>
      </div>
    );
  }

  const stats = result.stats;

  return (
    <div>
      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        {result.video_url ? (
          <video src={result.video_url} controls style={{ width: "100%", display: "block", background: "var(--tva-black)" }} />
        ) : (
          <div style={{ padding: "var(--card-padding)" }}>
            <h3 className="card-title" style={{ margin: 0 }}>
              Annotated video
            </h3>
            <p style={{ color: "var(--text-muted)" }}>No video available.</p>
          </div>
        )}
      </div>
      {result.video_url && (
        <p style={{ marginTop: -12, marginBottom: 20, fontSize: 13 }}>
          <a href={result.video_url} download>
            Download video
          </a>
        </p>
      )}

      {stats && (
        <>
          <div className="card">
            <h3 className="card-title">Shot speeds</h3>
            <ShotSpeedChart shots={stats.shot_speeds} />
            <div style={{ marginTop: 12 }}>
              <ShotStatsTable shots={stats.shot_speeds} />
            </div>
          </div>

          <div className="card">
            <h3 className="card-title">Match summary</h3>
            <div style={{ display: "flex", gap: 32 }}>
              <StatTile label="Rallies" value={stats.rally_count} />
              <StatTile label="Bounces" value={stats.total_bounces} />
              <StatTile label="Contacts" value={stats.total_contacts} />
            </div>
          </div>

          <div className="card">
            <h3 className="card-title">Bounce landing heatmap</h3>
            <BounceHeatmap locations={stats.bounce_locations} />
          </div>
        </>
      )}

      <p>
        <Link to="/">Back to videos</Link>
      </p>
    </div>
  );
}
