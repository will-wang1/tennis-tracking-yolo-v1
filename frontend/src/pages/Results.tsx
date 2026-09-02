import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { JobResult } from "../api/types";
import ShotStatsTable from "../components/ShotStatsTable";
import ShotSpeedChart from "../components/ShotSpeedChart";
import BounceHeatmap from "../components/BounceHeatmap";

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
      <div className="card">
        <h2>Annotated video</h2>
        {result.video_url ? (
          <>
            <video src={result.video_url} controls style={{ width: "100%", borderRadius: 8 }} />
            <p style={{ marginTop: 10 }}>
              <a href={result.video_url} download>
                Download video
              </a>
            </p>
          </>
        ) : (
          <p>No video available.</p>
        )}
      </div>

      {stats && (
        <>
          <div className="card">
            <h2>Shot speeds</h2>
            <ShotSpeedChart shots={stats.shot_speeds} />
            <ShotStatsTable shots={stats.shot_speeds} />
          </div>

          <div className="card">
            <h2>Match summary</h2>
            <p>
              {stats.rally_count} rally/rallies, {stats.total_bounces} bounces, {stats.total_contacts}{" "}
              contacts
            </p>
          </div>

          <div className="card">
            <h2>Bounce landing heatmap</h2>
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
