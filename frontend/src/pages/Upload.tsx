import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { JobOptions, PublicConfig, Video } from "../api/types";
import FeatureToggles from "../components/FeatureToggles";
import { DEFAULT_JOB_OPTIONS, PENDING_JOB_KEY } from "../constants";

const DEFAULT_OPTIONS: JobOptions = DEFAULT_JOB_OPTIONS;

function UploadIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
      <path
        d="M8 11V2M8 2L4.5 5.5M8 2l3.5 3.5M2.5 11v1.5A1.5 1.5 0 0 0 4 14h8a1.5 1.5 0 0 0 1.5-1.5V11"
        stroke="currentColor"
        strokeWidth="1.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
      <path d="M4 2.5v9l8-4.5-8-4.5Z" fill="currentColor" />
    </svg>
  );
}

export default function Upload() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [config, setConfig] = useState<PublicConfig>({ minimap_available: false, invite_required: false });
  const [uploading, setUploading] = useState(false);
  const [uploadFraction, setUploadFraction] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [options, setOptions] = useState<JobOptions>(DEFAULT_OPTIONS);
  const [pendingFileName, setPendingFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    api.listVideos().then(setVideos).catch(() => undefined);
    api.publicConfig().then(setConfig).catch(() => undefined);
  }, []);

  async function handleUpload() {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    setError(null);
    setUploading(true);
    setUploadFraction(0);
    try {
      const video = await api.uploadVideo(file, setUploadFraction);
      setVideos((prev) => [video, ...prev]);
      setSelectedVideoId(video.id);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setPendingFileName(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function goToCalibration(videoId: string) {
    sessionStorage.setItem(PENDING_JOB_KEY, JSON.stringify(options));
    navigate(`/videos/${videoId}/calibrate`);
  }

  async function runWithoutCalibration(videoId: string) {
    setError(null);
    try {
      const job = await api.createJob(videoId, options);
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't start the job");
    }
  }

  return (
    <div>
      <div className="card">
        <h3 className="card-title">Upload a match video</h3>
        <div className="dropzone" onClick={() => fileInputRef.current?.click()} style={{ cursor: "pointer" }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "50%",
              background: "var(--fill-accent-wash)",
              color: "var(--text-accent)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              margin: "0 auto 12px",
            }}
          >
            <UploadIcon />
          </div>
          <p style={{ margin: 0, fontSize: 14 }}>
            {pendingFileName || "Click to browse or drag & drop"}
          </p>
          <p style={{ margin: "4px 0 0", fontSize: 12, color: "var(--text-muted)" }}>MP4 · MOV · AVI — max 10 GB</p>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          disabled={uploading}
          style={{ display: "none" }}
          onChange={(e) => setPendingFileName(e.target.files?.[0]?.name ?? null)}
        />
        <div style={{ marginTop: 16, display: "flex", justifyContent: "center" }}>
          <button className="btn btn-primary" onClick={handleUpload} disabled={uploading || !pendingFileName}>
            {uploading ? `Uploading... ${(uploadFraction * 100).toFixed(0)}%` : "Choose file"}
          </button>
        </div>
        {uploading && (
          <div className="progress-track" style={{ marginTop: 12 }}>
            <div className="progress-fill" style={{ width: `${uploadFraction * 100}%` }} />
          </div>
        )}
        {error && <p className="error-text">{error}</p>}
      </div>

      <div className="card">
        <h3 className="card-title">Your videos</h3>
        {videos.length === 0 && (
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>No videos yet — upload one above to get started.</p>
        )}
        <ul className="video-list">
          {videos.map((video) => (
            <li key={video.id}>
              <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span
                  style={{
                    width: 28,
                    height: 28,
                    borderRadius: "var(--radius-sm)",
                    background: "var(--surface-inset)",
                    color: "var(--text-accent)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                  }}
                >
                  <PlayIcon />
                </span>
                <span>
                  <span className="filename">{video.filename}</span>
                  {video.duration_s && (
                    <span className="meta" style={{ marginLeft: 8 }}>
                      {Math.floor(video.duration_s / 60)}h {Math.round(video.duration_s % 60)}m
                    </span>
                  )}
                </span>
              </span>
              <button
                className="btn btn-secondary"
                onClick={() => setSelectedVideoId(video.id === selectedVideoId ? null : video.id)}
              >
                {selectedVideoId === video.id ? "Close" : "Analyze"}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {selectedVideoId && (
        <div className="card">
          <h3 className="card-title">Choose analysis options</h3>
          <FeatureToggles value={options} onChange={setOptions} minimapAvailable={config.minimap_available} />
          <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
            <button className="btn btn-primary" onClick={() => runWithoutCalibration(selectedVideoId)}>
              Run now
            </button>
            {options.speed && (
              <button className="btn btn-secondary" onClick={() => goToCalibration(selectedVideoId)}>
                Calibrate court first (for real km/h + landing heatmap)
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
