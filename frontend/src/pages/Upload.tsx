import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { JobOptions, PublicConfig, Video } from "../api/types";
import FeatureToggles from "../components/FeatureToggles";
import { DEFAULT_JOB_OPTIONS, PENDING_JOB_KEY } from "../constants";

const DEFAULT_OPTIONS: JobOptions = DEFAULT_JOB_OPTIONS;

export default function Upload() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [config, setConfig] = useState<PublicConfig>({ minimap_available: false });
  const [uploading, setUploading] = useState(false);
  const [uploadFraction, setUploadFraction] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [selectedVideoId, setSelectedVideoId] = useState<string | null>(null);
  const [options, setOptions] = useState<JobOptions>(DEFAULT_OPTIONS);
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
        <h2>Upload a match video</h2>
        <input ref={fileInputRef} type="file" accept="video/*" disabled={uploading} />
        <div style={{ marginTop: 12 }}>
          <button className="primary" onClick={handleUpload} disabled={uploading}>
            {uploading ? `Uploading... ${(uploadFraction * 100).toFixed(0)}%` : "Upload"}
          </button>
        </div>
        {uploading && (
          <div className="progress-track" style={{ marginTop: 10 }}>
            <div className="progress-fill" style={{ width: `${uploadFraction * 100}%` }} />
          </div>
        )}
        {error && <p className="error-text">{error}</p>}
      </div>

      <div className="card">
        <h2>Your videos</h2>
        {videos.length === 0 && <p>No videos yet - upload one above to get started.</p>}
        <ul className="video-list">
          {videos.map((video) => (
            <li key={video.id}>
              <span>
                {video.filename}
                {video.duration_s ? ` — ${video.duration_s.toFixed(1)}s` : ""}
              </span>
              <button
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
          <h2>Choose analysis options</h2>
          <FeatureToggles value={options} onChange={setOptions} minimapAvailable={config.minimap_available} />
          <div style={{ display: "flex", gap: 12, marginTop: 16 }}>
            <button className="primary" onClick={() => runWithoutCalibration(selectedVideoId)}>
              Run now
            </button>
            {options.speed && (
              <button onClick={() => goToCalibration(selectedVideoId)}>
                Calibrate court first (for real km/h + landing heatmap)
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
