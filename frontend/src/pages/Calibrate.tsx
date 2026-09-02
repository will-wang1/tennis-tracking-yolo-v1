import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { CalibrationFrame, JobOptions } from "../api/types";
import CalibrationCanvas, { type CalibrationPoints } from "../components/CalibrationCanvas";
import { DEFAULT_JOB_OPTIONS, PENDING_JOB_KEY } from "../constants";

const DEFAULT_OPTIONS: JobOptions = DEFAULT_JOB_OPTIONS;

export default function Calibrate() {
  const { videoId } = useParams<{ videoId: string }>();
  const [frame, setFrame] = useState<CalibrationFrame | null>(null);
  const [courtType, setCourtType] = useState<"singles" | "doubles">("singles");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (!videoId) return;
    api
      .createCalibrationFrame(videoId)
      .then(setFrame)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Couldn't load a calibration frame"));
  }, [videoId]);

  async function handleComplete(points: CalibrationPoints) {
    if (!videoId) return;
    setSubmitting(true);
    setError(null);
    try {
      const calibration = await api.createCalibration(videoId, points, courtType);
      const pendingRaw = sessionStorage.getItem(PENDING_JOB_KEY);
      const options: JobOptions = pendingRaw ? JSON.parse(pendingRaw) : DEFAULT_OPTIONS;
      sessionStorage.removeItem(PENDING_JOB_KEY);
      const job = await api.createJob(videoId, options, calibration.id);
      navigate(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Couldn't save the calibration");
      setSubmitting(false);
    }
  }

  return (
    <div className="card">
      <h2>Calibrate the court</h2>
      <p>
        Click the four near-camera court corners on the frame below, in order. This builds a fixed
        mapping from pixels to real court coordinates (meters), used for real-world shot speed (km/h)
        and the bounce landing heatmap.
      </p>
      <div className="form-field" style={{ maxWidth: 220 }}>
        <label htmlFor="court-type">Court type</label>
        <select
          id="court-type"
          value={courtType}
          onChange={(e) => setCourtType(e.target.value as "singles" | "doubles")}
        >
          <option value="singles">Singles</option>
          <option value="doubles">Doubles</option>
        </select>
      </div>
      {error && <p className="error-text">{error}</p>}
      {submitting && <p>Saving calibration and starting the job...</p>}
      {frame && !submitting && (
        <CalibrationCanvas
          frameUrl={frame.frame_url}
          frameWidth={frame.width}
          frameHeight={frame.height}
          onComplete={handleComplete}
        />
      )}
      {!frame && !error && <p>Loading a frame from the video...</p>}
    </div>
  );
}
