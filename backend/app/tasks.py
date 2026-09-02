"""Celery task that actually runs the tracking/analysis pipeline.

Runs in the worker process/container, which (unlike the API container) has
the full ML `requirements.txt` installed plus real model weights on disk -
see app.config.Settings for the paths, supplied at deploy time.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.celery_app import celery_app
from app.config import get_settings
from app.db import SessionLocal
from app.ffmpeg_utils import FFmpegError, transcode_to_h264
from app.models import Calibration, Job, Video
from app.storage import download_file, ensure_bucket, upload_file

from src.pipeline import PipelineOptions, run_pipeline

# The worker doesn't depend on the API container starting first, so it
# ensures the bucket exists itself rather than racing app.main's lifespan.
ensure_bucket()


def _set_progress(db, job: Job, value: int) -> None:
    value = max(0, min(100, value))
    if value != job.progress:
        job.progress = value
        db.commit()


@celery_app.task(name="app.tasks.process_video_task", bind=True)
def process_video_task(self, job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            return

        job.status = "running"
        job.started_at = datetime.now(timezone.utc)
        job.progress = 0
        db.commit()

        video = db.get(Video, job.video_id)
        calibration = db.get(Calibration, job.calibration_id) if job.calibration_id else None

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            source_path = tmp / f"source{Path(video.original_key).suffix}"
            raw_output_path = tmp / "raw_output.mp4"
            final_output_path = tmp / "output.mp4"
            stats_path = tmp / "stats.json"
            calibration_path = tmp / "calibration.json" if calibration else None

            download_file(video.original_key, str(source_path))
            if calibration is not None:
                download_file(calibration.s3_key, str(calibration_path))

            settings = get_settings()
            minimap = bool(job.options.get("minimap"))
            options = PipelineOptions(
                input=str(source_path),
                output=str(raw_output_path),
                detector_backend=settings.detector_backend,
                weights=settings.ball_weights_path,
                wasb_weights=settings.wasb_weights_path,
                tracknet_weights=settings.tracknet_weights_path,
                bounce_weights=settings.bounce_weights_path,
                device=settings.pipeline_device,
                bounce=bool(job.options.get("bounce")),
                speed=bool(job.options.get("speed")),
                sidebar=bool(job.options.get("sidebar")),
                minimap=minimap,
                show_court=minimap,
                court_weights=settings.court_weights_path or "",
                calibration=str(calibration_path) if calibration_path else None,
                stats=str(stats_path),
            )

            def progress_cb(stage: str, done: int, total: int) -> None:
                # Two frame passes (detect, render) map onto one 0-100 bar.
                fraction = done / total if total else 0.0
                base = 0 if stage == "detect" else 50
                _set_progress(db, job, base + int(fraction * 50))

            result = run_pipeline(options, progress_cb=progress_cb)

            try:
                transcode_to_h264(str(raw_output_path), str(final_output_path))
            except FFmpegError as exc:
                raise RuntimeError(f"Post-processing failed: {exc}") from exc

            stats_dict = (
                result.match_stats.to_dict()
                if result.match_stats is not None
                else {
                    "rally_count": 0,
                    "rallies": [],
                    "total_bounces": len(result.bounces),
                    "total_contacts": 0,
                    "total_unattributed": 0,
                    "shot_speeds": [
                        {
                            "start_frame": s.start_frame,
                            "end_frame": s.end_frame,
                            "peak_speed": round(s.peak_speed, 1),
                            "unit": s.unit,
                            "method": s.method,
                        }
                        for s in result.shots
                    ],
                    "bounce_locations": [
                        [round(b.world_x, 2), round(b.world_y, 2)]
                        for b in result.bounces
                        if b.world_x is not None and b.world_y is not None
                    ],
                    "near_shot_counts": None,
                    "far_shot_counts": None,
                }
            )
            stats_path.write_text(json.dumps(stats_dict, indent=2))

            output_key = f"videos/{job.user_id}/{job.video_id}/jobs/{job.id}/output.mp4"
            stats_key = f"videos/{job.user_id}/{job.video_id}/jobs/{job.id}/stats.json"
            upload_file(str(final_output_path), output_key, content_type="video/mp4")
            upload_file(str(stats_path), stats_key, content_type="application/json")

        job.output_video_key = output_key
        job.stats_json_key = stats_key
        job.status = "done"
        job.progress = 100
        job.finished_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as exc:  # noqa: BLE001 - report every failure to the Job row
        job = db.get(Job, job_id)
        if job is not None:
            job.status = "failed"
            job.error_message = str(exc)[:4000]
            job.finished_at = datetime.now(timezone.utc)
            db.commit()
        raise
    finally:
        db.close()
