import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import storage
from app.auth import get_current_user
from app.celery_app import celery_app
from app.config import get_settings
from app.db import get_db
from app.deps import get_owned_video
from app.models import Calibration, Job, User
from app.schemas import JobCreate, JobOut, JobResultOut

router = APIRouter(tags=["jobs"])


@router.post("/videos/{video_id}/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED)
def create_job(
    video_id: str,
    payload: JobCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = get_owned_video(video_id, db, user.id)
    settings = get_settings()

    if payload.minimap and not settings.court_weights_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Minimap/court overlay isn't available - no court-keypoint model is configured on this server",
        )

    if payload.calibration_id is not None:
        calibration = db.get(Calibration, payload.calibration_id)
        if calibration is None or calibration.video_id != video.id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Calibration not found")

    job = Job(
        video_id=video.id,
        user_id=user.id,
        calibration_id=payload.calibration_id,
        options={
            "bounce": payload.bounce,
            "speed": payload.speed,
            "sidebar": payload.sidebar,
            "minimap": payload.minimap,
        },
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatched by task NAME rather than importing app.tasks.process_video_task
    # directly - that module imports src.pipeline, which pulls in the full ML
    # stack (torch/tensorflow/ultralytics/catboost). The API container is
    # deliberately lightweight (see backend/Dockerfile) and must never import
    # that; only the worker container needs it.
    #
    # task_id is pinned to the job id so cancelling only needs the job id -
    # no second identifier to store and keep in sync.
    celery_app.send_task("app.tasks.process_video_task", args=[job.id], task_id=job.id)
    return job


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(
    active: bool = False,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """This user's jobs, newest first. `active=true` narrows to the ones still
    working, which is what a progress indicator outside the job page needs."""
    query = db.query(Job).filter(Job.user_id == user.id)
    if active:
        query = query.filter(Job.status.in_(("queued", "running")))
    return query.order_by(Job.created_at.desc()).all()


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.post("/jobs/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Stop a queued or running job. Also the way out of a job left stuck in
    `running` because its worker died mid-run - there's nothing left to
    revoke there, but the row still needs moving to a terminal state."""
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    if job.status not in ("queued", "running"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job is already {job.status}",
        )

    # Best effort: tells a live worker to drop it (or kill it mid-run). If the
    # broker is unreachable or no worker is listening, the status change below
    # still stands, and the task itself re-checks for cancellation before it
    # starts doing any real work.
    try:
        celery_app.control.revoke(job.id, terminate=True, signal="SIGTERM")
    except Exception:  # noqa: BLE001 - a broker hiccup must not block cancelling
        pass

    job.status = "cancelled"
    job.finished_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(job)
    return job


@router.get("/jobs/{job_id}/result", response_model=JobResultOut)
def get_job_result(job_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(Job, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.status != "done":
        return JobResultOut(id=job.id, status=job.status)

    stats = None
    if job.stats_json_key:
        stats = json.loads(storage.download_bytes(job.stats_json_key))

    return JobResultOut(
        id=job.id,
        status=job.status,
        video_url=storage.presigned_url(job.output_video_key) if job.output_video_key else None,
        stats=stats,
    )
