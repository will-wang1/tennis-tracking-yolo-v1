import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import get_settings
from app.db import get_db
from app import ffmpeg_utils, storage
from app.ffmpeg_utils import FFmpegError
from app.models import Job, User, Video
from app.schemas import JobOut, VideoOut

router = APIRouter(prefix="/videos", tags=["videos"])

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def _latest_jobs_by_video(db: Session, video_ids: list[str]) -> dict[str, Job]:
    """Newest job per video, in one query rather than one per video."""
    if not video_ids:
        return {}
    jobs = (
        db.query(Job)
        .filter(Job.video_id.in_(video_ids))
        .order_by(Job.created_at.desc())
        .all()
    )
    latest: dict[str, Job] = {}
    for job in jobs:
        latest.setdefault(job.video_id, job)
    return latest


def _with_latest_job(video: Video, job: Job | None) -> VideoOut:
    out = VideoOut.model_validate(video)
    out.latest_job = JobOut.model_validate(job) if job is not None else None
    return out


@router.post("", response_model=VideoOut, status_code=status.HTTP_201_CREATED)
def upload_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(_ALLOWED_SUFFIXES)}",
        )

    video = Video(user_id=user.id, filename=file.filename or "upload.mp4", original_key="")
    db.add(video)
    db.flush()  # assigns video.id without committing yet

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        total = 0
        while chunk := file.file.read(8 * 1024 * 1024):
            total += len(chunk)
            if total > settings.max_upload_bytes:
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds {settings.max_upload_bytes} byte limit",
                )
            tmp.write(chunk)
        tmp.flush()

        try:
            metadata = ffmpeg_utils.probe_video(tmp.name)
        except FFmpegError as exc:
            db.rollback()
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        key = f"videos/{user.id}/{video.id}/original{suffix}"
        storage.upload_file(tmp.name, key, content_type="video/mp4")

    video.original_key = key
    video.duration_s = metadata["duration_s"]
    video.fps = metadata["fps"]
    video.width = metadata["width"]
    video.height = metadata["height"]
    video.status = "uploaded"
    db.commit()
    db.refresh(video)
    return video


@router.get("", response_model=list[VideoOut])
def list_videos(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    videos = db.query(Video).filter(Video.user_id == user.id).order_by(Video.created_at.desc()).all()
    latest = _latest_jobs_by_video(db, [video.id for video in videos])
    return [_with_latest_job(video, latest.get(video.id)) for video in videos]


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    video = db.get(Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return _with_latest_job(video, _latest_jobs_by_video(db, [video.id]).get(video.id))
