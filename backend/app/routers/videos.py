import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import get_settings
from app.db import get_db
from app import ffmpeg_utils, storage
from app.ffmpeg_utils import FFmpegError
from app.models import User, Video
from app.schemas import VideoOut

router = APIRouter(prefix="/videos", tags=["videos"])

_ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


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
    return db.query(Video).filter(Video.user_id == user.id).order_by(Video.created_at.desc()).all()


@router.get("/{video_id}", response_model=VideoOut)
def get_video(video_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    video = db.get(Video, video_id)
    if video is None or video.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return video
