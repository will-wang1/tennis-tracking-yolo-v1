from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models import Video


def get_owned_video(video_id: str, db: Session, user_id: str) -> Video:
    video = db.get(Video, video_id)
    if video is None or video.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video not found")
    return video
