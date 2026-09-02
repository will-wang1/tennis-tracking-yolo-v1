"""4-point court calibration driven from the browser.

Reuses `src.analysis.court_calibration.CourtCalibration` directly - the same
class `scripts/calibrate_court.py` and the render pipeline
(`src/pipeline.py`) already use - so a calibration built here produces a
byte-identical homography to the existing CLI flow. The only new part is
where the 4 pixel points come from: a user clicking a frame in the browser
instead of reading pixel coordinates off an image by eye.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import ffmpeg_utils, storage
from app.auth import get_current_user
from app.db import get_db
from app.deps import get_owned_video
from app.ffmpeg_utils import FFmpegError
from app.models import Calibration, User
from app.schemas import CalibrationCreate, CalibrationFrameOut, CalibrationOut

from src.analysis.court_calibration import (
    CORNER_ORDER,
    DOUBLES_COURT_REFERENCE_POINTS,
    SINGLES_COURT_REFERENCE_POINTS,
    CourtCalibration,
)

router = APIRouter(prefix="/videos", tags=["calibration"])


@router.post("/{video_id}/calibration-frame", response_model=CalibrationFrameOut)
def create_calibration_frame(
    video_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    video = get_owned_video(video_id, db, user.id)

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = str(Path(tmp_dir) / "source")
        frame_path = str(Path(tmp_dir) / "frame.jpg")
        storage.download_file(video.original_key, video_path)
        try:
            ffmpeg_utils.extract_frame(video_path, frame_path)
        except FFmpegError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

        key = f"videos/{user.id}/{video.id}/calibration_frame.jpg"
        storage.upload_file(frame_path, key, content_type="image/jpeg")

    return CalibrationFrameOut(
        frame_url=storage.presigned_url(key),
        width=video.width or 0,
        height=video.height or 0,
    )


@router.post("/{video_id}/calibration", response_model=CalibrationOut, status_code=status.HTTP_201_CREATED)
def create_calibration(
    video_id: str,
    payload: CalibrationCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    video = get_owned_video(video_id, db, user.id)

    pixel_points_by_name = {
        "baseline_left": (payload.baseline_left.x, payload.baseline_left.y),
        "baseline_right": (payload.baseline_right.x, payload.baseline_right.y),
        "service_left": (payload.service_left.x, payload.service_left.y),
        "service_right": (payload.service_right.x, payload.service_right.y),
    }
    world_reference = (
        DOUBLES_COURT_REFERENCE_POINTS if payload.court_type == "doubles" else SINGLES_COURT_REFERENCE_POINTS
    )
    pixel_points = [pixel_points_by_name[name] for name in CORNER_ORDER]
    world_points = [world_reference[name] for name in CORNER_ORDER]

    calibration = CourtCalibration.from_points(pixel_points, world_points)

    key = f"videos/{user.id}/{video.id}/calibration_{video.id}.json"
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = str(Path(tmp_dir) / "calibration.json")
        calibration.save(local_path)
        storage.upload_file(local_path, key, content_type="application/json")

    row = Calibration(
        video_id=video.id,
        s3_key=key,
        court_type=payload.court_type,
        pixel_points=pixel_points_by_name,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
