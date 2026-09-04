from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    # Required only when the server has INVITE_CODE configured - see
    # app.config.Settings.invite_code.
    invite_code: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


class VideoOut(BaseModel):
    id: str
    filename: str
    duration_s: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class CalibrationFrameOut(BaseModel):
    frame_url: str
    width: int
    height: int


# Pixel points named per src.analysis.court_calibration.CORNER_ORDER -
# ("baseline_left", "baseline_right", "service_right", "service_left") -
# but accepted here as an explicit dict so the frontend doesn't have to
# guess an implicit order when it posts what the user clicked.
class CalibrationPoint(BaseModel):
    x: float
    y: float


class CalibrationCreate(BaseModel):
    baseline_left: CalibrationPoint
    baseline_right: CalibrationPoint
    service_left: CalibrationPoint
    service_right: CalibrationPoint
    court_type: str = Field(default="singles", pattern="^(singles|doubles)$")


class CalibrationOut(BaseModel):
    id: str
    video_id: str
    court_type: str
    pixel_points: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class JobCreate(BaseModel):
    bounce: bool = False
    speed: bool = False
    sidebar: bool = False
    minimap: bool = False
    calibration_id: Optional[str] = None


class JobOut(BaseModel):
    id: str
    video_id: str
    options: dict
    status: str
    progress: int
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    model_config = {"from_attributes": True}


class JobResultOut(BaseModel):
    id: str
    status: str
    video_url: Optional[str] = None
    stats: Optional[dict] = None


class PublicConfigOut(BaseModel):
    minimap_available: bool
    invite_required: bool
