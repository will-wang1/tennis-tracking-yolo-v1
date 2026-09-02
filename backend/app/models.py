import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    videos: Mapped[list["Video"]] = relationship(back_populates="user")


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    original_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="uploaded")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    user: Mapped["User"] = relationship(back_populates="videos")
    calibrations: Mapped[list["Calibration"]] = relationship(back_populates="video")
    jobs: Mapped[list["Job"]] = relationship(back_populates="video")


class Calibration(Base):
    __tablename__ = "calibrations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    court_type: Mapped[str] = mapped_column(String(16), default="singles")
    # pixel_points: {"baseline_left": [x, y], "baseline_right": [x, y],
    #                "service_left": [x, y], "service_right": [x, y]}
    pixel_points: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    video: Mapped["Video"] = relationship(back_populates="calibrations")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    video_id: Mapped[str] = mapped_column(String(36), ForeignKey("videos.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    calibration_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("calibrations.id"), nullable=True
    )
    # {"bounce": bool, "speed": bool, "sidebar": bool, "minimap": bool}
    options: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_video_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    stats_json_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    video: Mapped["Video"] = relationship(back_populates="jobs")
