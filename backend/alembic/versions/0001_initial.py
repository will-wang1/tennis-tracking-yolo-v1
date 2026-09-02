"""Initial schema: users, videos, calibrations, jobs.

Revision ID: 0001
Revises:
Create Date: 2026-09-02

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "videos",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("original_key", sa.String(1024), nullable=False),
        sa.Column("duration_s", sa.Float, nullable=True),
        sa.Column("fps", sa.Float, nullable=True),
        sa.Column("width", sa.Integer, nullable=True),
        sa.Column("height", sa.Integer, nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_videos_user_id", "videos", ["user_id"])

    op.create_table(
        "calibrations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("s3_key", sa.String(1024), nullable=False),
        sa.Column("court_type", sa.String(16), nullable=False),
        sa.Column("pixel_points", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_calibrations_video_id", "calibrations", ["video_id"])

    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("video_id", sa.String(36), sa.ForeignKey("videos.id"), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("calibration_id", sa.String(36), sa.ForeignKey("calibrations.id"), nullable=True),
        sa.Column("options", sa.JSON, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress", sa.Integer, nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("output_video_key", sa.String(1024), nullable=True),
        sa.Column("stats_json_key", sa.String(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_jobs_video_id", "jobs", ["video_id"])
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("calibrations")
    op.drop_table("videos")
    op.drop_table("users")
