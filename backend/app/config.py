"""Runtime configuration, read from environment variables (or a .env file in
dev). Model weight paths are configuration, not bundled files - see the
project README's "Deployment" notes: the ball detector and court-keypoint
checkpoints must be supplied on the server the worker runs on.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60 * 24 * 7
    # When set, /auth/register requires this value in the request body -
    # the simplest way to keep registration closed to people you've shared
    # the code with. Leave unset for open registration (the local-dev/test
    # default).
    invite_code: Optional[str] = None

    # Database
    database_url: str = "postgresql+psycopg2://tennis:tennis@localhost:5432/tennis"

    # Celery / Redis
    redis_url: str = "redis://localhost:6379/0"

    # S3-compatible object storage (works against real S3 or MinIO)
    s3_endpoint_url: Optional[str] = None  # e.g. http://minio:9000 for local dev; unset for AWS S3
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "tennis-tracking"
    s3_region: str = "us-east-1"
    # Serve presigned URLs pointed at this host instead of the internal
    # endpoint above - needed when the storage backend is only reachable
    # from inside the docker network but browsers need a public host.
    s3_public_endpoint_url: Optional[str] = None

    # ML pipeline - see src/pipeline.py::PipelineOptions. These must point at
    # real checkpoints on the machine the Celery worker runs on; court_weights
    # is optional (the minimap/show-court toggle is only offered to the
    # frontend when it's configured - see GET /config).
    detector_backend: str = "wasb"
    ball_weights_path: str = str(REPO_ROOT / "weights" / "ball_detector.pt")
    wasb_weights_path: str = str(REPO_ROOT / "weights" / "wasb_tennis_pretrained.pth.tar")
    tracknet_weights_path: str = str(REPO_ROOT / "weights" / "tracknet_pretrained.pt")
    bounce_weights_path: str = str(REPO_ROOT / "weights" / "bounce_catboost_pretrained.cbm")
    court_weights_path: Optional[str] = None
    pipeline_device: Optional[str] = None  # e.g. "cuda:0"; None lets each model pick its own default

    # Uploads
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
