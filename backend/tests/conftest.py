"""Test setup: SQLite instead of Postgres, a local-disk fake instead of S3,
faked ffmpeg (no real binary/video needed), and Celery dispatch captured
instead of actually publishing to Redis. None of this touches src/pipeline
or the ML stack - see app/routers/jobs.py's celery_app.send_task(name=...)
dispatch, which is what keeps the API process/tests free of that entirely.
"""

import os
import tempfile
from pathlib import Path

import pytest

_TMP_DB_DIR = tempfile.mkdtemp(prefix="tennis_tracking_test_db_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB_DIR}/test.db"
os.environ["JWT_SECRET"] = "test-secret"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["S3_BUCKET"] = "test-bucket"
os.environ["COURT_WEIGHTS_PATH"] = ""

from app import ffmpeg_utils, storage  # noqa: E402
from app.celery_app import celery_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_schema():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def client(tmp_path, monkeypatch):
    store_dir = tmp_path / "store"
    store_dir.mkdir()

    def fake_upload_file(local_path, key, content_type=None):
        dest = store_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local_path).read_bytes())

    def fake_download_file(key, local_path):
        Path(local_path).write_bytes((store_dir / key).read_bytes())

    def fake_upload_bytes(data, key, content_type=None):
        dest = store_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def fake_download_bytes(key):
        return (store_dir / key).read_bytes()

    def fake_presigned_url(key, expires_seconds=3600):
        return f"http://fake-storage.local/{key}"

    monkeypatch.setattr(storage, "upload_file", fake_upload_file)
    monkeypatch.setattr(storage, "download_file", fake_download_file)
    monkeypatch.setattr(storage, "upload_bytes", fake_upload_bytes)
    monkeypatch.setattr(storage, "download_bytes", fake_download_bytes)
    monkeypatch.setattr(storage, "presigned_url", fake_presigned_url)
    monkeypatch.setattr(storage, "ensure_bucket", lambda: None)

    def fake_probe_video(path):
        return {"duration_s": 12.3, "fps": 30.0, "width": 1920, "height": 1080}

    def fake_extract_frame(input_path, output_path, at_seconds=0.5):
        Path(output_path).write_bytes(b"fake-jpeg-bytes")

    monkeypatch.setattr(ffmpeg_utils, "probe_video", fake_probe_video)
    monkeypatch.setattr(ffmpeg_utils, "extract_frame", fake_extract_frame)

    sent_tasks: list[tuple[str, list]] = []
    monkeypatch.setattr(
        celery_app, "send_task", lambda name, args=None, **kw: sent_tasks.append((name, args))
    )

    with TestClient(app) as test_client:
        test_client.sent_tasks = sent_tasks  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def auth_headers(client):
    def _register(email: str = "player@example.com", password: str = "hunter2pass"):
        res = client.post("/auth/register", json={"email": email, "password": password})
        assert res.status_code == 201, res.text
        token = res.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}

    return _register


@pytest.fixture
def uploaded_video(client, auth_headers, tmp_path):
    headers = auth_headers()
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"not-a-real-video-just-bytes-for-the-test")
    with open(video_path, "rb") as fh:
        res = client.post(
            "/videos", headers=headers, files={"file": ("clip.mp4", fh, "video/mp4")}
        )
    assert res.status_code == 201, res.text
    return headers, res.json()
