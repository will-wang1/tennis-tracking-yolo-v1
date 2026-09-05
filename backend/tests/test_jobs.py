from app.db import SessionLocal
from app.models import Job


def _job_options(bounce=True, speed=True, sidebar=False, minimap=False):
    return {"bounce": bounce, "speed": speed, "sidebar": sidebar, "minimap": minimap}


def test_create_job_dispatches_celery_task_by_name(client, uploaded_video):
    headers, video = uploaded_video
    res = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options())
    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "queued"
    assert body["progress"] == 0
    assert body["options"] == _job_options()

    # Dispatched by task name (not by importing the task function directly -
    # see routers/jobs.py) so the API process never needs the ML stack.
    assert client.sent_tasks == [("app.tasks.process_video_task", [body["id"]])]


def test_minimap_rejected_when_no_court_weights_configured(client, uploaded_video):
    headers, video = uploaded_video
    res = client.post(
        f"/videos/{video['id']}/jobs", headers=headers, json=_job_options(minimap=True)
    )
    assert res.status_code == 400
    assert "minimap" in res.json()["detail"].lower() or "court" in res.json()["detail"].lower()


def test_job_status_transitions_are_visible_via_api(client, uploaded_video):
    """Simulates what the worker does to a Job row (without running the real
    pipeline) and checks the API reflects it - this is the contract the
    frontend's polling JobStatus page depends on."""
    headers, video = uploaded_video
    created = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()
    job_id = created["id"]

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        job.status = "running"
        job.progress = 42
        db.commit()
    finally:
        db.close()

    res = client.get(f"/jobs/{job_id}", headers=headers)
    assert res.status_code == 200
    assert res.json()["status"] == "running"
    assert res.json()["progress"] == 42


def test_job_result_before_completion_has_no_video_or_stats(client, uploaded_video):
    headers, video = uploaded_video
    created = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()

    res = client.get(f"/jobs/{created['id']}/result", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "queued"
    assert body["video_url"] is None
    assert body["stats"] is None


def test_job_result_after_completion_returns_video_and_stats(client, uploaded_video):
    from app import storage

    headers, video = uploaded_video
    created = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()
    job_id = created["id"]

    stats_key = f"videos/{video['id']}/jobs/{job_id}/stats.json"
    output_key = f"videos/{video['id']}/jobs/{job_id}/output.mp4"
    storage.upload_bytes(b'{"rally_count": 1, "total_bounces": 2}', stats_key, "application/json")

    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        job.status = "done"
        job.progress = 100
        job.output_video_key = output_key
        job.stats_json_key = stats_key
        db.commit()
    finally:
        db.close()

    res = client.get(f"/jobs/{job_id}/result", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "done"
    assert body["video_url"] == f"http://fake-storage.local/{output_key}"
    assert body["stats"] == {"rally_count": 1, "total_bounces": 2}


def test_job_for_non_owner_is_404(client, uploaded_video, auth_headers):
    headers, video = uploaded_video
    created = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()

    other_headers = auth_headers(email="not-the-owner-2@example.com")
    res = client.get(f"/jobs/{created['id']}", headers=other_headers)
    assert res.status_code == 404


def test_create_job_rejects_unknown_calibration(client, uploaded_video):
    headers, video = uploaded_video
    res = client.post(
        f"/videos/{video['id']}/jobs",
        headers=headers,
        json={**_job_options(), "calibration_id": "does-not-exist"},
    )
    assert res.status_code == 404


def _set_job(job_id: str, **fields) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        for key, value in fields.items():
            setattr(job, key, value)
        db.commit()
    finally:
        db.close()


def test_video_list_carries_the_latest_job_for_background_progress(client, uploaded_video):
    """The home screen shows analysis progress per video without a request
    per video, so the video list has to carry each video's newest job."""
    headers, video = uploaded_video

    before = client.get("/videos", headers=headers).json()
    assert before[0]["latest_job"] is None

    created = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()
    _set_job(created["id"], status="running", progress=37)

    listed = client.get("/videos", headers=headers).json()
    assert listed[0]["latest_job"]["id"] == created["id"]
    assert listed[0]["latest_job"]["status"] == "running"
    assert listed[0]["latest_job"]["progress"] == 37

    # A second run on the same video supersedes the first.
    newer = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()
    listed = client.get("/videos", headers=headers).json()
    assert listed[0]["latest_job"]["id"] == newer["id"]


def test_list_jobs_can_narrow_to_active_ones(client, uploaded_video):
    headers, video = uploaded_video
    finished = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()
    _set_job(finished["id"], status="done", progress=100)
    running = client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options()).json()
    _set_job(running["id"], status="running", progress=10)

    all_ids = {job["id"] for job in client.get("/jobs", headers=headers).json()}
    assert all_ids == {finished["id"], running["id"]}

    active = client.get("/jobs?active=true", headers=headers).json()
    assert [job["id"] for job in active] == [running["id"]]


def test_list_jobs_only_shows_your_own(client, uploaded_video, auth_headers):
    headers, video = uploaded_video
    client.post(f"/videos/{video['id']}/jobs", headers=headers, json=_job_options())

    other_headers = auth_headers(email="someone-else-jobs@example.com")
    assert client.get("/jobs", headers=other_headers).json() == []
