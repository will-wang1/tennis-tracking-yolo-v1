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
