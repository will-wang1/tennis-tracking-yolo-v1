def test_upload_video_returns_probed_metadata(uploaded_video):
    headers, video = uploaded_video
    assert video["filename"] == "clip.mp4"
    assert video["duration_s"] == 12.3
    assert video["fps"] == 30.0
    assert video["width"] == 1920
    assert video["height"] == 1080
    assert video["status"] == "uploaded"


def test_upload_rejects_unsupported_extension(client, auth_headers, tmp_path):
    headers = auth_headers()
    bogus = tmp_path / "notes.txt"
    bogus.write_text("hello")
    with open(bogus, "rb") as fh:
        res = client.post("/videos", headers=headers, files={"file": ("notes.txt", fh, "text/plain")})
    assert res.status_code == 400


def test_list_videos_scoped_to_owner(client, auth_headers, uploaded_video):
    owner_headers, video = uploaded_video
    other_headers = auth_headers(email="someone-else@example.com")

    mine = client.get("/videos", headers=owner_headers)
    assert mine.status_code == 200
    assert [v["id"] for v in mine.json()] == [video["id"]]

    others = client.get("/videos", headers=other_headers)
    assert others.status_code == 200
    assert others.json() == []


def test_get_video_404_for_non_owner(client, auth_headers, uploaded_video):
    _, video = uploaded_video
    other_headers = auth_headers(email="not-the-owner@example.com")
    res = client.get(f"/videos/{video['id']}", headers=other_headers)
    assert res.status_code == 404
