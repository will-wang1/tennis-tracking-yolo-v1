def test_register_then_me(client):
    res = client.post("/auth/register", json={"email": "a@example.com", "password": "hunter2pass"})
    assert res.status_code == 201
    token = res.json()["access_token"]

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "a@example.com"


def test_register_duplicate_email_rejected(client):
    payload = {"email": "dup@example.com", "password": "hunter2pass"}
    first = client.post("/auth/register", json=payload)
    assert first.status_code == 201
    second = client.post("/auth/register", json=payload)
    assert second.status_code == 409


def test_login_wrong_password_rejected(client):
    client.post("/auth/register", json={"email": "b@example.com", "password": "correct-password"})
    res = client.post("/auth/login", json={"email": "b@example.com", "password": "wrong-password"})
    assert res.status_code == 401


def test_login_success_round_trip(client):
    client.post("/auth/register", json={"email": "c@example.com", "password": "correct-password"})
    res = client.post("/auth/login", json={"email": "c@example.com", "password": "correct-password"})
    assert res.status_code == 200
    assert "access_token" in res.json()


def test_protected_route_requires_token(client):
    res = client.get("/videos")
    assert res.status_code == 401


def test_registration_open_by_default(client):
    res = client.get("/config")
    assert res.json()["invite_required"] is False


def test_registration_requires_invite_code_when_configured(client, monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("INVITE_CODE", "letmein")
    get_settings.cache_clear()
    try:
        assert client.get("/config").json()["invite_required"] is True

        no_code = client.post("/auth/register", json={"email": "nc@example.com", "password": "hunter2pass"})
        assert no_code.status_code == 403

        wrong_code = client.post(
            "/auth/register",
            json={"email": "wc@example.com", "password": "hunter2pass", "invite_code": "nope"},
        )
        assert wrong_code.status_code == 403

        right_code = client.post(
            "/auth/register",
            json={"email": "rc@example.com", "password": "hunter2pass", "invite_code": "letmein"},
        )
        assert right_code.status_code == 201
    finally:
        get_settings.cache_clear()
