import json

import numpy as np

from src.analysis.court_calibration import (
    CORNER_ORDER,
    SINGLES_COURT_REFERENCE_POINTS,
    CourtCalibration,
)

# Arbitrary but plausible pixel positions for the four near-court corners,
# roughly matching a broadcast-style behind-baseline camera angle.
_PIXELS = {
    "baseline_left": (340.0, 980.0),
    "baseline_right": (1580.0, 980.0),
    "service_left": (520.0, 650.0),
    "service_right": (1400.0, 650.0),
}


def test_calibration_frame_endpoint_returns_presigned_frame_url(uploaded_video, client):
    headers, video = uploaded_video
    res = client.post(f"/videos/{video['id']}/calibration-frame", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["frame_url"].startswith("http://fake-storage.local/")
    assert body["width"] == video["width"]
    assert body["height"] == video["height"]


def test_calibration_matches_direct_court_calibration_math(uploaded_video, client):
    """The API's homography must be byte-identical to calling
    CourtCalibration.from_points directly with the same points - this is the
    whole point of reusing the class instead of reimplementing the math."""
    headers, video = uploaded_video

    res = client.post(
        f"/videos/{video['id']}/calibration",
        headers=headers,
        json={
            "baseline_left": {"x": _PIXELS["baseline_left"][0], "y": _PIXELS["baseline_left"][1]},
            "baseline_right": {"x": _PIXELS["baseline_right"][0], "y": _PIXELS["baseline_right"][1]},
            "service_left": {"x": _PIXELS["service_left"][0], "y": _PIXELS["service_left"][1]},
            "service_right": {"x": _PIXELS["service_right"][0], "y": _PIXELS["service_right"][1]},
            "court_type": "singles",
        },
    )
    assert res.status_code == 201
    body = res.json()
    assert body["court_type"] == "singles"
    assert body["pixel_points"]["baseline_left"] == list(_PIXELS["baseline_left"])

    expected = CourtCalibration.from_points(
        [_PIXELS[name] for name in CORNER_ORDER],
        [SINGLES_COURT_REFERENCE_POINTS[name] for name in CORNER_ORDER],
    )

    # Round-trip through the API's stored file rather than trusting the
    # response body alone - re-derive world coords for a point both ways and
    # compare, proving the API produced the SAME homography direct use of
    # CourtCalibration.from_points would.
    from app import storage

    user_id = client.get("/auth/me", headers=headers).json()["id"]
    key = f"videos/{user_id}/{video['id']}/calibration_{video['id']}.json"
    saved = CourtCalibration(
        homography=np.array(json.loads(storage.download_bytes(key))["homography"], dtype=np.float64)
    )

    probe_x, probe_y = 900.0, 800.0
    assert saved.pixel_to_world(probe_x, probe_y) == expected.pixel_to_world(probe_x, probe_y)


def test_calibration_requires_all_four_points(uploaded_video, client):
    headers, video = uploaded_video
    res = client.post(
        f"/videos/{video['id']}/calibration",
        headers=headers,
        json={
            "baseline_left": {"x": 0, "y": 0},
            "baseline_right": {"x": 10, "y": 0},
            "service_left": {"x": 0, "y": 5},
            # service_right missing
            "court_type": "singles",
        },
    )
    assert res.status_code == 422


def test_calibration_404_for_non_owner(uploaded_video, client, auth_headers):
    _, video = uploaded_video
    other_headers = auth_headers(email="stranger@example.com")
    res = client.post(f"/videos/{video['id']}/calibration-frame", headers=other_headers)
    assert res.status_code == 404
