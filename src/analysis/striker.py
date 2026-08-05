"""Pick which detected person is the striker this frame, and estimate where
the court surface is near the ball.

Simplest useful rule for the striker: whichever person's bounding-box
center is nearest the ball. No temporal smoothing - a documented,
not-solved-here limitation, since this can flicker when the ball is near
the net between two closely-spaced players.
"""

from typing import Optional

import numpy as np

from src.detection.pose_detector import PersonPose
from src.tracking.ball_tracker import TrackedPosition

LEFT_ANKLE, RIGHT_ANKLE = 15, 16
MIN_KEYPOINT_CONFIDENCE = 0.3


def select_striker(poses: list[PersonPose], ball: Optional[TrackedPosition]) -> Optional[PersonPose]:
    if ball is None or not poses:
        return None

    distances = [float(np.hypot(p.center_x - ball.x, p.center_y - ball.y)) for p in poses]
    nearest_idx = int(np.argmin(distances))
    return poses[nearest_idx]


def select_players(poses: list[PersonPose], frame_width: float, count: int = 2) -> list[PersonPose]:
    """Filter pose detections down to the `count` most likely players.

    A standard behind-baseline broadcast shot centers the court in frame,
    with the chair umpire, ball kids, and camera crew off to the sides -
    so the `count` poses horizontally nearest the frame center are taken
    as the players. No temporal smoothing, same documented limitation as
    `select_striker`.
    """
    return sorted(poses, key=lambda p: abs(p.center_x - frame_width / 2))[:count]


def estimate_ground_y(poses: list[PersonPose], ball: Optional[TrackedPosition]) -> Optional[float]:
    """Estimate the court's ground level near the ball, for cross-checking
    bounce candidates (see `src.analysis.bounce_detector`). Uses whichever
    player is spatially closest to the ball as a reference: a standing
    player's feet touch the court, so their ankle height is a proxy for
    "where the ground is" at that player's on-screen depth. Falls back to
    the lowest confidently-detected keypoint if ankles aren't visible (e.g.
    occluded by the net or cut off by the frame edge). None if there's no
    pose/ball to reference at all.
    """
    nearest = select_striker(poses, ball)
    if nearest is None:
        return None

    ankle_ys = [
        nearest.keypoints[idx][1]
        for idx in (LEFT_ANKLE, RIGHT_ANKLE)
        if nearest.keypoint_confidence[idx] >= MIN_KEYPOINT_CONFIDENCE
    ]
    if ankle_ys:
        return float(max(ankle_ys))

    visible_ys = [
        nearest.keypoints[idx][1]
        for idx in range(len(nearest.keypoints))
        if nearest.keypoint_confidence[idx] >= MIN_KEYPOINT_CONFIDENCE
    ]
    return float(max(visible_ys)) if visible_ys else None
