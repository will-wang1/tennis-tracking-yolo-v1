"""Overlay tracked ball/pose/stroke/speed/bounce data on a frame.

Each drawer follows the same stateful "call once per frame" pattern:
construct once, then call `.draw(frame, ...)` every frame in a loop (see
main.py). Most drawers mutate and return the same `frame` array; the
exception is `SidebarDrawer`, which returns a new, wider array since it
changes the frame's width - call it last in the per-frame chain.
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration, FULL_COURT_REFERENCE_POINTS
from src.analysis.stroke_classifier import StrokePrediction
from src.detection.pose_detector import PersonPose
from src.tracking.ball_tracker import TrackedPosition

TRAIL_COLOR_DETECTED = (0, 255, 255)  # yellow
TRAIL_COLOR_INTERPOLATED = (0, 165, 255)  # orange

POSE_KEYPOINT_COLOR = (0, 0, 255)  # red
POSE_SKELETON_COLOR = (0, 255, 0)  # green
POSE_SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

BOUNCE_MARKER_COLOR = (255, 0, 255)  # magenta

SIDEBAR_BACKGROUND = (30, 30, 30)
SIDEBAR_TEXT_COLOR = (255, 255, 255)

COURT_LINE_COLOR = (0, 200, 255)  # orange
COURT_CORNER_COLOR = (0, 255, 255)  # yellow
COURT_LINE_EDGES = [
    ("baseline_far_left", "baseline_far_right"),
    ("baseline_near_left", "baseline_near_right"),
    ("baseline_far_left", "baseline_near_left"),
    ("baseline_far_right", "baseline_near_right"),
    ("singles_far_left", "singles_near_left"),
    ("singles_far_right", "singles_near_right"),
    ("service_far_left", "service_far_right"),
    ("service_near_left", "service_near_right"),
    ("center_service_far", "center_service_near"),
]
# The 4 doubles-court corners, as opposed to the 14 full keypoints - what a
# viewer would point to as "the corners of the court".
COURT_CORNER_NAMES = ("baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right")


class TrailDrawer:
    def __init__(self, trail_length: int = 15):
        self.trail: deque[TrackedPosition] = deque(maxlen=trail_length)

    def draw(self, frame: np.ndarray, position: Optional[TrackedPosition]) -> np.ndarray:
        if position is not None:
            self.trail.append(position)

        for i, point in enumerate(self.trail):
            color = TRAIL_COLOR_INTERPOLATED if point.interpolated else TRAIL_COLOR_DETECTED
            fade = (i + 1) / max(len(self.trail), 1)
            radius = max(2, int(6 * fade))
            cv2.circle(frame, (int(point.x), int(point.y)), radius, color, -1)

        return frame


class PoseDrawer:
    """Skeleton lines/joints for every detected player, plus a stroke label
    above the striker's head when a classifier prediction is available."""

    def draw(
        self,
        frame: np.ndarray,
        poses: list[PersonPose],
        striker: Optional[PersonPose] = None,
        stroke: Optional[StrokePrediction] = None,
    ) -> np.ndarray:
        for pose in poses:
            for a, b in POSE_SKELETON_EDGES:
                xa, ya = pose.keypoints[a]
                xb, yb = pose.keypoints[b]
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), POSE_SKELETON_COLOR, 2)
            for x, y in pose.keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, POSE_KEYPOINT_COLOR, -1)

        if stroke is not None and striker is not None:
            head_x, head_y = striker.keypoints[0]  # nose
            label = f"{stroke.label.upper()} ({stroke.confidence:.0%})"
            cv2.putText(
                frame,
                label,
                (int(head_x) - 40, int(head_y) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                POSE_SKELETON_COLOR,
                2,
            )

        return frame


class BounceMarkerDrawer:
    """Every bounce detected so far stays drawn for the rest of the video -
    a persistent landing map, not a transient flash."""

    def __init__(self):
        self.markers: list[BounceEvent] = []

    def draw(self, frame: np.ndarray, bounce: Optional[BounceEvent]) -> np.ndarray:
        if bounce is not None:
            self.markers.append(bounce)

        for marker in self.markers:
            x, y = int(marker.x), int(marker.y)
            cv2.drawMarker(frame, (x, y), BOUNCE_MARKER_COLOR, cv2.MARKER_TILTED_CROSS, 16, 2)

        return frame


class CourtOverlayDrawer:
    """Court-line wireframe + corner markers, reprojected fresh from
    whichever `CourtCalibration` is passed to `draw()` each call - a panning/
    zooming broadcast camera changes the pixel<->world mapping frame to
    frame, so the reprojection can't be cached once like a fixed-camera
    assumption would allow. `calibration=None` (the court detector lost the
    court that frame) just skips drawing for that frame."""

    def draw(self, frame: np.ndarray, calibration: Optional[CourtCalibration]) -> np.ndarray:
        if calibration is None:
            return frame

        inverse_homography = np.linalg.inv(calibration.homography)
        points_px: dict[str, tuple[float, float]] = {}
        for name, (world_x, world_y) in FULL_COURT_REFERENCE_POINTS.items():
            point = np.array([[[world_x, world_y]]], dtype=np.float64)
            px, py = cv2.perspectiveTransform(point, inverse_homography)[0, 0]
            points_px[name] = (float(px), float(py))

        for a, b in COURT_LINE_EDGES:
            xa, ya = points_px[a]
            xb, yb = points_px[b]
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), COURT_LINE_COLOR, 2)

        for name in COURT_CORNER_NAMES:
            x, y = points_px[name]
            cv2.circle(frame, (int(x), int(y)), 8, COURT_CORNER_COLOR, -1)

        return frame


class SidebarDrawer:
    """Composites a fixed-width text panel to the right of the frame.
    Returns a NEW, wider array - unlike the other drawers here, it doesn't
    mutate in place. Call this LAST in the per-frame chain."""

    def __init__(self, width: int = 250):
        self.width = width

    def draw(
        self,
        frame: np.ndarray,
        stroke: Optional[StrokePrediction],
        speed: Optional[tuple[float, str]],
    ) -> np.ndarray:
        """`speed` is (value, unit) - e.g. (42.3, "km/h") - the current
        LIVE instantaneous ball speed this frame, not a shot summary, so it
        updates continuously rather than only within bounce-segmented shots."""
        height = frame.shape[0]
        sidebar = np.full((height, self.width, 3), SIDEBAR_BACKGROUND, dtype=np.uint8)

        lines = ["Stroke:", stroke.label.upper() if stroke else "-", "", "Speed:"]
        if speed is not None:
            value, unit = speed
            lines.append(f"{value:.0f} {unit}")
        else:
            lines.append("-")

        for i, line in enumerate(lines):
            cv2.putText(
                sidebar,
                line,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                SIDEBAR_TEXT_COLOR,
                1,
            )

        return np.hstack([frame, sidebar])
