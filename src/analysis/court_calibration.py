"""Map pixel coordinates to real-world court coordinates (meters).

Two ways to build a calibration:

- Manual (`scripts/calibrate_court.py`): a human reads off the pixel
  coordinates of four known near-court corners from a still frame. Quick,
  needs no trained model, but per-video and only as precise as the human
  reading the pixels.
- Automatic (`scripts/calibrate_court_auto.py`, needs a trained
  `src.detection.court_keypoint_detector.CourtKeypointDetector`): the model
  detects up to all 14 standard court keypoints in one frame and
  `from_keypoints` fits a homography from however many it's confident
  about (more points than the minimum 4 makes the fit more robust to any
  single point's detection error, via `cv2.findHomography`'s RANSAC).

The homography is a ground-plane mapping - exact for points on the court
surface (like a bounce location), increasingly approximate the higher above
the court a point is (e.g. the ball mid-flight during a serve). That's a
known, accepted limitation: there's no calibrated depth/height model here.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Standard tennis court dimensions, meters. Origin at the near baseline-left
# corner (as seen by the camera), x along the baseline, y along the sideline
# toward the net.
SINGLES_COURT_REFERENCE_POINTS = {
    "baseline_left": (0.0, 0.0),
    "baseline_right": (8.23, 0.0),
    "service_left": (0.0, 5.485),
    "service_right": (8.23, 5.485),
}

DOUBLES_COURT_REFERENCE_POINTS = {
    "baseline_left": (0.0, 0.0),
    "baseline_right": (10.97, 0.0),
    "service_left": (0.0, 5.485),
    "service_right": (10.97, 5.485),
}

CORNER_ORDER = ("baseline_left", "baseline_right", "service_right", "service_left")

# The standard 14-keypoint tennis court layout used by
# src.detection.court_keypoint_detector (matching the well-established
# yastrebksv/TennisCourtDetector reference dataset's annotation order/
# semantics). Origin at the FAR baseline-left DOUBLES corner (keypoint 0),
# x increasing toward the right doubles sideline, y increasing toward the
# NEAR baseline (i.e. down the length of the court, toward the camera for
# a typical behind-baseline broadcast angle). Doubles court 10.97m wide,
# singles 8.23m (inset 1.37m from each doubles sideline), full length
# 23.77m, service line 5.485m from each baseline.
_DOUBLES_WIDTH = 10.97
_SINGLES_WIDTH = 8.23
_SINGLES_INSET = (_DOUBLES_WIDTH - _SINGLES_WIDTH) / 2  # 1.37
_COURT_LENGTH = 23.77
_SERVICE_LINE_FROM_BASELINE = 5.485
_CENTER_X = _SINGLES_INSET + _SINGLES_WIDTH / 2  # 5.485, by construction symmetric

FULL_COURT_REFERENCE_POINTS = {
    "baseline_far_left": (0.0, 0.0),
    "baseline_far_right": (_DOUBLES_WIDTH, 0.0),
    "baseline_near_left": (0.0, _COURT_LENGTH),
    "baseline_near_right": (_DOUBLES_WIDTH, _COURT_LENGTH),
    "singles_far_left": (_SINGLES_INSET, 0.0),
    "singles_near_left": (_SINGLES_INSET, _COURT_LENGTH),
    "singles_far_right": (_SINGLES_INSET + _SINGLES_WIDTH, 0.0),
    "singles_near_right": (_SINGLES_INSET + _SINGLES_WIDTH, _COURT_LENGTH),
    "service_far_left": (_SINGLES_INSET, _SERVICE_LINE_FROM_BASELINE),
    "service_far_right": (_SINGLES_INSET + _SINGLES_WIDTH, _SERVICE_LINE_FROM_BASELINE),
    "service_near_left": (_SINGLES_INSET, _COURT_LENGTH - _SERVICE_LINE_FROM_BASELINE),
    "service_near_right": (_SINGLES_INSET + _SINGLES_WIDTH, _COURT_LENGTH - _SERVICE_LINE_FROM_BASELINE),
    "center_service_far": (_CENTER_X, _SERVICE_LINE_FROM_BASELINE),
    "center_service_near": (_CENTER_X, _COURT_LENGTH - _SERVICE_LINE_FROM_BASELINE),
}


@dataclass
class CourtCalibration:
    homography: np.ndarray  # 3x3, maps pixel (x, y) -> world (X, Y) meters

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(point, self.homography)
        wx, wy = mapped[0, 0]
        return float(wx), float(wy)

    def world_to_pixel(self, world_x: float, world_y: float) -> tuple[float, float]:
        """The inverse of `pixel_to_world`: where a fixed COURT position
        reprojects to on screen this frame.

        This is what keeps something drawn at a court-relative position
        visually locked to the court as a panning/zooming broadcast camera
        moves - the homography changes frame to frame, so the reprojection
        has to be recomputed fresh each call, the same as
        `CourtOverlayDrawer` already does for the court lines themselves.
        Anything meant to represent a fixed spot on the court (a bounce
        mark, say) needs this rather than a pixel position captured once
        and held fixed on screen, which drifts away from the court lines
        the moment the camera moves - measured on the zverev clip, a fixed
        court point reprojects up to 94px away from where it started over
        the clip's 744 frames.
        """
        point = np.array([[[world_x, world_y]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(point, np.linalg.inv(self.homography))
        x, y = mapped[0, 0]
        return float(x), float(y)

    def pixel_distance_to_meters(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Distance in meters between two pixel points, both mapped through
        the homography first - not a uniform px-per-meter scalar, since
        perspective means scale varies across the court."""
        wx1, wy1 = self.pixel_to_world(x1, y1)
        wx2, wy2 = self.pixel_to_world(x2, y2)
        return float(np.hypot(wx2 - wx1, wy2 - wy1))

    @classmethod
    def from_points(
        cls,
        pixel_points: list[tuple[float, float]],
        world_points: list[tuple[float, float]],
    ) -> "CourtCalibration":
        if len(pixel_points) != 4 or len(world_points) != 4:
            raise ValueError("Exactly 4 point correspondences are required")
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        homography = cv2.getPerspectiveTransform(src, dst)
        return cls(homography=homography.astype(np.float64))

    @classmethod
    def from_keypoints(
        cls,
        detected_pixel_points: dict[str, tuple[float, float]],
        world_points: dict[str, tuple[float, float]] = FULL_COURT_REFERENCE_POINTS,
        min_points: int = 4,
    ) -> "CourtCalibration":
        """Build a calibration from however many named keypoints were
        confidently detected (see
        `src.detection.court_keypoint_detector.CourtKeypointDetector`) -
        `detected_pixel_points` need not include all 14. With exactly 4
        points this is equivalent to `from_points`; with more, fits via
        `cv2.findHomography`'s RANSAC, which is more robust to any single
        point's detection error than an exact 4-point transform.
        """
        common_names = [name for name in world_points if name in detected_pixel_points]
        if len(common_names) < min_points:
            raise ValueError(
                f"Need at least {min_points} matched keypoints, got {len(common_names)}"
            )

        src = np.array([detected_pixel_points[name] for name in common_names], dtype=np.float32)
        dst = np.array([world_points[name] for name in common_names], dtype=np.float32)

        if len(common_names) == 4:
            homography = cv2.getPerspectiveTransform(src, dst)
        else:
            homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
        return cls(homography=homography.astype(np.float64))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"homography": self.homography.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        data = json.loads(Path(path).read_text())
        return cls(homography=np.array(data["homography"], dtype=np.float64))
