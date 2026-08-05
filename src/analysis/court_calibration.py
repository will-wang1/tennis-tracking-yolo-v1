"""Map pixel coordinates to real-world court coordinates (meters).

There's no automatic court-line detection in this repo yet, so calibration
is a one-time manual step per camera angle: a human reads off the pixel
coordinates of four known court corners (see `scripts/calibrate_court.py`)
and this module turns that into a homography.

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


@dataclass
class CourtCalibration:
    homography: np.ndarray  # 3x3, maps pixel (x, y) -> world (X, Y) meters

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        point = np.array([[[x, y]]], dtype=np.float64)
        mapped = cv2.perspectiveTransform(point, self.homography)
        wx, wy = mapped[0, 0]
        return float(wx), float(wy)

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"homography": self.homography.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        data = json.loads(Path(path).read_text())
        return cls(homography=np.array(data["homography"], dtype=np.float64))
