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

# The court's four outer corners in order around its boundary, used by
# `CourtCalibration.is_plausible_view` to judge a fit by what it does to the
# court's own shape. Order matters: consecutive entries must be adjacent
# corners, or the convexity test below reads a self-intersecting quad.
_COURT_BOUNDARY = (
    FULL_COURT_REFERENCE_POINTS["baseline_far_left"],
    FULL_COURT_REFERENCE_POINTS["baseline_far_right"],
    FULL_COURT_REFERENCE_POINTS["baseline_near_right"],
    FULL_COURT_REFERENCE_POINTS["baseline_near_left"],
)

# How far outside the frame a reprojected court corner may land, in
# multiples of the frame's larger side. A real broadcast view keeps the
# court near the frame; a near-degenerate fit throws corners thousands of
# px away (measured up to 260,000px on real footage - see
# `is_plausible_view`). Generous on purpose: this is meant to catch fits
# that are numerically broken, not fits that are merely unusual.
DEFAULT_MAX_EXTENT_FRAMES = 3.0
# Smallest share of the frame a genuine court view may cover. Measured: the
# smallest real court view on the Alcaraz-Djokovic clip covered 6.1% of
# frame, the broken slivers this rejects covered 0.10-0.56%, so 1% sits an
# order of magnitude clear of real footage on one side and ~2x clear of the
# failures on the other.
DEFAULT_MIN_COURT_AREA_FRACTION = 0.01


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

    def is_plausible_view(
        self,
        frame_width: int,
        frame_height: int,
        max_extent_frames: float = DEFAULT_MAX_EXTENT_FRAMES,
        min_area_fraction: float = DEFAULT_MIN_COURT_AREA_FRACTION,
    ) -> bool:
        """Whether this homography could describe a real camera looking at a
        real court, judged by reprojecting the court's own outer corners.

        `from_keypoints` accepts any four matched keypoints without checking
        the fit that results, and a 4-point fit is exact, so it has no
        residual to check even in principle. That means a frame showing a
        replay, a closeup, or a crowd shot can produce a confident,
        completely wrong homography, and nothing downstream can tell.

        FOUND ON REAL FOOTAGE: on the Alcaraz-Djokovic clip, 133 of 3725
        frames carried a calibration that reprojected the court's corners
        up to 260,000px away - the court mapped essentially to infinity.
        Every one was accepted and used. All 133 fell between the camera
        cuts at frames 2233 and 2331 (89.3s-93.2s), a replay shot: cut
        detection correctly reset the calibration going in, the very next
        frame fit garbage to the replay's own lines, and main.py's
        carry-forward then held that single bad fit across the whole shot.
        So the missing piece is not cut detection - which worked - but a
        check on the fit itself.

        Three tests, most principled first:

        - CONVEX: a homography maps the court rectangle to a convex
          quadrilateral for any view from in front of the court plane. A
          non-convex result means corners crossed the horizon, which no
          real camera position produces. No threshold, no tuning.
        - EXTENT: corners must be finite and land within
          `max_extent_frames` frame-sizes of the image. A corner thousands
          of px outside means a near-degenerate, nearly edge-on fit.
        - AREA: the court must cover at least `min_area_fraction` of the
          frame. The most arbitrary of the three, but measured with a wide
          margin: on the same clip the smallest genuine court view covered
          6.1% of frame while the broken slivers this catches covered
          0.10-0.56%, an order of magnitude apart.

        Measured on that clip, each test independently rejected only frames
        inside the bad shot and NONE of the other 2850 - and the nearest
        hand-labelled event of any kind sits 1.06s away from anything the
        combined gate rejects, so applying it costs no labelled event.
        """
        try:
            inverse = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            return False

        source = np.array(_COURT_BOUNDARY, dtype=np.float64).reshape(1, -1, 2)
        corners = cv2.perspectiveTransform(source, inverse)[0]
        if not np.all(np.isfinite(corners)):
            return False

        limit = max_extent_frames * max(frame_width, frame_height)
        if max(max(abs(x), abs(y)) for x, y in corners) > limit:
            return False

        signs = []
        for i in range(4):
            ax, ay = corners[i]
            bx, by = corners[(i + 1) % 4]
            cx, cy = corners[(i + 2) % 4]
            signs.append((bx - ax) * (cy - by) - (by - ay) * (cx - bx) > 0)
        if any(signs) and not all(signs):
            return False

        area = 0.0
        for (x1, y1), (x2, y2) in zip(corners, np.roll(corners, -1, axis=0)):
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0 >= min_area_fraction * frame_width * frame_height

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

        # findHomography RETURNS None when RANSAC cannot fit one at all -
        # near-collinear keypoints, or too few points agreeing to form a
        # consensus. Not hypothetical: this crashed a cache build outright
        # partway through clips/paul_alcaraz_25fps.mp4 with
        # "'NoneType' object has no attribute 'astype'". Raising here keeps
        # the failure in the same category callers already handle - a frame
        # with no usable fit, same as one with too few keypoints - instead
        # of taking down a job that is minutes deep in a video.
        if homography is None:
            raise ValueError(
                f"no homography could be fitted from {len(common_names)} matched keypoints "
                "(degenerate configuration - points near-collinear or no RANSAC consensus)"
            )
        return cls(homography=homography.astype(np.float64))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"homography": self.homography.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> "CourtCalibration":
        data = json.loads(Path(path).read_text())
        return cls(homography=np.array(data["homography"], dtype=np.float64))
