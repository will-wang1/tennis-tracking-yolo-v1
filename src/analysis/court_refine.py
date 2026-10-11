"""Tighten a court calibration onto the painted lines.

The court comes from a keypoint detector (TennisCourtNet) whose corners are
a few pixels off - near-corner calls on Dingles split into two clusters
~80px apart - and every bounce position, every foot position, inherits that
error: on a fence camera one pixel at the far baseline is ~10x more court
than at the near one. This moves the four outer corners (the whole
homography: 8 numbers) until the court lines the calibration predicts sit
on the lines actually painted in the picture.

The picture is the MEDIAN of many frames of the court view, which removes
players and balls and leaves the court; its lines are picked out with a
white top-hat (bright, thin structure) and blurred a little so the score
rises smoothly as a predicted line slides onto a painted one. The score is
the mean line strength along all nine court lines, sampled at sub-pixel
positions, each line weighted equally. Corners move at most MAX_SHIFT_PX:
further, and a line could lock onto the next court's.
"""

from typing import Optional

import cv2
import numpy as np

from src.analysis.court_calibration import _LINE_EDGES, FULL_COURT_REFERENCE_POINTS, CourtCalibration

MAX_SHIFT_PX = 15.0
CORNERS = ("baseline_far_left", "baseline_far_right", "baseline_near_right", "baseline_near_left")
SAMPLES_PER_LINE = 120


def line_strength_image(median_gray: np.ndarray, blur_sigma: float = 1.5) -> np.ndarray:
    """Bright thin structure (painted lines) of a frame, smoothed."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    tophat = cv2.morphologyEx(median_gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    return cv2.GaussianBlur(tophat, (0, 0), blur_sigma)


def _calibration_from_corners(pixels: np.ndarray) -> CourtCalibration:
    world = np.array([FULL_COURT_REFERENCE_POINTS[c] for c in CORNERS], np.float32)
    h = cv2.getPerspectiveTransform(pixels.astype(np.float32), world)
    return CourtCalibration(homography=h)


def line_score(strength: np.ndarray, calibration: CourtCalibration) -> float:
    """Mean line strength under the court lines this calibration predicts."""
    height, width = strength.shape
    per_line = []
    for a, b in _LINE_EDGES:
        pa = np.array(calibration.world_to_pixel(*FULL_COURT_REFERENCE_POINTS[a]))
        pb = np.array(calibration.world_to_pixel(*FULL_COURT_REFERENCE_POINTS[b]))
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        t = np.linspace(0.03, 0.97, SAMPLES_PER_LINE)[:, None]
        pts = pa + t * (pb - pa)
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < width - 1) & (pts[:, 1] >= 0) & (pts[:, 1] < height - 1)
        if inside.sum() < 10:
            continue
        p = pts[inside].astype(np.float32)
        vals = cv2.remap(strength, p[:, 0].reshape(-1, 1), p[:, 1].reshape(-1, 1), cv2.INTER_LINEAR)
        per_line.append(float(vals.mean()))
    return float(np.mean(per_line)) if per_line else 0.0


def refine_calibration(median_gray: np.ndarray, calibration: CourtCalibration,
                       max_shift: float = MAX_SHIFT_PX) -> tuple[CourtCalibration, dict]:
    """The calibration moved onto the painted lines, and a record of the
    change: score before and after, and how far each corner moved."""
    from scipy.optimize import minimize

    start = np.array([calibration.world_to_pixel(*FULL_COURT_REFERENCE_POINTS[c]) for c in CORNERS], float)
    record = {"refined": False}
    best_cal, best_score = calibration, None
    x = np.zeros(8)
    for sigma in (4.0, 1.5):  # coarse first, so the score has a wide basin, then sharp
        strength = line_strength_image(median_gray, sigma)
        if best_score is None:
            record["score_before"] = round(line_score(line_strength_image(median_gray, 1.5), calibration), 3)

        def cost(v):
            shift = v.reshape(4, 2)
            over = np.clip(np.linalg.norm(shift, axis=1) - max_shift, 0, None)
            try:
                cal = _calibration_from_corners(start + shift)
            except cv2.error:
                return 1e9
            return -line_score(strength, cal) + 10.0 * float((over ** 2).sum())

        res = minimize(cost, x, method="Powell", options={"xtol": 0.05, "ftol": 1e-4, "maxiter": 4000})
        x = res.x
        best_score = -res.fun
    shift = x.reshape(4, 2)
    refined = _calibration_from_corners(start + shift)
    after = line_score(line_strength_image(median_gray, 1.5), refined)
    record.update(score_after=round(after, 3),
                  corner_shift_px={c: [round(float(d[0]), 2), round(float(d[1]), 2)] for c, d in zip(CORNERS, shift)})
    if after > record["score_before"]:
        record["refined"] = True
        return refined, record
    return calibration, record  # never make it worse


def median_court_frame(video: str, frames: list[int]) -> Optional[np.ndarray]:
    """Grayscale median of these frames: the court without players or balls."""
    capture = cv2.VideoCapture(video)
    stack = []
    for f in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = capture.read()
        if ok:
            stack.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(stack) < 3:
        return None
    return np.median(np.stack(stack), axis=0).astype(np.uint8)
