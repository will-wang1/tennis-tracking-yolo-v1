"""Bounce detection via the pretrained CatBoost regressor from
yastrebksv/TennisProject's bounce_detector.py - trained on trajectory-shape
features (lag/lead x,y differences and their ratios) rather than pixel
appearance, so unlike the ball/court models it needs no image input at all,
just the tracked (x, y) sequence. Ported near-verbatim; only the public
surface (`detect_bounces_catboost`) is new, to match this project's
`BounceEvent`/`TrackedPosition` types instead of the original's bare lists.

Replaces `src.analysis.bounce_detector.detect_bounces`' hand-tuned geometric
heuristics (y-prominence, x-reversal, ground-proximity) with one model
trained specifically to discriminate bounces from contacts - at the usual
pretrained-on-someone-else's-footage cost: it wasn't tuned on this project's
cameras.
"""

from typing import Optional

import catboost as ctb
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition


class CatBoostBounceDetector:
    def __init__(self, model_path: str, threshold: float = 0.45):
        self.model = ctb.CatBoostRegressor()
        self.model.load_model(str(model_path))
        self.threshold = threshold

    def predict_frames(self, x_ball: list, y_ball: list, smooth: bool = True) -> set[int]:
        """`x_ball`/`y_ball` are mutated in place by the smoothing step
        (missed frames get a cubic-spline-extrapolated value filled in) -
        callers that want the post-smoothing values back (to place a bounce
        marker, say) can just read the same lists back out afterward."""
        if smooth:
            x_ball, y_ball = self._smooth_predictions(x_ball, y_ball)
        features, frame_nums = self._prepare_features(x_ball, y_ball)
        if features.empty:
            return set()
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self._postprocess(ind_bounce, preds)
        return {frame_nums[i] for i in ind_bounce}

    def _prepare_features(self, x_ball: list, y_ball: list) -> tuple[pd.DataFrame, list]:
        labels = pd.DataFrame({"frame": range(len(x_ball)), "x-coordinate": x_ball, "y-coordinate": y_ball})

        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels[f"x_lag_{i}"] = labels["x-coordinate"].shift(i)
            labels[f"x_lag_inv_{i}"] = labels["x-coordinate"].shift(-i)
            labels[f"y_lag_{i}"] = labels["y-coordinate"].shift(i)
            labels[f"y_lag_inv_{i}"] = labels["y-coordinate"].shift(-i)
            labels[f"x_diff_{i}"] = abs(labels[f"x_lag_{i}"] - labels["x-coordinate"])
            labels[f"y_diff_{i}"] = labels[f"y_lag_{i}"] - labels["y-coordinate"]
            labels[f"x_diff_inv_{i}"] = abs(labels[f"x_lag_inv_{i}"] - labels["x-coordinate"])
            labels[f"y_diff_inv_{i}"] = labels[f"y_lag_inv_{i}"] - labels["y-coordinate"]
            labels[f"x_div_{i}"] = abs(labels[f"x_diff_{i}"] / (labels[f"x_diff_inv_{i}"] + eps))
            labels[f"y_div_{i}"] = labels[f"y_diff_{i}"] / (labels[f"y_diff_inv_{i}"] + eps)

        for i in range(1, num):
            labels = labels[labels[f"x_lag_{i}"].notna()]
            labels = labels[labels[f"x_lag_inv_{i}"].notna()]
        labels = labels[labels["x-coordinate"].notna()]

        colnames_x = (
            [f"x_diff_{i}" for i in range(1, num)]
            + [f"x_diff_inv_{i}" for i in range(1, num)]
            + [f"x_div_{i}" for i in range(1, num)]
        )
        colnames_y = (
            [f"y_diff_{i}" for i in range(1, num)]
            + [f"y_diff_inv_{i}" for i in range(1, num)]
            + [f"y_div_{i}" for i in range(1, num)]
        )
        features = labels[colnames_x + colnames_y]
        return features, list(labels["frame"])

    def _smooth_predictions(self, x_ball: list, y_ball: list) -> tuple[list, list]:
        is_none = [int(x is None) for x in x_ball]
        interp = 5
        counter = 0
        for num in range(interp, len(x_ball) - 1):
            if not x_ball[num] and sum(is_none[num - interp : num]) == 0 and counter < 3:
                x_ext, y_ext = self._extrapolate(x_ball[num - interp : num], y_ball[num - interp : num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num + 1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num + 1], y_ball[num + 1]))
                    if dist > 80:
                        x_ball[num + 1], y_ball[num + 1], is_none[num + 1] = None, None, 1
                counter += 1
            else:
                counter = 0
        return x_ball, y_ball

    @staticmethod
    def _extrapolate(x_coords: list, y_coords: list) -> tuple[float, float]:
        xs = list(range(len(x_coords)))
        x_ext = CubicSpline(xs, x_coords, bc_type="natural")(len(x_coords))
        y_ext = CubicSpline(xs, y_coords, bc_type="natural")(len(x_coords))
        return float(x_ext), float(y_ext)

    @staticmethod
    def _postprocess(ind_bounce: np.ndarray, preds: np.ndarray) -> list:
        filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i - 1]) != 1:
                filtered.append(ind_bounce[i])
            elif preds[ind_bounce[i]] > preds[ind_bounce[i - 1]]:
                filtered[-1] = ind_bounce[i]
        return filtered


def filter_bounces_near_players(
    bounces: list[BounceEvent],
    player_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]],
    reach_margin: float = 50.0,
) -> list[BounceEvent]:
    """Drop a candidate that lands inside (or within `reach_margin` px of) a
    detected player's box that frame.

    The CatBoost model was trained on trajectory shape alone (see this
    module's docstring), so it has no way to distinguish a real court
    bounce from a racket CONTACT - both produce the same kind of sharp
    trajectory reversal. A contact happens AT the player, not out on open
    court, so proximity to a player's detected box is a cheap signal the
    model itself never sees. Only affects frames where player boxes were
    actually computed (i.e. `--minimap`) - a frame with no box entry passes
    every candidate through untouched.
    """
    filtered = []
    for bounce in bounces:
        boxes = player_boxes_by_frame.get(bounce.frame_idx)
        near_player = boxes is not None and any(
            (x1 - reach_margin) <= bounce.x <= (x2 + reach_margin)
            and (y1 - reach_margin) <= bounce.y <= (y2 + reach_margin)
            for x1, y1, x2, y2 in boxes
        )
        if not near_player:
            filtered.append(bounce)
    return filtered


def detect_bounces_catboost(
    positions: list[TrackedPosition],
    model: CatBoostBounceDetector,
    num_frames: Optional[int] = None,
    calibration: Optional[CourtCalibration] = None,
) -> list[BounceEvent]:
    """`num_frames` should be the video's total frame count if known (main.py
    always knows it) - falls back to the highest tracked frame_idx + 1 if
    not, which undercounts whenever the trajectory doesn't reach the last
    frame."""
    if num_frames is None:
        num_frames = (max((p.frame_idx for p in positions), default=-1)) + 1

    x_ball: list = [None] * num_frames
    y_ball: list = [None] * num_frames
    for p in positions:
        x_ball[p.frame_idx] = p.x
        y_ball[p.frame_idx] = p.y

    bounce_frames = model.predict_frames(x_ball, y_ball)

    events = []
    for frame_idx in sorted(bounce_frames):
        x, y = x_ball[frame_idx], y_ball[frame_idx]
        if x is None:
            continue
        world_x = world_y = None
        if calibration is not None:
            world_x, world_y = calibration.pixel_to_world(x, y)
        events.append(BounceEvent(frame_idx=frame_idx, x=x, y=y, world_x=world_x, world_y=world_y))
    return events
