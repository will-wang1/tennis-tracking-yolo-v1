"""Turn a per-frame stream of raw YOLO detections into a smooth trajectory.

Even a well fine-tuned detector will miss the ball on some frames (motion
blur, occlusion by a player, low contrast against the court) and will
occasionally fire on the wrong thing (a shoe, a line, the net). Two cheap
post-processing passes clean that up without touching the model:

1. Outlier rejection - a detection that implies the ball teleported further
   than physically possible since the last accepted point is almost always
   a false positive, so it's discarded rather than trusted.
2. Interpolation - short gaps (a few missed frames) are filled in by linear
   interpolation between the surrounding accepted points, which is what the
   real trajectory would have looked like anyway. Gaps longer than
   `max_interpolation_gap` are left as missing rather than papered over,
   since a straight line is a bad guess once the ball may have bounced or
   been hit in between.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.detection.ball_detector import Detection


@dataclass
class TrackedPosition:
    frame_idx: int
    x: float
    y: float
    interpolated: bool


class BallTracker:
    def __init__(
        self,
        max_pixels_per_frame: float = 150.0,
        max_interpolation_gap: int = 8,
    ):
        self.max_pixels_per_frame = max_pixels_per_frame
        self.max_interpolation_gap = max_interpolation_gap

    def _reject_outliers(
        self, detections: Sequence[Optional[Detection]]
    ) -> list[Optional[Detection]]:
        cleaned: list[Optional[Detection]] = []
        last: Optional[Detection] = None
        last_idx: Optional[int] = None
        for idx, det in enumerate(detections):
            if det is not None and last is not None:
                jump = np.hypot(det.x - last.x, det.y - last.y)
                elapsed = idx - last_idx
                if jump > self.max_pixels_per_frame * elapsed:
                    det = None
            cleaned.append(det)
            if det is not None:
                last = det
                last_idx = idx
        return cleaned

    def track(self, detections: Sequence[Optional[Detection]]) -> list[TrackedPosition]:
        """`detections[i]` is the raw detector output for frame i, or None."""
        cleaned = self._reject_outliers(detections)

        xs = [d.x if d is not None else np.nan for d in cleaned]
        ys = [d.y if d is not None else np.nan for d in cleaned]
        df = pd.DataFrame({"x": xs, "y": ys})
        was_missing = df["x"].isna()

        # pandas' own `limit=` fills the first N points of an over-long gap
        # rather than skipping the whole gap, so gap length is checked by
        # hand: only runs of missing frames short enough to trust get
        # interpolated at all.
        interpolated_df = df.interpolate(method="linear", limit_area="inside")
        gap_id = (~was_missing).cumsum()
        gap_length = was_missing.groupby(gap_id).transform("sum")
        fillable = was_missing & (gap_length <= self.max_interpolation_gap)
        df.loc[fillable, ["x", "y"]] = interpolated_df.loc[fillable, ["x", "y"]]

        positions = []
        for i, row in df.iterrows():
            if pd.isna(row["x"]):
                continue
            positions.append(
                TrackedPosition(
                    frame_idx=i,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    interpolated=bool(was_missing[i]),
                )
            )
        return positions
