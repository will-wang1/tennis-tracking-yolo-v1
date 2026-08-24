"""Turn a per-frame stream of raw YOLO detections into a smooth trajectory.

Even a well fine-tuned detector will miss the ball on some frames (motion
blur, occlusion by a player, low contrast against the court) and will
occasionally fire on the wrong thing (a shoe, a line, the net, a broadcast
graphic). Three cheap post-processing passes clean that up without touching
the model:

1. Outlier rejection - a detection that implies the ball teleported further
   than physically possible since the last accepted point is almost always
   a false positive, so it's discarded rather than trusted.
2. Static lock-on rejection - a false positive on something fixed on screen
   (a graphic, a line, the net cord) tends to re-detect at roughly the same
   spot frame after frame - not pixel-identical, but confined to a small
   area, which outlier rejection alone can't catch since each individual
   step is small. A ball in play is essentially never confined like that
   for long, so a run of accepted detections that wanders within a tight
   radius is treated as a lock-on and discarded rather than tracked.
3. Interpolation - short gaps (a few missed frames) are filled in by linear
   interpolation between the surrounding accepted points, which is what the
   real trajectory would have looked like anyway. Gaps longer than
   `max_interpolation_gap` are left as missing rather than papered over,
   since a straight line is a bad guess once the ball may have bounced or
   been hit in between.
4. Smoothing - a Savitzky-Golay filter over each contiguous run of frames
   (real + interpolated), which removes single-pixel-scale jitter while
   still tracking the trajectory's actual curvature (unlike a moving
   average, it doesn't lag behind or flatten a genuine direction change).
   This matters most for a heatmap-based detector like TrackNet: its
   reported position comes from a Hough-circle center on a 640x360
   heatmap, then scaled back up to the source resolution - on a 1080p
   video that's a 3x scale-up, so even 1px of heatmap-level quantization
   noise becomes ~3px of visible on-screen jitter frame to frame, on an
   otherwise straight-line shot.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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
        static_lockon_frames: int = 10,
        static_lockon_radius: float = 20.0,
        smoothing_window: int = 9,
        smoothing_polyorder: int = 2,
    ):
        self.max_pixels_per_frame = max_pixels_per_frame
        self.max_interpolation_gap = max_interpolation_gap
        self.static_lockon_frames = static_lockon_frames
        self.static_lockon_radius = static_lockon_radius
        if smoothing_window >= 3 and smoothing_window % 2 == 0:
            smoothing_window += 1  # savgol_filter requires an odd window
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder

    def _reject_outliers(
        self, detections: Sequence[Optional[Detection]]
    ) -> list[Optional[Detection]]:
        cleaned: list[Optional[Detection]] = []
        # (frame_idx, detection) for everything accepted so far, oldest first
        accepted: list[tuple[int, Detection]] = []

        for idx, det in enumerate(detections):
            if det is not None and accepted:
                last_idx, last = accepted[-1]
                jump = np.hypot(det.x - last.x, det.y - last.y)
                elapsed = idx - last_idx
                if jump > self.max_pixels_per_frame * elapsed:
                    det = None

            if det is not None and len(accepted) >= self.static_lockon_frames - 1:
                window = accepted[-(self.static_lockon_frames - 1) :] + [(idx, det)]
                xs = [d.x for _, d in window]
                ys = [d.y for _, d in window]
                span = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
                if span <= self.static_lockon_radius:
                    det = None

            cleaned.append(det)
            if det is not None:
                accepted.append((idx, det))
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
        return self._smooth(positions)

    def _smooth(self, positions: list[TrackedPosition]) -> list[TrackedPosition]:
        """Savitzky-Golay filter within each contiguous (no-gap) run of
        frames - smoothing across a gap would blend two positions that
        aren't actually temporally adjacent trajectory samples."""
        if self.smoothing_window < 3:
            return positions

        smoothed: list[TrackedPosition] = []
        run: list[TrackedPosition] = []

        def flush(run: list[TrackedPosition]) -> None:
            if len(run) < self.smoothing_window:
                smoothed.extend(run)
                return
            polyorder = min(self.smoothing_polyorder, self.smoothing_window - 1)
            xs = savgol_filter([p.x for p in run], self.smoothing_window, polyorder, mode="interp")
            ys = savgol_filter([p.y for p in run], self.smoothing_window, polyorder, mode="interp")
            for p, x, y in zip(run, xs, ys):
                smoothed.append(TrackedPosition(frame_idx=p.frame_idx, x=float(x), y=float(y), interpolated=p.interpolated))

        for p in positions:
            if run and p.frame_idx != run[-1].frame_idx + 1:
                flush(run)
                run = []
            run.append(p)
        flush(run)
        return smoothed
