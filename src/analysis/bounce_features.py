"""Turn a short window of ball positions centered on a candidate frame into
a fixed-length feature vector for bounce classification.

This module is imported by BOTH scripts/train_bounce_classifier.py and
src.analysis.bounce_classifier.BounceClassifier - training and inference
must always compute features the exact same way, or the model silently
degrades (same pattern as src.analysis.stroke_features).

Candidates come from a broad, low-threshold local-max-in-y scan (see
scripts/extract_bounce_candidates.py) - the same shape of event
src.analysis.bounce_detector's heuristic looks at, but scored by a learned
classifier instead of fixed thresholds, so it can pick up on the softer,
harder-to-threshold cues (how sharply the ball decelerates, how consistent
the horizontal direction is) that distinguish a real bounce from a contact
or from ordinary trajectory noise.

Every distance/speed feature is normalized by the window's own mean
frame-to-frame speed - the same "make it scale-invariant" idea
stroke_features.py uses shoulder width for, here so the same shape of dip
reads the same whether the camera is zoomed in tight or shooting a wide
broadcast angle, and whether the ball is moving fast or slow through the
window.
"""

from typing import Sequence

import numpy as np

from src.tracking.ball_tracker import TrackedPosition

FEATURE_NAMES = [
    "y_prominence_before",
    "y_prominence_after",
    "dx_before",
    "dx_after",
    "speed_ratio",
    "mean_speed",
    "window_length",
]


def extract_features(window: Sequence[TrackedPosition], center_idx: int) -> np.ndarray:
    """`window`: consecutive-ish TrackedPositions bracketing a candidate
    local-max-in-y frame at `window[center_idx]` (need at least one frame on
    each side). Returns a fixed-length vector, len(FEATURE_NAMES)."""
    if center_idx <= 0 or center_idx >= len(window) - 1:
        raise ValueError("center_idx needs at least one frame on each side")

    first, center, last = window[0], window[center_idx], window[-1]

    steps = np.array(
        [np.hypot(b.x - a.x, b.y - a.y) for a, b in zip(window, window[1:])], dtype=np.float64
    )
    mean_speed = float(steps.mean()) if len(steps) else 0.0
    scale = mean_speed + 1e-6

    y_prominence_before = (center.y - first.y) / scale
    y_prominence_after = (center.y - last.y) / scale
    dx_before = (center.x - first.x) / scale
    dx_after = (last.x - center.x) / scale

    before_steps = steps[:center_idx]
    after_steps = steps[center_idx:]
    speed_before = float(before_steps.mean()) if len(before_steps) else 0.0
    speed_after = float(after_steps.mean()) if len(after_steps) else 0.0
    speed_ratio = speed_after / (speed_before + 1e-6)

    return np.array(
        [
            y_prominence_before,
            y_prominence_after,
            dx_before,
            dx_after,
            speed_ratio,
            mean_speed,
            float(len(window)),
        ]
    )
