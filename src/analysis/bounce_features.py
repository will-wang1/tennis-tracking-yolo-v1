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


def extract_window_sequence(window: Sequence[TrackedPosition], center_idx: int) -> np.ndarray:
    """Every position in `window`, translated relative to `window[center_idx]`
    - (x - center.x, y - center.y) per frame - so a bounce/contact's shape
    reads the same regardless of where on court it happened. This is the
    raw input for a windowed sequence model (scripts/train_bounce_lstm.py,
    src.analysis.bounce_lstm_classifier), as opposed to `extract_features`'
    hand-picked summary statistics: instead of us deciding which properties
    of the shape matter (prominence, direction reversal, speed ratio), the
    model learns that from many examples of the raw shape itself.

    Deliberately left in raw pixel units, not further scaled by speed or
    frame dimensions the way `extract_features` and
    scripts/*_bounce_dataset.py's normalization does - unlike that summary
    vector, the model here sees enough of the RAW shape that scale itself
    is informative (how large a dip looks is part of what distinguishes a
    clean near-court bounce from a barely-there far-court one), so
    training data needs real examples spanning both, not a normalization
    trick that would erase the difference.

    Returns shape (len(window), 2). Requires at least one frame on each
    side of the center, same as `extract_features`.
    """
    if center_idx <= 0 or center_idx >= len(window) - 1:
        raise ValueError("center_idx needs at least one frame on each side")
    center = window[center_idx]
    return np.array([[p.x - center.x, p.y - center.y] for p in window], dtype=np.float64)
