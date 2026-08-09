"""Inference-time bounce/not_bounce scoring for a candidate trajectory window.

Wraps a scikit-learn Pipeline trained by scripts/train_bounce_classifier.py.
Call `.bounce_probability()` / `.is_bounce()` with a window of
TrackedPositions bracketing a candidate frame - the same window shape
src.analysis.bounce_features.extract_features expects. Used by
src.analysis.bounce_detector.detect_bounces_ml to discriminate real bounces
from contacts/noise among a broad candidate scan, learned from data instead
of the fixed x-reversal/ground-proximity thresholds detect_bounces uses.
"""

from pathlib import Path
from typing import Sequence

import joblib

from src.analysis.bounce_features import extract_features
from src.tracking.ball_tracker import TrackedPosition


class BounceClassifier:
    def __init__(self, model_path: str | Path):
        bundle = joblib.load(model_path)
        self.model = bundle["pipeline"]
        self._bounce_class_idx = list(self.model.classes_).index("bounce")

    def bounce_probability(self, window: Sequence[TrackedPosition], center_idx: int) -> float:
        """Probability that `window[center_idx]` is a real bounce."""
        features = extract_features(window, center_idx).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        return float(probabilities[self._bounce_class_idx])

    def is_bounce(self, window: Sequence[TrackedPosition], center_idx: int, threshold: float = 0.5) -> bool:
        return self.bounce_probability(window, center_idx) >= threshold
