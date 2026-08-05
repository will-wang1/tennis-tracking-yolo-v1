"""Inference-time stroke classification from a stream of striker poses.

Wraps a scikit-learn Pipeline trained by scripts/train_stroke_classifier.py.
Call `.predict()` once per frame with that frame's striker pose (or None) -
same stateful "call every frame" pattern as src.visualize.draw.TrailDrawer.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib

from src.analysis.stroke_features import extract_features
from src.detection.pose_detector import PersonPose

STROKE_LABELS = ["forehand", "backhand", "serve", "volley", "other"]


@dataclass
class StrokePrediction:
    frame_idx: int
    label: str
    confidence: float


class StrokeClassifier:
    def __init__(self, model_path: str | Path, window: int = 7):
        self.model = joblib.load(model_path)
        self.window_size = window
        self.window: deque[PersonPose] = deque(maxlen=window)

    def predict(self, frame_idx: int, striker_pose: Optional[PersonPose]) -> Optional[StrokePrediction]:
        """Call once per frame. A None pose resets the window - velocity
        features shouldn't bridge a gap where tracking/striker selection
        failed. Returns None until the window has `window_size` consecutive
        frames."""
        if striker_pose is None:
            self.window.clear()
            return None

        self.window.append(striker_pose)
        if len(self.window) < self.window_size:
            return None

        features = extract_features(list(self.window)).reshape(1, -1)
        label = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        return StrokePrediction(frame_idx=frame_idx, label=label, confidence=confidence)
