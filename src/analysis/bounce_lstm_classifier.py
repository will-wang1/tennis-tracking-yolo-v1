"""Inference-time bounce/not_bounce scoring using a trained windowed LSTM.

Counterpart to src.analysis.bounce_classifier.BounceClassifier with the
SAME call interface (duck-typed - src.analysis.bounce_detector.detect_bounces'
`classifier` parameter accepts either one), so main.py can point
--bounce-classifier at either a scripts/train_bounce_classifier.py .pkl or a
scripts/train_bounce_lstm.py .keras file interchangeably. See
scripts/train_bounce_lstm.py's module docstring for what makes this one
different: it scores the raw trajectory shape directly instead of a
hand-picked feature vector.

TensorFlow is a heavy, optional dependency - only imported here, lazily,
inside __init__, so a project that never trains/uses the LSTM path doesn't
need it installed (main.py's default --bounce-classifier path is the
RandomForest one, which doesn't touch this module at all).
"""

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from src.analysis.bounce_features import extract_window_sequence
from src.tracking.ball_tracker import TrackedPosition


class BounceLstmClassifier:
    def __init__(self, model_path: str | Path):
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path)
        meta_path = Path(str(model_path) + ".json")
        if meta_path.exists():
            self.window_length = json.loads(meta_path.read_text())["window_length"]
        else:
            # older/hand-built models without a companion .json - fall back
            # to the model's own declared input shape
            self.window_length = self.model.input_shape[1]

    def bounce_probability(self, window: Sequence[TrackedPosition], center_idx: int) -> float:
        """Probability that `window[center_idx]` is a real bounce. `window`
        must have exactly the length this model was trained on (see this
        model's companion .json, written by scripts/train_bounce_lstm.py) -
        src.analysis.bounce_detector's `_window_around` already builds
        windows of the matching default size (window=9) unless overridden."""
        if len(window) != self.window_length:
            raise ValueError(
                f"window has {len(window)} positions, this model expects {self.window_length} - "
                "pass a matching classifier_window to detect_bounces/detect_bounces_ml"
            )
        sequence = extract_window_sequence(window, center_idx)
        probability = self.model.predict(sequence[np.newaxis, ...], verbose=0)[0, 0]
        return float(probability)

    def is_bounce(self, window: Sequence[TrackedPosition], center_idx: int, threshold: float = 0.5) -> bool:
        return self.bounce_probability(window, center_idx) >= threshold
