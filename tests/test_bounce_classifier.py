import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from src.analysis.bounce_classifier import BounceClassifier
from src.analysis.bounce_features import FEATURE_NAMES
from src.tracking.ball_tracker import TrackedPosition


def pos(frame_idx, x, y):
    return TrackedPosition(frame_idx=frame_idx, x=x, y=y, interpolated=False)


def _save_dummy_model(path: Path, constant_label: str) -> None:
    # a DummyClassifier that always predicts one class with probability 1 -
    # enough to test BounceClassifier's wiring (class-index lookup, feature
    # extraction call) without needing a real trained model in this test
    X = np.zeros((4, len(FEATURE_NAMES)))
    y = np.array(["bounce", "not_bounce", "bounce", "not_bounce"])
    model = DummyClassifier(strategy="constant", constant=constant_label)
    model.fit(X, y)
    joblib.dump({"pipeline": model, "feature_names": FEATURE_NAMES}, path)


class BounceClassifierTest(unittest.TestCase):
    def test_bounce_probability_is_one_for_always_bounce_model(self):
        window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            _save_dummy_model(model_path, "bounce")
            classifier = BounceClassifier(model_path)

            self.assertAlmostEqual(classifier.bounce_probability(window, center_idx=2), 1.0)
            self.assertTrue(classifier.is_bounce(window, center_idx=2))

    def test_bounce_probability_is_zero_for_always_not_bounce_model(self):
        window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            _save_dummy_model(model_path, "not_bounce")
            classifier = BounceClassifier(model_path)

            self.assertAlmostEqual(classifier.bounce_probability(window, center_idx=2), 0.0)
            self.assertFalse(classifier.is_bounce(window, center_idx=2))

    def test_threshold_controls_acceptance(self):
        window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.pkl"
            _save_dummy_model(model_path, "bounce")
            classifier = BounceClassifier(model_path)

            self.assertTrue(classifier.is_bounce(window, center_idx=2, threshold=0.99))
            self.assertFalse(classifier.is_bounce(window, center_idx=2, threshold=1.01))


if __name__ == "__main__":
    unittest.main()
