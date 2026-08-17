import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.tracking.ball_tracker import TrackedPosition

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None


def pos(frame_idx, x, y):
    return TrackedPosition(frame_idx=frame_idx, x=x, y=y, interpolated=False)


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "tensorflow not installed - only needed for the LSTM bounce path")
class BounceLstmClassifierTest(unittest.TestCase):
    def _save_stub_model(self, tmp_dir: Path, window_length: int) -> Path:
        import tensorflow as tf
        from tensorflow.keras import layers

        model = tf.keras.Sequential(
            [
                layers.Input(shape=(window_length, 2)),
                layers.LSTM(4, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy")
        model_path = tmp_dir / "stub.keras"
        model.save(model_path)
        Path(str(model_path) + ".json").write_text(json.dumps({"window_length": window_length}))
        return model_path

    def test_returns_a_probability_in_zero_one(self):
        from src.analysis.bounce_lstm_classifier import BounceLstmClassifier

        with tempfile.TemporaryDirectory() as tmp:
            model_path = self._save_stub_model(Path(tmp), window_length=5)
            classifier = BounceLstmClassifier(model_path)

            window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
            probability = classifier.bounce_probability(window, center_idx=2)

            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_rejects_wrong_window_length(self):
        from src.analysis.bounce_lstm_classifier import BounceLstmClassifier

        with tempfile.TemporaryDirectory() as tmp:
            model_path = self._save_stub_model(Path(tmp), window_length=9)
            classifier = BounceLstmClassifier(model_path)

            window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
            with self.assertRaises(ValueError):
                classifier.bounce_probability(window, center_idx=2)

    def test_is_bounce_applies_threshold(self):
        from src.analysis.bounce_lstm_classifier import BounceLstmClassifier

        with tempfile.TemporaryDirectory() as tmp:
            model_path = self._save_stub_model(Path(tmp), window_length=5)
            classifier = BounceLstmClassifier(model_path)

            window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
            probability = classifier.bounce_probability(window, center_idx=2)

            self.assertEqual(classifier.is_bounce(window, 2, threshold=0.0), True)
            self.assertEqual(classifier.is_bounce(window, 2, threshold=1.01), False)
            self.assertEqual(classifier.is_bounce(window, 2, threshold=probability), True)


if __name__ == "__main__":
    unittest.main()
