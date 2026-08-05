import unittest
from unittest.mock import patch

import numpy as np

from src.analysis.stroke_classifier import StrokeClassifier
from src.detection.pose_detector import PersonPose


def make_pose():
    return PersonPose(
        keypoints=np.zeros((17, 2)),
        keypoint_confidence=np.ones(17),
        bbox_confidence=0.9,
        center_x=0.0,
        center_y=0.0,
    )


class FakeModel:
    """Duck-typed stand-in for a sklearn Pipeline - no real trained model
    or file needed to test StrokeClassifier's control flow."""

    def predict(self, X):
        return ["forehand"]

    def predict_proba(self, X):
        return np.array([[0.1, 0.7, 0.1, 0.1]])


class StrokeClassifierTest(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.analysis.stroke_classifier.joblib.load", return_value=FakeModel())
        self.addCleanup(patcher.stop)
        patcher.start()
        self.classifier = StrokeClassifier("unused_path.pkl", window=3)

    def test_returns_none_until_window_is_full(self):
        self.assertIsNone(self.classifier.predict(0, make_pose()))
        self.assertIsNone(self.classifier.predict(1, make_pose()))

    def test_predicts_once_window_is_full(self):
        self.classifier.predict(0, make_pose())
        self.classifier.predict(1, make_pose())
        result = self.classifier.predict(2, make_pose())

        self.assertIsNotNone(result)
        self.assertEqual(result.frame_idx, 2)
        self.assertEqual(result.label, "forehand")
        self.assertAlmostEqual(result.confidence, 0.7)

    def test_none_pose_resets_window(self):
        self.classifier.predict(0, make_pose())
        self.classifier.predict(1, make_pose())
        self.assertIsNone(self.classifier.predict(2, None))  # resets the window

        # only 1 frame since the reset - window (size 3) not full yet
        self.assertIsNone(self.classifier.predict(3, make_pose()))
        self.assertIsNone(self.classifier.predict(4, make_pose()))
        self.assertIsNotNone(self.classifier.predict(5, make_pose()))


if __name__ == "__main__":
    unittest.main()
