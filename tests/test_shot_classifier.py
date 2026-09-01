import unittest

import numpy as np

from src.analysis.shot_classifier import ShotEventTracker, ShotPrediction, keypoints_to_features


def _all_visible_keypoints(score: float = 0.9) -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = np.linspace(0.1, 0.9, 17)  # y
    keypoints[:, 1] = np.linspace(0.2, 0.8, 17)  # x
    keypoints[:, 2] = score
    return keypoints


class KeypointsToFeaturesTest(unittest.TestCase):
    def test_drops_the_four_face_keypoints(self):
        keypoints = _all_visible_keypoints()

        features = keypoints_to_features(keypoints)

        self.assertEqual(features.shape, (26,))

    def test_preserves_y_x_order_and_values(self):
        keypoints = _all_visible_keypoints()

        features = keypoints_to_features(keypoints)

        # first surviving keypoint is index 0 (nose) - not one of the discarded face points
        self.assertAlmostEqual(features[0], keypoints[0, 0])  # y
        self.assertAlmostEqual(features[1], keypoints[0, 1])  # x

    def test_returns_none_when_a_non_face_keypoint_is_not_visible(self):
        keypoints = _all_visible_keypoints()
        keypoints[10, 2] = 0.0  # left_hip, not a discarded face point

        features = keypoints_to_features(keypoints)

        self.assertIsNone(features)

    def test_ignores_face_keypoint_visibility(self):
        keypoints = _all_visible_keypoints()
        keypoints[1, 2] = 0.0  # left_eye - already discarded regardless

        features = keypoints_to_features(keypoints)

        self.assertIsNotNone(features)
        self.assertEqual(features.shape, (26,))


class ShotEventTrackerTest(unittest.TestCase):
    def _prediction(self, label: str, confidence: float) -> ShotPrediction:
        return ShotPrediction(label=label, confidence=confidence, probabilities=np.zeros(4))

    def test_counts_a_confident_shot(self):
        tracker = ShotEventTracker()

        tracker.update(100, self._prediction("forehand", 0.99))

        self.assertEqual(tracker.counts["forehand"], 1)
        self.assertEqual(tracker.events, [(100, "forehand")])

    def test_ignores_neutral(self):
        tracker = ShotEventTracker()

        tracker.update(100, self._prediction("neutral", 0.99))

        self.assertEqual(tracker.counts, {"forehand": 0, "backhand": 0, "serve": 0})

    def test_ignores_low_confidence(self):
        tracker = ShotEventTracker()

        tracker.update(100, self._prediction("forehand", 0.5))

        self.assertEqual(tracker.counts["forehand"], 0)

    def test_debounces_a_sustained_high_confidence_run_into_one_shot(self):
        tracker = ShotEventTracker(min_frame_gap=60)

        for frame_idx in range(100, 130):  # well within one debounce window
            tracker.update(frame_idx, self._prediction("forehand", 0.99))

        self.assertEqual(tracker.counts["forehand"], 1)

    def test_counts_a_second_shot_after_the_gap_elapses(self):
        tracker = ShotEventTracker(min_frame_gap=60)

        tracker.update(0, self._prediction("forehand", 0.99))
        for frame_idx in range(1, 61):
            tracker.update(frame_idx, None)
        tracker.update(61, self._prediction("backhand", 0.99))

        self.assertEqual(tracker.counts["forehand"], 1)
        self.assertEqual(tracker.counts["backhand"], 1)


if __name__ == "__main__":
    unittest.main()
