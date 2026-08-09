import unittest

from src.analysis.bounce_features import FEATURE_NAMES, extract_features
from src.tracking.ball_tracker import TrackedPosition


def pos(frame_idx, x, y):
    return TrackedPosition(frame_idx=frame_idx, x=x, y=y, interpolated=False)


class ExtractFeaturesTest(unittest.TestCase):
    def test_returns_one_value_per_feature_name(self):
        window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
        features = extract_features(window, center_idx=2)

        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_rejects_center_at_window_edge(self):
        window = [pos(0, 0, 0), pos(1, 10, 50)]
        with self.assertRaises(ValueError):
            extract_features(window, center_idx=0)
        with self.assertRaises(ValueError):
            extract_features(window, center_idx=1)

    def test_translation_invariant(self):
        # a bounce shape shifted by a constant offset in both x and y should
        # produce the exact same feature vector - features must describe
        # SHAPE, not absolute screen position
        window_a = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
        window_b = [pos(i, p.x + 500, p.y + 300) for i, p in enumerate(window_a)]

        features_a = extract_features(window_a, center_idx=2)
        features_b = extract_features(window_b, center_idx=2)

        for a, b in zip(features_a, features_b):
            self.assertAlmostEqual(a, b, places=6)

    def test_v_shape_has_positive_y_prominence_both_sides(self):
        # a real bounce: y rises then falls back down around the center
        window = [pos(0, 0, 0), pos(1, 10, 50), pos(2, 20, 100), pos(3, 30, 50), pos(4, 40, 0)]
        features = extract_features(window, center_idx=2)
        by_name = dict(zip(FEATURE_NAMES, features))

        self.assertGreater(by_name["y_prominence_before"], 0)
        self.assertGreater(by_name["y_prominence_after"], 0)

    def test_direction_reversal_shows_opposite_signed_dx(self):
        # ball moving right then redirected left across the candidate - a
        # contact shape, not a bounce continuing in one direction
        window = [pos(0, 0, 100), pos(1, 20, 150), pos(2, 40, 200), pos(3, 20, 150), pos(4, 0, 100)]
        features = extract_features(window, center_idx=2)
        by_name = dict(zip(FEATURE_NAMES, features))

        self.assertGreater(by_name["dx_before"], 0)
        self.assertLess(by_name["dx_after"], 0)


if __name__ == "__main__":
    unittest.main()
