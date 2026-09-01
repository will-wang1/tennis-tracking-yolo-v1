import unittest

import numpy as np

from src.detection.wasb_ball_detector import WASBBallDetector, _get_affine_transform


class WasbPostprocessTest(unittest.TestCase):
    def _identity_transform(self):
        # maps heatmap pixel coords straight through unchanged, so expected
        # centroid coordinates are easy to reason about in the test itself
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def test_finds_centroid_of_a_single_blob(self):
        heatmap = np.zeros((20, 30), dtype=np.float32)
        heatmap[10:13, 15:18] = 0.9  # a compact 3x3 blob, well above threshold

        result = WASBBallDetector._postprocess(heatmap, self._identity_transform())

        self.assertIsNotNone(result)
        x, y, confidence = result
        self.assertAlmostEqual(x, 16.0, places=3)
        self.assertAlmostEqual(y, 11.0, places=3)
        self.assertAlmostEqual(confidence, 0.9, places=3)

    def test_returns_none_when_nothing_crosses_the_threshold(self):
        heatmap = np.full((20, 30), 0.1, dtype=np.float32)

        result = WASBBallDetector._postprocess(heatmap, self._identity_transform())

        self.assertIsNone(result)

    def test_ignores_a_blob_larger_than_a_plausible_ball(self):
        heatmap = np.zeros((20, 30), dtype=np.float32)
        heatmap[:, :] = 0.0
        heatmap[2:18, 2:28] = 0.8  # a huge blob - not a compact ball-sized dot

        result = WASBBallDetector._postprocess(heatmap, self._identity_transform())

        self.assertIsNone(result)

    def test_picks_the_higher_scoring_blob_when_two_are_present(self):
        heatmap = np.zeros((20, 30), dtype=np.float32)
        heatmap[2:4, 2:4] = 0.6  # weaker, smaller blob
        heatmap[10:13, 20:23] = 0.95  # stronger blob - should win

        result = WASBBallDetector._postprocess(heatmap, self._identity_transform())

        self.assertIsNotNone(result)
        x, y, _ = result
        self.assertAlmostEqual(x, 21.0, places=3)
        self.assertAlmostEqual(y, 11.0, places=3)

    def test_affine_transform_round_trips_through_inverse(self):
        center = np.array([960.0, 540.0], dtype=np.float32)
        scale = 1920.0
        output_size = (512, 288)

        forward = _get_affine_transform(center, scale, output_size)
        inverse = _get_affine_transform(center, scale, output_size, inv=True)

        original_pt = np.array([960.0, 540.0, 1.0], dtype=np.float32)
        model_space = forward @ original_pt
        back_to_original = inverse @ np.array([model_space[0], model_space[1], 1.0], dtype=np.float32)

        self.assertAlmostEqual(back_to_original[0], 960.0, places=1)
        self.assertAlmostEqual(back_to_original[1], 540.0, places=1)


if __name__ == "__main__":
    unittest.main()
