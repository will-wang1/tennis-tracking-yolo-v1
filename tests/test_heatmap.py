import unittest

import numpy as np

from src.detection.heatmap import accumulate_heatmap


class HeatmapTest(unittest.TestCase):
    def test_single_point_lights_up_its_neighborhood(self):
        heat = accumulate_heatmap([(5, 5)], [1.0], width=10, height=10, radius=2)

        self.assertGreater(heat[5, 5], 0)
        self.assertEqual(heat[0, 0], 0)  # far corner untouched

    def test_repeated_hits_at_same_spot_accumulate(self):
        points = [(5, 5)] * 10
        weights = [1.0] * 10
        heat = accumulate_heatmap(points, weights, width=10, height=10, radius=2)

        single_hit = accumulate_heatmap([(5, 5)], [1.0], width=10, height=10, radius=2)
        self.assertAlmostEqual(heat[5, 5], 10 * single_hit[5, 5])

    def test_empty_points_produces_all_zero_heat(self):
        heat = accumulate_heatmap([], [], width=10, height=10, radius=2)
        self.assertTrue(np.all(heat == 0))

    def test_weight_scales_contribution(self):
        low = accumulate_heatmap([(5, 5)], [0.1], width=10, height=10, radius=2)
        high = accumulate_heatmap([(5, 5)], [0.9], width=10, height=10, radius=2)
        self.assertLess(low[5, 5], high[5, 5])


if __name__ == "__main__":
    unittest.main()
