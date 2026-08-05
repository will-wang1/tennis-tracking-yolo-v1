import tempfile
import unittest
from pathlib import Path

from src.analysis.court_calibration import CourtCalibration

PIXEL_POINTS = [(340.0, 980.0), (1580.0, 980.0), (1400.0, 650.0), (520.0, 650.0)]
WORLD_POINTS = [(0.0, 0.0), (8.23, 0.0), (8.23, 5.485), (0.0, 5.485)]


class CourtCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.calibration = CourtCalibration.from_points(PIXEL_POINTS, WORLD_POINTS)

    def test_input_corners_round_trip(self):
        for (px, py), (wx, wy) in zip(PIXEL_POINTS, WORLD_POINTS):
            mapped_x, mapped_y = self.calibration.pixel_to_world(px, py)
            self.assertAlmostEqual(mapped_x, wx, places=3)
            self.assertAlmostEqual(mapped_y, wy, places=3)

    def test_midpoint_maps_near_world_midpoint(self):
        px = (PIXEL_POINTS[0][0] + PIXEL_POINTS[1][0]) / 2
        py = (PIXEL_POINTS[0][1] + PIXEL_POINTS[1][1]) / 2
        mapped_x, mapped_y = self.calibration.pixel_to_world(px, py)

        # not exactly the world midpoint - perspective distortion means the
        # mapping isn't affine - but should be close for a modest camera angle
        self.assertAlmostEqual(mapped_x, 4.115, delta=0.5)
        self.assertAlmostEqual(mapped_y, 0.0, delta=0.5)

    def test_pixel_distance_to_meters(self):
        distance = self.calibration.pixel_distance_to_meters(*PIXEL_POINTS[0], *PIXEL_POINTS[1])
        self.assertAlmostEqual(distance, 8.23, places=2)

    def test_requires_exactly_four_points(self):
        with self.assertRaises(ValueError):
            CourtCalibration.from_points(PIXEL_POINTS[:3], WORLD_POINTS[:3])

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            self.calibration.save(path)
            loaded = CourtCalibration.load(path)

            for (px, py) in PIXEL_POINTS:
                original = self.calibration.pixel_to_world(px, py)
                restored = loaded.pixel_to_world(px, py)
                self.assertAlmostEqual(original[0], restored[0], places=6)
                self.assertAlmostEqual(original[1], restored[1], places=6)


if __name__ == "__main__":
    unittest.main()
