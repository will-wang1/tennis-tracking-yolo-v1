import tempfile
import unittest
from pathlib import Path

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration

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

    def test_full_court_reference_has_all_fourteen_named_points(self):
        self.assertEqual(len(FULL_COURT_REFERENCE_POINTS), 14)
        # doubles width 10.97m, full length 23.77m - sanity-check the extremes
        xs = [p[0] for p in FULL_COURT_REFERENCE_POINTS.values()]
        ys = [p[1] for p in FULL_COURT_REFERENCE_POINTS.values()]
        self.assertAlmostEqual(max(xs) - min(xs), 10.97, places=2)
        self.assertAlmostEqual(max(ys) - min(ys), 23.77, places=2)

    def test_from_keypoints_with_exactly_four_matches_from_points(self):
        # a simple, exactly-known pixel<->world relationship: pixel = world * 100
        names = ["baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right"]
        pixel_points = {name: (wx * 100, wy * 100) for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items() if name in names}

        calibration = CourtCalibration.from_keypoints(pixel_points)
        wx, wy = calibration.pixel_to_world(548.5, 1188.5)  # center-ish point, pixel = world*100
        self.assertAlmostEqual(wx, 5.485, places=1)
        self.assertAlmostEqual(wy, 11.885, places=1)

    def test_from_keypoints_uses_only_matched_names(self):
        # an unrelated key that doesn't appear in the world reference must
        # be silently ignored, not crash or get included
        names = ["baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right"]
        pixel_points = {name: (wx * 100, wy * 100) for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items() if name in names}
        pixel_points["not_a_real_keypoint"] = (9999.0, 9999.0)

        calibration = CourtCalibration.from_keypoints(pixel_points)
        wx, wy = calibration.pixel_to_world(0.0, 0.0)
        self.assertAlmostEqual(wx, 0.0, places=1)
        self.assertAlmostEqual(wy, 0.0, places=1)

    def test_from_keypoints_with_more_than_four_is_more_robust(self):
        # 6 correspondences with one point perturbed by noise - the RANSAC
        # fit should still recover a sane homography rather than being
        # thrown off by the one outlier the way an exact 4-point fit would
        names = [
            "baseline_far_left", "baseline_far_right", "baseline_near_left",
            "baseline_near_right", "singles_far_left", "singles_near_right",
        ]
        pixel_points = {name: (wx * 100, wy * 100) for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items() if name in names}
        pixel_points["singles_near_right"] = (
            pixel_points["singles_near_right"][0] + 500,  # way off
            pixel_points["singles_near_right"][1],
        )

        calibration = CourtCalibration.from_keypoints(pixel_points)
        wx, wy = calibration.pixel_to_world(0.0, 0.0)
        self.assertAlmostEqual(wx, 0.0, delta=0.5)
        self.assertAlmostEqual(wy, 0.0, delta=0.5)

    def test_from_keypoints_requires_minimum_points(self):
        pixel_points = {"baseline_far_left": (0.0, 0.0), "baseline_far_right": (100.0, 0.0)}
        with self.assertRaises(ValueError):
            CourtCalibration.from_keypoints(pixel_points)

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
