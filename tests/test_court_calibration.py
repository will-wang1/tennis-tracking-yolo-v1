import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration

PIXEL_POINTS = [(340.0, 980.0), (1580.0, 980.0), (1400.0, 650.0), (520.0, 650.0)]
WORLD_POINTS = [(0.0, 0.0), (8.23, 0.0), (8.23, 5.485), (0.0, 5.485)]

# The court's four outer corners as a typical behind-the-baseline broadcast
# camera sees them on a 1920x1080 frame: a trapezoid, far baseline narrower
# than the near one. Covers ~27% of frame, matching the ~25.6% median
# measured across real footage.
REALISTIC_KEYPOINTS = {
    "baseline_far_left": (700.0, 300.0),
    "baseline_far_right": (1220.0, 300.0),
    "baseline_near_right": (1570.0, 950.0),
    "baseline_near_left": (350.0, 950.0),
}


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

    def test_unfittable_keypoints_raise_rather_than_return_none(self):
        # cv2.findHomography returns None (not a matrix) when RANSAC finds no
        # consensus - five exactly-collinear points here. This crashed a real
        # cache build partway through a clip with an AttributeError on None;
        # callers handle ValueError, so that is what it must raise.
        collinear = {
            name: (100.0 + 10 * i, 100.0 + 10 * i)
            for i, name in enumerate(list(FULL_COURT_REFERENCE_POINTS)[:5])
        }
        with self.assertRaises(ValueError):
            CourtCalibration.from_keypoints(collinear)

    def test_a_normal_broadcast_view_is_plausible(self):
        calibration = CourtCalibration.from_keypoints(REALISTIC_KEYPOINTS)
        self.assertTrue(calibration.is_plausible_view(1920, 1080))

    def test_a_near_degenerate_fit_is_rejected(self):
        # All four points nearly collinear - the court plane seen edge-on,
        # which throws the reprojected corners towards infinity. This is the
        # shape of the real failure: on the Alcaraz-Djokovic clip a replay
        # shot produced a fit reprojecting court corners 260,000px away.
        collapsed = {
            "baseline_far_left": (900.0, 500.0),
            "baseline_far_right": (1000.0, 500.4),
            "baseline_near_right": (1100.0, 500.8),
            "baseline_near_left": (1200.0, 501.2),
        }
        calibration = CourtCalibration.from_keypoints(collapsed)
        self.assertFalse(calibration.is_plausible_view(1920, 1080))

    def test_a_court_reprojecting_to_a_sliver_is_rejected(self):
        # A fit that maps the whole court into a few thousand square pixels.
        # Measured margin on real footage: the smallest genuine court view
        # covered 6.1% of frame, slivers like this covered 0.10-0.56%.
        sliver = {
            "baseline_far_left": (900.0, 500.0),
            "baseline_far_right": (930.0, 500.0),
            "baseline_near_right": (930.0, 530.0),
            "baseline_near_left": (900.0, 530.0),
        }
        calibration = CourtCalibration.from_keypoints(sliver)
        self.assertFalse(calibration.is_plausible_view(1920, 1080))

    def test_plausibility_is_relative_to_frame_size(self):
        # The same homography judged against a much smaller frame: the court
        # now covers a large share of it, so an area rule expressed in
        # absolute pixels would flip the verdict. It must not.
        calibration = CourtCalibration.from_keypoints(REALISTIC_KEYPOINTS)
        self.assertTrue(calibration.is_plausible_view(1920, 1080))
        self.assertTrue(calibration.is_plausible_view(960, 540))

    def test_a_singular_homography_is_rejected_rather_than_raising(self):
        calibration = CourtCalibration(homography=np.zeros((3, 3), dtype=np.float64))
        self.assertFalse(calibration.is_plausible_view(1920, 1080))

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
