import unittest

import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.visualize.draw import CourtOverlayDrawer


def scaled_calibration(scale):
    # simple, exactly-known pixel<->world relationship: pixel = world * scale
    names = ["baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right"]
    pixel_points = {
        name: (wx * scale, wy * scale)
        for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items()
        if name in names
    }
    return CourtCalibration.from_keypoints(pixel_points)


class CourtOverlayDrawerTest(unittest.TestCase):
    def setUp(self):
        self.calibration = scaled_calibration(scale=1.0)
        self.drawer = CourtOverlayDrawer()

    def test_draw_mutates_and_returns_same_frame(self):
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        result = self.drawer.draw(frame, self.calibration)

        self.assertIs(result, frame)
        self.assertTrue((frame != 0).any())  # something got drawn

    def test_none_calibration_leaves_frame_untouched(self):
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        result = self.drawer.draw(frame, None)

        self.assertIs(result, frame)
        self.assertTrue((frame == 0).all())

    def test_draws_a_corner_marker_at_the_reprojected_pixel_location(self):
        # pixel = world * 1.0, so baseline_far_left (world (0, 0)) reprojects
        # to pixel (0, 0) - a corner marker (circle, radius 8) should light
        # up pixels right around there
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        self.drawer.draw(frame, self.calibration)

        self.assertTrue((frame[0:5, 0:5] != 0).any())

    def test_recomputes_reprojection_fresh_each_call_for_a_different_calibration(self):
        # a "panning camera" scenario: two different calibrations passed to
        # the SAME drawer instance across two calls must each draw at their
        # own reprojected locations - proving there's no stale cached state
        # from a previous call
        small_scale = scaled_calibration(scale=1.0)
        large_scale = scaled_calibration(scale=2.0)

        frame_small = np.zeros((60, 60, 3), dtype=np.uint8)
        self.drawer.draw(frame_small, small_scale)

        frame_large = np.zeros((60, 60, 3), dtype=np.uint8)
        self.drawer.draw(frame_large, large_scale)

        # baseline_near_right reprojects further out under the 2x
        # calibration than the 1x one, so the two frames' drawn pixels
        # must differ
        self.assertFalse(np.array_equal(frame_small, frame_large))


if __name__ == "__main__":
    unittest.main()
