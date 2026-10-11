import unittest

import cv2
import numpy as np

from src.analysis.court_calibration import _LINE_EDGES, FULL_COURT_REFERENCE_POINTS as P
from src.analysis.court_refine import CORNERS, _calibration_from_corners, refine_calibration

TRUE = np.array([[671, 193], [1215, 193], [1900, 844], [42, 852]], float)


def _court_image():
    cal = _calibration_from_corners(TRUE)
    img = np.full((1080, 1920), 90, np.uint8)
    for a, b in _LINE_EDGES:
        pa = tuple(int(round(v)) for v in cal.world_to_pixel(*P[a]))
        pb = tuple(int(round(v)) for v in cal.world_to_pixel(*P[b]))
        cv2.line(img, pa, pb, 230, 3 if max(pa[1], pb[1]) < 400 else 6, cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


class RefineTest(unittest.TestCase):
    def test_a_fit_several_pixels_off_is_pulled_onto_the_lines(self):
        start = TRUE + np.array([[6, -4], [-5, 3], [7, 6], [-6, -5]])
        refined, record = refine_calibration(_court_image(), _calibration_from_corners(start))
        got = np.array([refined.world_to_pixel(*P[c]) for c in CORNERS])
        self.assertTrue(record["refined"])
        self.assertLess(np.abs(got - TRUE).max(), 1.5)
        self.assertGreater(record["score_after"], record["score_before"])

    def test_a_picture_with_no_lines_keeps_the_original(self):
        flat = np.full((1080, 1920), 90, np.uint8)
        cal = _calibration_from_corners(TRUE)
        refined, record = refine_calibration(flat, cal)
        self.assertFalse(record["refined"])
        np.testing.assert_allclose(refined.homography, cal.homography)


if __name__ == "__main__":
    unittest.main()
