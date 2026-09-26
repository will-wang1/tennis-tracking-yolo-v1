import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.analysis.court_calibration import _LINE_EDGES, FULL_COURT_REFERENCE_POINTS, CourtCalibration

_SPEC = importlib.util.spec_from_file_location(
    "session_report_script", Path(__file__).resolve().parent.parent / "scripts" / "session_report.py"
)
session_report_script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(session_report_script)

CORNERS = {
    "baseline_far_left": (700.0, 200.0),
    "baseline_far_right": (1220.0, 200.0),
    "baseline_near_right": (1900.0, 850.0),
    "baseline_near_left": (20.0, 850.0),
}


def _calibration(dx=0.0):
    corners = {k: (x + dx, y) for k, (x, y) in CORNERS.items()}
    return CourtCalibration.from_keypoints(corners, world_points={k: FULL_COURT_REFERENCE_POINTS[k] for k in corners})


def _court_frame(calibration):
    image = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    for a, b in _LINE_EDGES:
        pa = tuple(int(v) for v in calibration.world_to_pixel(*FULL_COURT_REFERENCE_POINTS[a]))
        pb = tuple(int(v) for v in calibration.world_to_pixel(*FULL_COURT_REFERENCE_POINTS[b]))
        cv2.line(image, pa, pb, (230, 230, 230), 5)
    return image


class FitCourtTest(unittest.TestCase):
    def test_non_court_shots_are_cutaways_and_do_not_drag_the_fit(self):
        """The full-Dingles failure: a long non-court stretch whose (carried
        forward, wrong) calibrations would pull an all-frames median off the
        lines. The court shot must still be found and fitted correctly."""
        court = _calibration()
        wrong = _calibration(dx=300.0)  # what the cache holds during the talk-to-camera shot
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
            per_frame = {}
            for f in range(40):  # court view
                writer.write(_court_frame(court))
                per_frame[f] = court
            for f in range(40, 100):  # a longer non-court shot
                writer.write(rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8))
                per_frame[f] = wrong
            writer.release()

            fitted, cutaways, shots = session_report_script.fit_court(path, per_frame, 100, 25.0)

        self.assertEqual(len(cutaways), 1)
        self.assertGreaterEqual(cutaways[0][0], 38)
        x, _ = fitted.world_to_pixel(*FULL_COURT_REFERENCE_POINTS["baseline_far_left"])
        self.assertAlmostEqual(x, 700.0, delta=2.0)  # the court's own fit, not dragged 300px

    def test_no_court_anywhere_is_a_loud_failure_not_an_empty_report(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "clip.mp4")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1920, 1080))
            for _ in range(30):
                writer.write(rng.integers(0, 255, size=(1080, 1920, 3), dtype=np.uint8))
            writer.release()
            with self.assertRaises(SystemExit):
                session_report_script.fit_court(path, {f: _calibration() for f in range(30)}, 30, 25.0)


if __name__ == "__main__":
    unittest.main()


class BestFitTest(unittest.TestCase):
    def test_a_split_detector_is_settled_by_the_picture_not_the_median(self):
        # Two clusters of per-frame fits either side of the truth-free middle:
        # the median lands between them and fits neither; scoring picks the
        # cluster the picture agrees with.
        truth = _calibration(0.0)
        wrong = _calibration(80.0)
        per_frame = {**{f: truth for f in range(0, 7)}, **{f: wrong for f in range(7, 12)}}
        gray = cv2.cvtColor(_court_frame(truth), cv2.COLOR_BGR2GRAY)
        from src.analysis.court_calibration import court_line_contrast

        fit, score = session_report_script.best_fit(per_frame, lambda c: court_line_contrast(gray, c))
        self.assertTrue(np.allclose(fit.homography, truth.homography))
        self.assertGreater(score, 50)


class SceneCutCacheTest(unittest.TestCase):
    def test_cuts_are_reused_until_the_video_changes(self):
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            video.write_bytes(b"x")
            cache = Path(tmp) / "scene_cuts.json"
            with mock.patch("src.analysis.scene_cuts.detect_scene_cuts", return_value=[5, 9]) as detect, \
                    mock.patch("src.video.io.VideoReader"):
                self.assertEqual(session_report_script.scene_cuts(str(video), cache), [5, 9])
                self.assertEqual(session_report_script.scene_cuts(str(video), cache), [5, 9])
                self.assertEqual(detect.call_count, 1)
                video.write_bytes(b"longer")
                session_report_script.scene_cuts(str(video), cache)
                self.assertEqual(detect.call_count, 2)
