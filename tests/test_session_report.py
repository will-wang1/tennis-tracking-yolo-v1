import unittest

import numpy as np

from src.detection.ball_detector import Detection
from src.session.report import ball_on_court, in_play_mask, runs
from src.tracking.candidate_tracker import BallTrack

FPS = 25.0


def _track(start, end):
    return BallTrack({f: Detection(x=0.0, y=0.0, confidence=0.9) for f in range(start, end + 1)})


class InPlayMaskTest(unittest.TestCase):
    def test_a_short_gap_between_two_stretches_is_bridged(self):
        mask = in_play_mask([_track(0, 49), _track(60, 99)], 100, FPS)  # 10-frame gap = 0.4s
        self.assertTrue(mask[50:60].all())

    def test_a_long_gap_is_real_dead_time(self):
        mask = in_play_mask([_track(0, 49), _track(100, 149)], 150, FPS)  # 2s gap
        self.assertFalse(mask[50:100].any())

    def test_gaps_at_the_ends_are_never_bridged(self):
        mask = in_play_mask([_track(10, 80)], 100, FPS)
        self.assertFalse(mask[:10].any())
        self.assertFalse(mask[81:].any())


class RunsTest(unittest.TestCase):
    def test_reports_half_open_ranges(self):
        self.assertEqual(runs(np.array([0, 1, 1, 0, 1], dtype=bool)), [(1, 3), (4, 5)])


class BallOnCourtTest(unittest.TestCase):
    def test_court_and_near_run_off_count_adjacent_court_does_not(self):
        self.assertTrue(ball_on_court((5.0, 12.0)))
        self.assertTrue(ball_on_court((-1.5, 24.5)))
        self.assertFalse(ball_on_court((-7.6, 10.3)))  # measured neighbouring-court track
        self.assertFalse(ball_on_court((11.5, -25.9)))  # measured far-behind track


class PeopleNotTrackedTest(unittest.TestCase):
    def test_people_metrics_are_null_not_zero_when_tracking_was_not_run(self):
        from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
        from src.analysis.impact_pipeline import ImpactAnalysis
        from src.session.report import build_session_report

        corners = {k: v for k, v in {
            "baseline_far_left": (700.0, 200.0), "baseline_far_right": (1220.0, 200.0),
            "baseline_near_right": (1900.0, 850.0), "baseline_near_left": (20.0, 850.0),
        }.items()}
        calibration = CourtCalibration.from_keypoints(corners, world_points={k: FULL_COURT_REFERENCE_POINTS[k] for k in corners})
        report = build_session_report(
            clip_name="t", fps=FPS, num_frames=250, calibration=calibration, ball_tracks=[],
            impacts=ImpactAnalysis(positions=[], impacts=[]), people=[], coverage=None,
        )
        for name in ("people_on_court", "movement_near_court", "near_court_moving_share", "contacts_attributed"):
            self.assertIsNone(report["metrics"][name]["value"], name)
            self.assertIn("not run", report["metrics"][name]["caveats"][0])
