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
