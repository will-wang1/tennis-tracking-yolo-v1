import unittest

import numpy as np

from src.analysis.swings import ball_near, detect_swings, peaks, wrist_travel


def _pose(wrist_xy, shoulder_xy=(50.0, 30.0), conf=0.9):
    xy = np.zeros((17, 2))
    xy[5] = (shoulder_xy[0] - 10, shoulder_xy[1])
    xy[6] = (shoulder_xy[0] + 10, shoulder_xy[1])
    xy[9] = wrist_xy
    xy[10] = wrist_xy
    return xy, np.full(17, conf)


def _swing_poses(n, at, height=100.0, shift_per_frame=(0.0, 0.0)):
    """A person whose wrist sweeps 80px across the body around frame `at`;
    the whole body can also drift (running)."""
    poses = {}
    for f in range(n):
        dx = shift_per_frame[0] * f
        wx = 30.0 + (80.0 if f >= at else 0.0) * min(1.0, max(0.0, (f - at + 3) / 6.0))
        poses[f] = _pose((wx + dx, 60.0), shoulder_xy=(50.0 + dx, 30.0))
    return poses, {f: height for f in range(n)}


class WristTravelTest(unittest.TestCase):
    def test_a_sweep_across_the_body_is_one_peak_in_player_heights(self):
        poses, heights = _swing_poses(80, at=40)
        signal = wrist_travel(poses, heights, 80)
        self.assertAlmostEqual(float(np.nanmax(signal)), 0.8, places=2)
        self.assertEqual(len(peaks(signal, 0.3, 25)), 1)

    def test_running_alone_is_not_a_swing(self):
        poses = {f: _pose((30.0 + 5 * f, 60.0), shoulder_xy=(50.0 + 5 * f, 30.0)) for f in range(80)}
        signal = wrist_travel(poses, {f: 100.0 for f in range(80)}, 80)
        self.assertEqual(peaks(signal, 0.3, 25), [])

    def test_low_confidence_wrists_are_ignored(self):
        poses = {f: _pose((30.0 + (80 if f >= 40 else 0), 60.0), conf=0.1) for f in range(80)}
        self.assertTrue(np.all(np.isnan(wrist_travel(poses, {f: 100.0 for f in range(80)}, 80))))


class PeaksTest(unittest.TestCase):
    def test_close_peaks_keep_the_strongest(self):
        s = np.zeros(100)
        s[30], s[40], s[80] = 0.4, 0.6, 0.5
        self.assertEqual(peaks(s, 0.3, 25), [40, 80])


class BallNearTest(unittest.TestCase):
    def test_near_far_and_unseen(self):
        box = {f: (100, 100, 150, 200) for f in range(50)}
        self.assertTrue(ball_near(20, box, {22: [(160.0, 150.0)]}))
        self.assertFalse(ball_near(20, box, {22: [(900.0, 900.0)]}))
        self.assertIsNone(ball_near(20, box, {}))


class DetectSwingsTest(unittest.TestCase):
    def test_end_to_end_skips_objects_and_far_balls(self):
        poses, _ = _swing_poses(80, at=40)
        rows = [[(1, 0.0, 0.0, 100.0, 100.0, 0.9), (2, 300.0, 0.0, 400.0, 100.0, 0.9)] for _ in range(80)]
        by_frame = [{1: poses[f], 2: poses[f]} for f in range(80)]
        self.assertEqual(detect_swings(rows, by_frame, {}, skip_tracklets={2}), [])  # no ball seen
        near_ball = {f: [(50.0, 50.0)] for f in range(80)}
        found = detect_swings(rows, by_frame, near_ball, skip_tracklets={2})
        self.assertEqual([(s.tracklet, s.ball_near) for s in found], [(1, True)])
        far_ball = {f: [(2000.0, 2000.0)] for f in range(80)}
        self.assertEqual(detect_swings(rows, by_frame, far_ball, skip_tracklets={2}), [])


if __name__ == "__main__":
    unittest.main()


class BallReversalTest(unittest.TestCase):
    def _ball(self, turn, toward):
        # A ball coming at the player (y moving by `toward` a frame) that turns round at `turn`.
        return {f: [(120.0, 150.0 + toward * 4.0 * (f - turn) * (1 if f <= turn else -1))] for f in range(20, 60)}

    def test_near_player_ball_down_then_up_is_a_contact_at_the_turn(self):
        from src.analysis.swings import ball_reversal
        boxes = {f: (100.0, 100.0, 150.0, 200.0) for f in range(20, 60)}
        self.assertEqual(ball_reversal(40, "near", boxes, self._ball(43, toward=1)), 43)

    def test_the_wrong_way_round_is_not_a_contact(self):
        from src.analysis.swings import ball_reversal
        boxes = {f: (100.0, 100.0, 150.0, 200.0) for f in range(20, 60)}
        self.assertIsNone(ball_reversal(40, "far", boxes, self._ball(43, toward=1)))
        self.assertIsNone(ball_reversal(40, "near", boxes, {}))
