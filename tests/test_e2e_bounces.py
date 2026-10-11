import unittest
from types import SimpleNamespace

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.e2e_bounces import e2e_bounces

COURT = CourtCalibration(homography=np.diag([0.01, 0.01, 1.0]))  # 100 px a metre


def track(points):
    return SimpleNamespace(detections={f: SimpleNamespace(x=x, y=y) for f, (x, y) in points.items()})


class E2EBouncesTest(unittest.TestCase):
    def test_placed_at_the_balls_lowest_point_in_the_called_half(self):
        # A near-half ball bottoming out (largest y) at frame 51, 2 px from x=500.
        near = track({f: (500.0, 2000.0 - 10 * abs(f - 51)) for f in range(40, 60)})
        far = track({f: (500.0, 500.0 - 10 * abs(f - 50)) for f in range(40, 60)})
        out = e2e_bounces([{"frame": 50, "label": "near_court_bounce", "score": 0.9}], [far, near], COURT, 25.0)
        self.assertEqual(out[0]["position_m"], [5.0, 20.0])
        self.assertEqual((out[0]["half"], out[0]["depth"], out[0]["bounds"]), ("near", "deep", "singles"))

    def test_no_ball_seen_still_counts_without_a_position_and_swings_are_ignored(self):
        events = [{"frame": 10, "label": "far_court_bounce"}, {"frame": 12, "label": "far_court_swing"}]
        out = e2e_bounces(events, [], COURT, 25.0)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["position_m"])

    def test_cutaway_frames_are_skipped(self):
        self.assertEqual(e2e_bounces([{"frame": 10, "label": "far_court_bounce"}], [], COURT, 25.0, {10}), [])


if __name__ == "__main__":
    unittest.main()


def _bounce_path(true_t, lost=0, hit_back=False, seed=0, noise=0.5):
    """Image path of a near-end bounce at (600, 900) at time true_t; `lost`
    frames missing around it; `hit_back` turns it round across the court
    after the gap (a hit, not a bounce)."""
    rng = np.random.default_rng(seed)
    path = {}
    for f in range(36, 65):
        if 0 < f - true_t + lost / 2 < lost:
            continue
        dt = f - true_t
        x = 600 + 6 * dt if dt < 0 or not hit_back else 600 - 6 * dt
        y = 900 + 14 * dt - 0.2 * dt * dt if dt < 0 else 900 - 9 * dt + 0.4 * dt * dt
        path[f] = (x + rng.normal(0, noise), y + rng.normal(0, noise))
    return path


class BetweenFramesTest(unittest.TestCase):
    def test_a_bounce_between_frames_is_placed_within_a_few_pixels(self):
        from src.analysis.e2e_bounces import bounce_between_frames

        for true_t in (50.0, 50.3, 50.7):
            x, y, t, how = bounce_between_frames(_bounce_path(true_t), 50)
            self.assertEqual(how, "path fit")
            self.assertLess(np.hypot(x - 600, y - 900), 5)
            self.assertAlmostEqual(t, true_t, delta=0.3)

    def test_the_ball_lost_across_the_bounce_is_bridged(self):
        from src.analysis.e2e_bounces import bounce_between_frames

        x, y, _, how = bounce_between_frames(_bounce_path(50.4, lost=7), 50)
        self.assertEqual(how, "path fit")
        self.assertLess(np.hypot(x - 600, y - 900), 15)

    def test_hit_back_after_the_gap_uses_the_incoming_path_at_the_call(self):
        from src.analysis.e2e_bounces import bounce_between_frames

        x, y, t, how = bounce_between_frames(_bounce_path(50.0, lost=7, hit_back=True), 50)
        self.assertEqual(how, "incoming path at the call")
        self.assertEqual(t, 50.0)
        self.assertLess(np.hypot(x - 600, y - 900), 10)
