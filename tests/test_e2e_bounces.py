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
