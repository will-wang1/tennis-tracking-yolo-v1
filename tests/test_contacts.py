import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.session.contacts import contact_points, credit_bounces, per_player, where_from

COURT = CourtCalibration(homography=np.diag([0.01, 0.01, 1.0]))  # 100 px a metre


def _rows(n, boxes):
    """people rows: frame -> [(tracklet, x1, y1, x2, y2, conf)]"""
    return [[(t, *b, 0.9) for t, b in boxes.items()] for _ in range(n)]


class ContactPointsTest(unittest.TestCase):
    def test_feet_side_and_mirroring_for_far_players(self):
        # Near player (feet at y=22m) with the ball to the picture's right;
        # far player (feet at y=2m) with the ball to the picture's right too.
        rows = _rows(20, {1: (400, 2000, 500, 2200), 2: (400, 100, 450, 200)})
        swings = [{"frame": 10, "contact_frame": 10, "tracklet": 1, "player": 5, "confirmed_by": ["ball"]},
                  {"frame": 12, "contact_frame": 12, "tracklet": 2, "player": 6, "confirmed_by": ["ball"]},
                  {"frame": 14, "contact_frame": 14, "tracklet": 1, "player": 5, "confirmed_by": ["e2e"]}]
        balls = {10: [(560.0, 2100.0)], 12: [(480.0, 150.0)], 14: [(560.0, 2100.0)]}
        pts = contact_points(swings, rows, balls, COURT, 25.0)
        self.assertEqual(len(pts), 2)  # the e2e-only swing has no seen contact
        near, far = pts
        self.assertEqual((near["end"], near["side"], near["stood_m"]), ("near", "right", [4.5, 22.0]))
        self.assertEqual((far["end"], far["side"]), ("far", "left"))  # faces the camera: mirrored

    def test_where_they_stood(self):
        self.assertEqual(where_from(14.0), "at the net")
        self.assertEqual(where_from(23.0), "around the baseline")
        self.assertEqual(where_from(27.0), "deep behind the baseline")


class CreditTest(unittest.TestCase):
    def test_a_bounce_goes_to_the_last_shot_from_the_other_end(self):
        contacts = [{"frame": 100, "end": "near", "player": 5}, {"frame": 140, "end": "far", "player": 6}]
        bounces = [{"frame": 120, "bounce_frame": 120.4, "half": "far", "depth": "deep", "bounds": "singles", "position_m": [5, 3]},
                   {"frame": 160, "half": "near", "depth": "short", "bounds": "out", "position_m": [12, 15]}]
        credit_bounces(contacts, bounces, 25.0)
        self.assertEqual(contacts[0]["landed"]["depth"], "deep")
        self.assertEqual(contacts[1]["landed"]["bounds"], "out")
        summary = per_player([dict(c, stood="around the baseline", side="right") for c in contacts])
        self.assertEqual((summary["5"]["shots_deep"], summary["5"]["shots_in"]), (1, 1))
        self.assertEqual(summary["6"]["shots_out"], 1)


if __name__ == "__main__":
    unittest.main()
