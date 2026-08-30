import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.geometric_bounce_detector import detect_bounces_geometric
from src.tracking.ball_tracker import TrackedPosition


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration(homography=np.eye(3, dtype=np.float64))


def _positions_from_y(ys: list[float]) -> list[TrackedPosition]:
    return [TrackedPosition(frame_idx=i, x=5.0, y=y, interpolated=False) for i, y in enumerate(ys)]


class DetectBouncesGeometricTest(unittest.TestCase):
    def test_flags_a_clean_local_max_in_y(self):
        # ball descending (y increasing on screen), hits a low point at frame 3, rises again
        positions = _positions_from_y([0, 10, 20, 30, 20, 10, 0])

        bounces = detect_bounces_geometric(positions, min_y_prominence=3.0, min_frame_gap=2)

        self.assertEqual([b.frame_idx for b in bounces], [3])

    def test_ignores_a_low_prominence_wobble(self):
        positions = _positions_from_y([0, 10, 20, 21, 20, 30, 40])  # tiny 1px wobble at frame 3

        bounces = detect_bounces_geometric(positions, min_y_prominence=3.0, min_frame_gap=2)

        self.assertEqual(bounces, [])

    def test_attaches_world_coordinates_when_calibration_available(self):
        positions = _positions_from_y([0, 5, 10, 15, 10, 5, 0])  # y stays within plausible court bounds
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_geometric(
            positions, calibrations_by_frame=calibrations_by_frame, min_y_prominence=3.0, min_frame_gap=2
        )

        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0].world_x, 5.0)
        self.assertEqual(bounces[0].world_y, 15.0)

    def test_drops_a_candidate_near_a_player(self):
        positions = _positions_from_y([0, 10, 20, 30, 20, 10, 0])
        player_boxes_by_frame = {3: [(0.0, 25.0, 10.0, 35.0)]}  # the bounce frame's (x=5, y=30) sits inside this box

        bounces = detect_bounces_geometric(
            positions, player_boxes_by_frame=player_boxes_by_frame, min_y_prominence=3.0, min_frame_gap=2
        )

        self.assertEqual(bounces, [])

    def test_drops_an_off_court_candidate(self):
        positions = _positions_from_y([0, 10, 20, 30, 20, 10, 0])
        # world_x for the bounce frame lands way outside the doubles court
        calibrations_by_frame = {
            i: CourtCalibration(homography=np.array([[100, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
            for i in range(7)
        }

        bounces = detect_bounces_geometric(
            positions, calibrations_by_frame=calibrations_by_frame, min_y_prominence=3.0, min_frame_gap=2
        )

        self.assertEqual(bounces, [])


if __name__ == "__main__":
    unittest.main()
