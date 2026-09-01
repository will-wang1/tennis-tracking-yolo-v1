import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.velocity_bounce_detector import detect_bounces_by_velocity
from src.tracking.ball_tracker import TrackedPosition


def _identity_calibration() -> CourtCalibration:
    return CourtCalibration(homography=np.eye(3, dtype=np.float64))


def _positions_from_x(xs: list[float], y: float = 0.0) -> list[TrackedPosition]:
    return [TrackedPosition(frame_idx=i, x=x, y=y, interpolated=False) for i, x in enumerate(xs)]


class DetectBouncesByVelocityTest(unittest.TestCase):
    def test_flags_deceleration_followed_by_reacceleration_in_the_same_direction(self):
        # fast approach (100/frame), sharp drop to 5, then resumes at ~90/frame in the SAME (+x) direction - a bounce
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual([b.frame_idx for b in bounces], [1])

    def test_detects_a_bounce_right_at_the_net_line(self):
        # same bounce pattern, but at the net's world_y - unlike
        # bounce_ensemble's net exclusion, this module makes no spatial
        # trade-off, so a genuine near-net bounce should still be caught
        xs = [0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0]
        positions = [TrackedPosition(frame_idx=i, x=x, y=11.885, interpolated=False) for i, x in enumerate(xs)]
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual([b.frame_idx for b in bounces], [1])

    def test_ignores_deceleration_followed_by_reversal(self):
        # same sharp drop, but the ball then moves back the way it came - a contact, not a bounce
        positions = _positions_from_x([0.0, 100.0, 105.0, 15.0, -75.0, -165.0, -255.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(bounces, [])

    def test_ignores_deceleration_with_no_reacceleration(self):
        # sharp drop, then the ball stops completely (absorbed/blocked) rather than resuming speed
        positions = _positions_from_x([0.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(bounces, [])

    def test_ignores_a_drop_below_the_minimum_speed(self):
        positions = _positions_from_x([0.0, 0.1, 0.11, 0.111, 0.1111, 0.11111, 0.111111])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(bounces, [])

    def test_ignores_a_gentle_deceleration(self):
        positions = _positions_from_x([0.0, 100.0, 190.0, 280.0, 370.0, 460.0, 550.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(bounces, [])

    def test_skips_a_triple_spanning_an_interpolated_position(self):
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        positions[1] = TrackedPosition(frame_idx=1, x=100.0, y=0.0, interpolated=True)
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(bounces, [])

    def test_skips_a_candidate_frame_with_no_calibration(self):
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}
        del calibrations_by_frame[1]  # the candidate bounce frame itself can't be projected

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(bounces, [])

    def test_merges_nearby_flagged_frames_into_one_event(self):
        # a wobbly bounce that flags on two adjacent-ish frames should still yield one event
        positions = _positions_from_x([0.0, 100.0, 105.0, 108.0, 198.0, 288.0, 378.0, 468.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0, min_frame_gap=8)

        self.assertEqual(len(bounces), 1)

    def test_returned_event_carries_both_pixel_and_world_coordinates(self):
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(len(bounces), 1)
        bounce = bounces[0]
        self.assertEqual(bounce.x, 100.0)
        self.assertEqual(bounce.world_x, 100.0)  # identity calibration


if __name__ == "__main__":
    unittest.main()
