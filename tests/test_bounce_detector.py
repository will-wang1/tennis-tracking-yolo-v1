import unittest

from src.analysis.bounce_detector import detect_bounces, find_trajectory_breakpoints
from src.tracking.ball_tracker import TrackedPosition


def pos(frame_idx, x, y):
    return TrackedPosition(frame_idx=frame_idx, x=x, y=y, interpolated=False)


class FindTrajectoryBreakpointsTest(unittest.TestCase):
    def test_finds_extrema_regardless_of_horizontal_direction(self):
        # this is a horizontal-reversal shape that detect_bounces would
        # reject as a contact - breakpoints don't care, both count as a
        # shot boundary
        positions = [pos(0, 0, 0), pos(1, 50, 50), pos(2, -50, 0)]
        self.assertEqual(find_trajectory_breakpoints(positions, min_y_prominence=5.0, min_frame_gap=1), [1])

    def test_ignores_non_adjacent_frames(self):
        positions = [pos(9, 0, 50), pos(10, 0, 100), pos(30, 0, 20), pos(31, 0, 10)]
        self.assertEqual(find_trajectory_breakpoints(positions, min_y_prominence=5.0, min_frame_gap=1), [])

    def test_filters_below_prominence(self):
        positions = [pos(0, 0, 0), pos(1, 0, 2), pos(2, 0, 0)]
        self.assertEqual(find_trajectory_breakpoints(positions, min_y_prominence=5.0, min_frame_gap=1), [])


class BounceDetectorTest(unittest.TestCase):
    def test_v_shape_is_a_bounce(self):
        # ball falls (y increasing) then rises (y decreasing) - a bounce
        positions = [pos(0, 0, 0), pos(1, 0, 50), pos(2, 0, 100), pos(3, 0, 50), pos(4, 0, 0)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0].frame_idx, 2)
        self.assertEqual(bounces[0].y, 100)

    def test_inverted_v_apex_is_not_a_bounce(self):
        # ball rises (y decreasing) then falls (y increasing) - a shot apex, not a bounce
        positions = [pos(0, 0, 100), pos(1, 0, 50), pos(2, 0, 0), pos(3, 0, 50), pos(4, 0, 100)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(bounces, [])

    def test_w_shape_is_two_bounces(self):
        positions = [
            pos(0, 0, 0),
            pos(1, 0, 100),  # bounce 1
            pos(2, 0, 0),
            pos(3, 0, 100),  # bounce 2
            pos(4, 0, 0),
        ]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual([b.frame_idx for b in bounces], [1, 3])

    def test_monotonic_trajectory_has_no_bounce(self):
        positions = [pos(i, 0, i * 10) for i in range(5)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(bounces, [])

    def test_sub_prominence_wiggle_is_filtered(self):
        # y bumps up by 2px then back down - noise, not a real bounce
        positions = [pos(0, 0, 0), pos(1, 0, 2), pos(2, 0, 0)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(bounces, [])

    def test_nearby_extrema_are_merged_keeping_most_prominent(self):
        positions = [
            pos(0, 0, 0),
            pos(1, 0, 90),
            pos(2, 0, 80),
            pos(3, 0, 100),  # true peak, close to the frame-1 candidate
            pos(4, 0, 0),
        ]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=5)

        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0].frame_idx, 3)
        self.assertEqual(bounces[0].y, 100)

    def test_does_not_compare_across_a_real_tracking_gap(self):
        # frames 10 and 30 are NOT temporal neighbors - a 19-frame gap sits
        # between them (e.g. the ball was lost and re-acquired far away).
        # Comparing them as if adjacent would fabricate a bounce out of two
        # physically unrelated points.
        positions = [pos(9, 0, 50), pos(10, 0, 100), pos(30, 0, 20), pos(31, 0, 10)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(bounces, [])

    def test_default_prominence_ignores_realistic_frame_to_frame_noise(self):
        # a 10px local peak is bigger than the OLD 5px default (which would
        # have flagged this as a bounce) but well within the noise band of
        # real footage, where the ball moves ~9px/frame at the median even
        # mid-flight. The default threshold needs to sit above that noise
        # floor or ordinary motion gets mistaken for bounces.
        positions = [pos(0, 0, 0), pos(1, 0, 12), pos(2, 0, 2)]
        bounces = detect_bounces(positions)  # default min_y_prominence

        self.assertEqual(bounces, [])

    def test_horizontal_reversal_is_a_contact_not_a_bounce(self):
        # ball approaches (x increasing) then gets redirected back the way
        # it came (x decreasing) at the same moment y reverses - a racket
        # contact, not the ball hitting the court.
        positions = [pos(0, 0, 0), pos(1, 50, 50), pos(2, -50, 0)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(bounces, [])

    def test_continued_horizontal_direction_is_still_a_bounce(self):
        # x keeps increasing on both sides - the ball keeps travelling the
        # same way, just redirected vertically by the court. A real bounce.
        positions = [pos(0, 0, 0), pos(1, 50, 50), pos(2, 100, 0)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1)

        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0].frame_idx, 1)

    def test_small_x_reversal_is_not_treated_as_a_reversal(self):
        # a near-vertical shot (lob) has tiny, noisy dx on both sides - not
        # a meaningful horizontal reversal either way, so it shouldn't be
        # rejected as a contact.
        positions = [pos(0, 0, 0), pos(1, 2, 50), pos(2, -2, 0)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1, min_x_reversal=15.0)

        self.assertEqual(len(bounces), 1)

    def test_candidate_far_above_ground_reference_is_rejected(self):
        # local max in y, no horizontal reversal - but it's 200px above the
        # nearby player's feet, i.e. at racket height, not the court.
        positions = [pos(0, 0, 500), pos(1, 0, 550), pos(2, 0, 500)]
        ground_y_by_frame = {1: 750.0}  # player's feet are much lower on screen
        bounces = detect_bounces(
            positions, min_y_prominence=5.0, min_frame_gap=1,
            ground_y_by_frame=ground_y_by_frame, max_height_above_ground=60.0,
        )

        self.assertEqual(bounces, [])

    def test_candidate_near_ground_reference_is_kept(self):
        positions = [pos(0, 0, 700), pos(1, 0, 750), pos(2, 0, 700)]
        ground_y_by_frame = {1: 760.0}  # close to the candidate's own y
        bounces = detect_bounces(
            positions, min_y_prominence=5.0, min_frame_gap=1,
            ground_y_by_frame=ground_y_by_frame, max_height_above_ground=60.0,
        )

        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0].frame_idx, 1)

    def test_missing_ground_reference_for_frame_falls_back_to_trajectory_only(self):
        # no entry for frame 1 in ground_y_by_frame - should still be
        # evaluated by the trajectory-only signals rather than rejected.
        positions = [pos(0, 0, 500), pos(1, 0, 550), pos(2, 0, 500)]
        bounces = detect_bounces(
            positions, min_y_prominence=5.0, min_frame_gap=1,
            ground_y_by_frame={99: 900.0},
        )

        self.assertEqual(len(bounces), 1)

    def test_populates_world_coordinates_when_calibration_given(self):
        from src.analysis.court_calibration import CourtCalibration

        pixel_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        world_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        calibration = CourtCalibration.from_points(pixel_points, world_points)

        positions = [pos(0, 50, 0), pos(1, 50, 100), pos(2, 50, 0)]
        bounces = detect_bounces(positions, min_y_prominence=5.0, min_frame_gap=1, calibration=calibration)

        self.assertEqual(len(bounces), 1)
        self.assertIsNotNone(bounces[0].world_x)
        self.assertIsNotNone(bounces[0].world_y)
        self.assertAlmostEqual(bounces[0].world_x, 5.0, places=3)
        self.assertAlmostEqual(bounces[0].world_y, 10.0, places=3)


if __name__ == "__main__":
    unittest.main()
