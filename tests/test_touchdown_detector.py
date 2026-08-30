import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.touchdown_detector import (
    classify_touchdowns,
    looks_like_touchdown,
    player_reach_ratio,
)
from src.tracking.ball_tracker import TrackedPosition


def _impact(frame_idx=30, y=500.0):
    return BounceCandidate(
        frame_idx=frame_idx, t=float(frame_idx), x=900.0, y=y,
        restitution=0.5, horizontal_ratio=0.8, speed_ratio=0.7, rmse=1.0,
    )


def _positions(court_y_by_frame):
    """Positions whose identity-calibrated world y is the given value, so a
    test can state the projected court position directly."""
    return [
        TrackedPosition(frame_idx=f, x=900.0, y=court_y, interpolated=False)
        for f, court_y in sorted(court_y_by_frame.items())
    ]


def _identity_calibrations(frames):
    return {f: CourtCalibration(homography=np.eye(3, dtype=np.float64)) for f in frames}


def _approach(before_rate, after_rate, impact_frame=30, span=14, start=10.0):
    """A ball closing on the camera at `before_rate` court-metres per frame,
    then at `after_rate` once past the impact."""
    court_y = {}
    for offset in range(-span, 0):
        court_y[impact_frame + offset] = start + before_rate * offset
    for offset in range(0, span + 1):
        court_y[impact_frame + offset] = start + after_rate * offset
    return court_y


class LooksLikeTouchdownTest(unittest.TestCase):
    def test_flags_an_approach_that_abruptly_slows(self):
        self.assertTrue(looks_like_touchdown(48.0, 6.0, min_approach=3.0, min_slowdown=3.0))

    def test_ignores_a_steady_approach(self):
        self.assertFalse(looks_like_touchdown(48.0, 47.4, min_approach=3.0, min_slowdown=3.0))

    def test_flags_a_receding_ball_whose_recession_accelerates(self):
        # the inflation is symmetric: a receding ball's recession is
        # suppressed while it falls and accelerates once it lands
        self.assertTrue(looks_like_touchdown(-30.0, -54.0, min_approach=3.0, min_slowdown=3.0))

    def test_ignores_a_ball_sent_back_down_the_court(self):
        # direction along the court reversed - a player returned it
        self.assertFalse(looks_like_touchdown(8.0, -30.0, min_approach=3.0, min_slowdown=3.0))

    def test_ignores_a_ball_barely_moving_along_the_court(self):
        # the sign is noise at this magnitude, so direction means nothing
        self.assertFalse(looks_like_touchdown(0.5, -20.0, min_approach=3.0, min_slowdown=3.0))

    def test_ignores_an_approach_that_speeds_up(self):
        self.assertFalse(looks_like_touchdown(18.0, 54.0, min_approach=3.0, min_slowdown=3.0))

    def test_ignores_missing_measurements(self):
        self.assertFalse(looks_like_touchdown(None, 6.0, min_approach=3.0, min_slowdown=3.0))
        self.assertFalse(looks_like_touchdown(48.0, None, min_approach=3.0, min_slowdown=3.0))


class ClassifyTouchdownsTest(unittest.TestCase):
    def _classify(self, court_y, impact=None, **kwargs):
        impact = impact or _impact()
        positions = _positions(court_y)
        calibrations = _identity_calibrations(court_y)
        return classify_touchdowns([impact], positions, calibrations, 60.0, **kwargs)[0]

    def test_calls_a_slowing_approach_a_bounce(self):
        result = self._classify(_approach(0.8, 0.1))

        self.assertTrue(result.is_bounce)
        self.assertEqual(result.reason, "bounce")
        self.assertAlmostEqual(result.approach_before, 48.0, places=1)
        self.assertAlmostEqual(result.approach_after, 6.0, places=1)

    def test_leaves_a_steady_approach_unattributed(self):
        # nothing touched down, but nothing here says a racket did it either
        result = self._classify(_approach(0.8, 0.8))

        self.assertFalse(result.is_bounce)
        self.assertEqual(result.kind, "unknown")
        self.assertEqual(result.reason, "approach did not slow - nothing touched down")

    def test_calls_an_accelerating_recession_a_bounce(self):
        result = self._classify(_approach(-0.5, -0.9))

        self.assertTrue(result.is_bounce)
        self.assertEqual(result.reason, "bounce")

    def test_calls_a_reversed_direction_a_contact(self):
        result = self._classify(_approach(0.5, -0.9))

        self.assertFalse(result.is_bounce)
        self.assertEqual(result.kind, "contact")
        self.assertEqual(result.reason, "ball sent back down the court (a return)")

    def test_rejects_an_impact_at_the_top_of_frame(self):
        # the ball leaving or re-entering view stops and restarts the
        # trajectory, which looks exactly like an impact
        result = self._classify(_approach(0.8, 0.1), impact=_impact(y=70.0))

        self.assertFalse(result.is_bounce)
        self.assertIn("frame edge", result.reason)

    def test_reports_when_there_are_too_few_positions_to_judge(self):
        result = self._classify({30: 10.0, 31: 10.5})

        self.assertFalse(result.is_bounce)
        self.assertEqual(result.reason, "not enough tracked positions either side")

    def test_ignores_interpolated_positions(self):
        court_y = _approach(0.8, 0.1)
        positions = [
            TrackedPosition(frame_idx=f, x=900.0, y=v, interpolated=True)
            for f, v in sorted(court_y.items())
        ]

        result = classify_touchdowns(
            [_impact()], positions, _identity_calibrations(court_y), 60.0
        )[0]

        self.assertFalse(result.is_bounce)
        self.assertEqual(result.reason, "not enough tracked positions either side")

    def test_skips_frames_that_have_no_calibration(self):
        court_y = _approach(0.8, 0.1)

        result = classify_touchdowns([_impact()], _positions(court_y), {}, 60.0)[0]

        self.assertFalse(result.is_bounce)

    def test_keeps_the_original_impact_alongside_the_verdict(self):
        impact = _impact(frame_idx=30)

        result = self._classify(_approach(0.8, 0.1), impact=impact)

        self.assertIs(result.impact, impact)

    def test_classifies_several_impacts_independently(self):
        court_y = {}
        court_y.update(_approach(0.8, 0.1, impact_frame=30))
        court_y.update(_approach(-0.6, 0.9, impact_frame=80))

        results = classify_touchdowns(
            [_impact(frame_idx=30), _impact(frame_idx=80)],
            _positions(court_y),
            _identity_calibrations(court_y),
            60.0,
        )

        self.assertEqual([r.is_bounce for r in results], [True, False])


class PlayerReachRatioTest(unittest.TestCase):
    def test_reports_nothing_when_no_boxes_were_recorded(self):
        # absence of proof, not proof of absence - a missed player detection
        # must not be read as "nobody was there"
        self.assertIsNone(player_reach_ratio(100.0, 100.0, []))

    def test_is_zero_inside_a_box(self):
        self.assertEqual(player_reach_ratio(100.0, 200.0, [(80.0, 150.0, 120.0, 250.0)]), 0.0)

    def test_measures_distance_in_box_heights(self):
        # 50px from the box edge, against a 100px-tall box
        self.assertAlmostEqual(
            player_reach_ratio(170.0, 200.0, [(80.0, 150.0, 120.0, 250.0)]), 0.5
        )

    def test_the_same_pixel_gap_counts_for_less_beside_a_taller_player(self):
        near = player_reach_ratio(170.0, 200.0, [(80.0, 150.0, 120.0, 250.0)])  # 100px tall
        far = player_reach_ratio(170.0, 200.0, [(80.0, 150.0, 120.0, 350.0)])  # 200px tall

        self.assertLess(far, near)

    def test_takes_the_nearest_box(self):
        boxes = [(80.0, 150.0, 120.0, 250.0), (600.0, 150.0, 640.0, 250.0)]

        self.assertAlmostEqual(player_reach_ratio(640.0, 200.0, boxes), 0.0)

    def test_measures_diagonally_from_a_box_corner(self):
        # 30 right and 40 below the corner of a 100px-tall box -> 50px -> 0.5
        self.assertAlmostEqual(
            player_reach_ratio(150.0, 290.0, [(80.0, 150.0, 120.0, 250.0)]), 0.5
        )


class PlayerReachGateTest(unittest.TestCase):
    """A "return" verdict claims a racket sent the ball back, so it needs a
    racket in reach. Withholding that verdict is the only thing player boxes
    are allowed to do - see classify_touchdowns."""

    def setUp(self):
        self.court_y = _approach(0.5, -0.9)  # direction reverses at the impact
        self.impact = _impact()  # at x=900, y=500

    def _classify(self, boxes, **kwargs):
        return classify_touchdowns(
            [self.impact],
            _positions(self.court_y),
            _identity_calibrations(self.court_y),
            60.0,
            player_boxes_by_frame=boxes,
            **kwargs,
        )[0]

    def _box_at(self, gap_px, height=100.0):
        """A player box `gap_px` to the left of the impact."""
        right = 900.0 - gap_px
        return {self.impact.frame_idx: [(right - 40.0, 450.0, right, 450.0 + height)]}

    def test_a_reversal_beside_a_player_is_a_contact(self):
        result = self._classify(self._box_at(20.0), max_reach_ratio=0.6)

        self.assertEqual(result.kind, "contact")
        self.assertAlmostEqual(result.player_reach, 0.2)

    def test_a_reversal_out_of_everyones_reach_is_a_bounce(self):
        # the dropshot case: only the court was there to hit the ball, so the
        # projected reversal must be the height term, not a return
        result = self._classify(self._box_at(200.0), max_reach_ratio=0.6)

        self.assertEqual(result.kind, "bounce")
        self.assertIn("out of every player", result.reason)

    def test_a_far_player_who_is_simply_small_still_counts_as_in_reach(self):
        # 80px away is 0.8 box heights from a 100px player and only 0.4 from
        # a 200px one - the ratio, not the pixel gap, is what decides
        self.assertEqual(self._classify(self._box_at(80.0, height=100.0)).kind, "bounce")
        self.assertEqual(self._classify(self._box_at(80.0, height=200.0)).kind, "contact")

    def test_no_boxes_recorded_leaves_the_contact_verdict_alone(self):
        result = self._classify({self.impact.frame_idx: []})

        self.assertEqual(result.kind, "contact")
        self.assertIsNone(result.player_reach)

    def test_passing_no_player_boxes_at_all_leaves_the_contact_verdict_alone(self):
        result = self._classify(None)

        self.assertEqual(result.kind, "contact")

    def test_a_bounce_at_a_players_feet_is_still_a_bounce(self):
        # proximity is necessary evidence for a contact, never sufficient -
        # on the US Open clip a hand-labelled bounce lands 0.02 box heights
        # from a player
        court_y = _approach(0.8, 0.1)  # slows without reversing
        result = classify_touchdowns(
            [self.impact],
            _positions(court_y),
            _identity_calibrations(court_y),
            60.0,
            player_boxes_by_frame={self.impact.frame_idx: [(880.0, 480.0, 920.0, 580.0)]},
        )[0]

        self.assertEqual(result.kind, "bounce")

    def test_a_reversal_out_of_reach_that_does_not_slow_is_unattributed(self):
        # receding, then closing: no racket could have done it, and a landing
        # never reverses this way either
        court_y = _approach(-0.5, 0.9)
        result = classify_touchdowns(
            [self.impact],
            _positions(court_y),
            _identity_calibrations(court_y),
            60.0,
            player_boxes_by_frame=self._box_at(200.0),
        )[0]

        self.assertEqual(result.kind, "unknown")
        self.assertEqual(result.reason, "direction reversed with no player in reach")

    def test_an_impact_at_the_frame_edge_stays_unattributed_whoever_is_near(self):
        result = classify_touchdowns(
            [_impact(y=50.0)],
            _positions(self.court_y),
            _identity_calibrations(self.court_y),
            60.0,
            player_boxes_by_frame=self._box_at(20.0),
        )[0]

        self.assertEqual(result.kind, "unknown")


class LooksLikeTouchdownReversalTest(unittest.TestCase):
    def test_a_reversal_is_refused_by_default(self):
        self.assertFalse(looks_like_touchdown(8.0, -13.0, 3.0, 3.0))

    def test_a_reversal_is_accepted_when_reversal_is_allowed(self):
        self.assertTrue(looks_like_touchdown(8.0, -13.0, 3.0, 3.0, allow_reversal=True))

    def test_allowing_reversal_does_not_waive_the_slowdown_test(self):
        self.assertFalse(looks_like_touchdown(-8.0, 13.0, 3.0, 3.0, allow_reversal=True))

    def test_allowing_reversal_does_not_waive_the_minimum_approach(self):
        self.assertFalse(looks_like_touchdown(1.0, -13.0, 3.0, 3.0, allow_reversal=True))


if __name__ == "__main__":
    unittest.main()
