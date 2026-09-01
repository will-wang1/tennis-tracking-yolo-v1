import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.parabolic_bounce_detector import (
    _fit_arc,
    detect_bounces_parabolic,
    find_bounce_candidates,
    find_impacts,
)
from src.tracking.ball_tracker import TrackedPosition


def _impact_trajectory(
    impact_frame: int = 20,
    x_impact: float = 500.0,
    y_impact: float = 800.0,
    vx_in: float = 6.0,
    vy_in: float = 8.0,
    gravity: float = 1.0,
    restitution: float = 0.7,
    horizontal_factor: float = 0.8,
    span: int = 10,
) -> list[TrackedPosition]:
    """Two exact free-flight arcs meeting at `impact_frame`, in the pixel
    conventions the detector works in: y grows downward, so a positive
    `vy_in` is the ball descending toward the court and positive `gravity`
    curves it further down.

    `restitution` is how much of the incoming vertical speed comes back out
    (a court gives back less than it received; a racket can give back much
    more), and `horizontal_factor` scales the horizontal velocity through
    the impact - below 1 for the friction of a real bounce, above 1 or
    negative for a racket driving or turning the ball.
    """
    positions = []
    vx_out = vx_in * horizontal_factor
    vy_out = -vy_in * restitution
    for offset in range(-span, span + 1):
        if offset < 0:
            x = x_impact + vx_in * offset
            y = y_impact + vy_in * offset + 0.5 * gravity * offset**2
        else:
            x = x_impact + vx_out * offset
            y = y_impact + vy_out * offset + 0.5 * gravity * offset**2
        positions.append(
            TrackedPosition(frame_idx=impact_frame + offset, x=x, y=y, interpolated=False)
        )
    return positions


def _single_flight(
    start_frame: int = 10,
    count: int = 21,
    x0: float = 400.0,
    y0: float = 300.0,
    vx: float = 6.0,
    vy: float = -8.0,
    gravity: float = 1.0,
) -> list[TrackedPosition]:
    """One uninterrupted parabola - a ball simply flying, rising over the
    net and falling again with no impact anywhere in the window."""
    return [
        TrackedPosition(
            frame_idx=start_frame + i,
            x=x0 + vx * i,
            y=y0 + vy * i + 0.5 * gravity * i**2,
            interpolated=False,
        )
        for i in range(count)
    ]


class FindBounceCandidatesTest(unittest.TestCase):
    def test_finds_a_clean_bounce_and_measures_it(self):
        positions = _impact_trajectory(restitution=0.7, horizontal_factor=0.8)

        candidates = find_bounce_candidates(positions)

        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.frame_idx, 20)
        self.assertAlmostEqual(candidate.t, 20.0, places=1)
        self.assertAlmostEqual(candidate.x, 500.0, delta=2.0)
        self.assertAlmostEqual(candidate.y, 800.0, delta=2.0)
        self.assertAlmostEqual(candidate.restitution, 0.7, delta=0.05)
        self.assertAlmostEqual(candidate.horizontal_ratio, 0.8, delta=0.05)

    def test_ignores_uninterrupted_flight_over_the_net(self):
        # the apex here is a local MIN in screen y - the ball rising over
        # the net - which no amount of thresholding should turn into a bounce
        candidates = find_bounce_candidates(_single_flight())

        self.assertEqual(candidates, [])

    def test_ignores_a_descending_ball_that_never_rises_again(self):
        candidates = find_bounce_candidates(_single_flight(vy=2.0))

        self.assertEqual(candidates, [])

    def test_rejects_a_contact_that_returns_more_vertical_speed_than_it_received(self):
        # a racket lifting the ball away faster than it arrived: no passive
        # surface can do this, whatever the rest of the trajectory looks like
        positions = _impact_trajectory(restitution=1.6, horizontal_factor=0.9)

        self.assertEqual(find_bounce_candidates(positions), [])

    def test_rejects_a_contact_that_turns_the_ball_around_horizontally(self):
        positions = _impact_trajectory(restitution=0.7, horizontal_factor=-0.9)

        self.assertEqual(find_bounce_candidates(positions), [])

    def test_rejects_a_contact_that_adds_horizontal_speed(self):
        positions = _impact_trajectory(restitution=0.6, horizontal_factor=1.8)

        self.assertEqual(find_bounce_candidates(positions), [])

    def test_rejects_an_impact_that_adds_total_speed(self):
        # a skidding drive: restitution (0.95) and horizontal gain (1.1) each
        # stay inside their own tolerance, but on a shallow trajectory the
        # horizontal component dominates, so the ball still leaves ~10%
        # faster overall than it arrived - energy went in, so not the court
        positions = _impact_trajectory(
            vx_in=20.0, vy_in=3.0, restitution=0.95, horizontal_factor=1.1
        )

        self.assertEqual(find_bounce_candidates(positions), [])

    def test_rejects_a_noisy_stretch_that_fakes_a_corner(self):
        # one badly mis-detected position in the middle of ordinary flight -
        # exactly the false positive that single-frame peak detection can't
        # tell apart from a real bounce corner
        positions = _single_flight()
        spike = positions[10]
        positions[10] = TrackedPosition(
            frame_idx=spike.frame_idx, x=spike.x, y=spike.y + 60.0, interpolated=False
        )

        self.assertEqual(find_bounce_candidates(positions), [])

    def test_still_finds_a_bounce_when_the_impact_frames_are_undetected(self):
        # the frames right at the impact are the blurriest of the flight and
        # are routinely missed by the detector
        positions = [p for p in _impact_trajectory() if abs(p.frame_idx - 20) > 1]

        candidates = find_bounce_candidates(positions)

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].t, 20.0, delta=1.0)

    def test_rejects_a_junction_fitted_from_too_few_samples(self):
        # a barely-constrained arc can invent a junction out of noise; this
        # is what produced a hand-confirmed false positive on real footage
        # immediately after an eight-frame dropout
        positions = [p for p in _impact_trajectory() if p.frame_idx >= 16]

        self.assertEqual(find_bounce_candidates(positions), [])

        # the same trajectory with enough samples either side is found
        self.assertEqual(
            [c.frame_idx for c in find_bounce_candidates(_impact_trajectory())], [20]
        )

    def test_ignores_interpolated_positions(self):
        # a bounce made entirely of interpolated (straight-line, fabricated)
        # samples on one side leaves too few real ones to fit an arc
        positions = [
            TrackedPosition(frame_idx=p.frame_idx, x=p.x, y=p.y, interpolated=p.frame_idx > 20)
            for p in _impact_trajectory()
        ]

        self.assertEqual(find_bounce_candidates(positions), [])

    def test_merges_neighbouring_detections_of_one_impact(self):
        # several candidate frames straddle the same junction; only one
        # bounce should come out
        candidates = find_bounce_candidates(_impact_trajectory(), min_frame_gap=8)

        self.assertEqual(len(candidates), 1)

    def test_reports_two_separate_bounces_far_enough_apart(self):
        first = _impact_trajectory(impact_frame=20, span=10)
        second = _impact_trajectory(impact_frame=60, x_impact=900.0, y_impact=700.0, span=10)

        candidates = find_bounce_candidates(first + second)

        self.assertEqual([c.frame_idx for c in candidates], [20, 60])

    def test_skips_a_window_broken_by_a_long_tracking_gap(self):
        # a dropout longer than max_internal_gap truncates the arc window to
        # too few samples to fit, rather than fitting across the dropout
        positions = [p for p in _impact_trajectory() if not (14 <= p.frame_idx <= 17)]

        self.assertEqual(find_bounce_candidates(positions, max_internal_gap=2), [])

    def test_works_at_realistic_frame_numbers(self):
        # raw frame indices in the hundreds make a naive quadratic fit
        # numerically unstable; the arcs are fitted about the candidate
        candidates = find_bounce_candidates(_impact_trajectory(impact_frame=1200))

        self.assertEqual([c.frame_idx for c in candidates], [1200])

    def test_accepts_positions_in_arbitrary_order(self):
        positions = list(reversed(_impact_trajectory()))

        self.assertEqual([c.frame_idx for c in find_bounce_candidates(positions)], [20])

    def test_returns_nothing_for_an_empty_trajectory(self):
        self.assertEqual(find_bounce_candidates([]), [])


class FindImpactsTest(unittest.TestCase):
    def test_reports_a_racket_contact_as_an_impact_rather_than_dropping_it(self):
        # the ball leaves faster upward than it arrived - only a racket does
        # that, but it is still an impact worth reporting
        impacts = find_impacts(_impact_trajectory(restitution=1.6, horizontal_factor=0.9))

        self.assertEqual(len(impacts), 1)
        self.assertFalse(impacts[0].is_bounce)
        self.assertIn("restitution", impacts[0].reason)

    def test_labels_a_ball_turned_around_as_a_contact(self):
        impacts = find_impacts(_impact_trajectory(restitution=0.7, horizontal_factor=-0.9))

        self.assertEqual([i.is_bounce for i in impacts], [False])
        self.assertEqual(impacts[0].reason, "turned the ball around")

    def test_labels_a_clean_bounce_as_a_bounce(self):
        impacts = find_impacts(_impact_trajectory(restitution=0.7, horizontal_factor=0.8))

        self.assertEqual([i.is_bounce for i in impacts], [True])
        self.assertEqual(impacts[0].reason, "bounce")

    def test_reports_uninterrupted_flight_as_no_impact_at_all(self):
        # nothing struck the ball, so there is no impact of either kind
        self.assertEqual(find_impacts(_single_flight()), [])

    def test_a_bounce_wins_its_cluster_over_neighbouring_contact_readings(self):
        # frames either side of the true junction blend the two sides'
        # velocities and read as contacts; the bounce must still survive
        impacts = find_impacts(_impact_trajectory(restitution=0.7, horizontal_factor=0.8))

        self.assertEqual(len(impacts), 1)
        self.assertTrue(impacts[0].is_bounce)

    def test_finds_both_a_contact_and_a_later_bounce(self):
        contact = _impact_trajectory(impact_frame=20, restitution=1.6, horizontal_factor=0.9, span=10)
        bounce = _impact_trajectory(
            impact_frame=60, x_impact=900.0, y_impact=700.0, restitution=0.7, span=10
        )

        impacts = find_impacts(contact + bounce)

        self.assertEqual([(i.frame_idx, i.is_bounce) for i in impacts], [(20, False), (60, True)])


class DetectBouncesParabolicTest(unittest.TestCase):
    def test_returns_bounce_events_with_pixel_coordinates(self):
        events = detect_bounces_parabolic(_impact_trajectory())

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].frame_idx, 20)
        self.assertAlmostEqual(events[0].x, 500.0, delta=2.0)
        self.assertIsNone(events[0].world_x)

    def test_projects_the_landing_spot_when_a_calibration_is_available(self):
        positions = _impact_trajectory()
        calibrations = {
            p.frame_idx: CourtCalibration(homography=np.eye(3, dtype=np.float64))
            for p in positions
        }

        events = detect_bounces_parabolic(positions, calibrations_by_frame=calibrations)

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].world_x, 500.0, delta=2.0)
        self.assertAlmostEqual(events[0].world_y, 800.0, delta=2.0)

    def test_leaves_world_coordinates_unset_for_an_uncalibrated_frame(self):
        events = detect_bounces_parabolic(_impact_trajectory(), calibrations_by_frame={})

        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0].world_x)

    def test_forwards_tuning_parameters(self):
        positions = _impact_trajectory(restitution=0.7)

        self.assertEqual(detect_bounces_parabolic(positions, min_restitution=0.9), [])


def _arc_samples(count=8, x0=700.0, y0=400.0, vx=13.0, vy=9.0, gravity=1.2, start_frame=100):
    """One clean stretch of free flight, the shape _fit_arc is meant to model."""
    return [
        TrackedPosition(
            frame_idx=start_frame + i,
            x=x0 + vx * i,
            y=y0 + vy * i + 0.5 * gravity * i * i,
            interpolated=False,
        )
        for i in range(count)
    ]


class FitArcOutlierTest(unittest.TestCase):
    """A least-squares fit has no defence against one bad detection, and the
    caller throws away any arc whose RMSE is too high - so a single misplaced
    blob used to cost a whole bounce. See _fit_arc."""

    T_REF = 100.0

    def test_fits_clean_flight_exactly(self):
        arc = _fit_arc(_arc_samples(), t_ref=self.T_REF)

        self.assertLess(arc.rmse, 1e-6)

    def test_one_bad_detection_does_not_wreck_the_fit(self):
        samples = _arc_samples()
        samples[4] = TrackedPosition(
            frame_idx=samples[4].frame_idx,
            x=samples[4].x - 14.0,
            y=samples[4].y + 12.0,
            interpolated=False,
        )
        arc = _fit_arc(samples, t_ref=self.T_REF)

        self.assertLess(arc.rmse, 1.0)

    def test_the_velocity_survives_a_bad_detection(self):
        samples = _arc_samples()
        clean = _fit_arc(samples, t_ref=self.T_REF).velocity(104.0)
        samples[4] = TrackedPosition(
            frame_idx=samples[4].frame_idx, x=samples[4].x - 14.0, y=samples[4].y + 12.0,
            interpolated=False,
        )
        corrupted = _fit_arc(samples, t_ref=self.T_REF).velocity(104.0)

        self.assertAlmostEqual(clean[0], corrupted[0], delta=0.5)
        self.assertAlmostEqual(clean[1], corrupted[1], delta=0.5)

    def test_does_not_rescue_a_stretch_that_is_noisy_throughout(self):
        # every sample off the parabola, not one - dropping the worst leaves
        # the rest just as unlike free flight, so the caller's RMSE gate
        # still rejects it
        samples = _arc_samples()
        wobble = [9.0, -8.0, 7.0, -9.0, 8.0, -7.0, 9.0, -8.0]
        samples = [
            TrackedPosition(frame_idx=p.frame_idx, x=p.x, y=p.y + w, interpolated=False)
            for p, w in zip(samples, wobble)
        ]
        arc = _fit_arc(samples, t_ref=self.T_REF)

        self.assertGreater(arc.rmse, 4.0)

    def test_drops_at_most_one_sample(self):
        samples = _arc_samples()
        for i in (2, 5):
            samples[i] = TrackedPosition(
                frame_idx=samples[i].frame_idx, x=samples[i].x, y=samples[i].y + 20.0,
                interpolated=False,
            )
        arc = _fit_arc(samples, t_ref=self.T_REF)

        self.assertGreater(arc.rmse, 4.0)

    def test_keeps_enough_samples_to_over_determine_the_fit(self):
        # six samples with one outlier: dropping it would leave five, which
        # barely constrains a quadratic, so the fit keeps them all and the
        # caller's RMSE gate decides
        samples = _arc_samples(count=6)
        samples[3] = TrackedPosition(
            frame_idx=samples[3].frame_idx, x=samples[3].x, y=samples[3].y + 20.0,
            interpolated=False,
        )
        arc = _fit_arc(samples, t_ref=self.T_REF, min_samples=6)

        self.assertGreater(arc.rmse, 4.0)

    def test_still_refuses_too_few_samples(self):
        self.assertIsNone(_fit_arc(_arc_samples(count=5), t_ref=self.T_REF, min_samples=6))

    def test_a_small_wobble_is_not_treated_as_an_outlier(self):
        # a sub-pixel disagreement is ordinary detector noise; removing the
        # largest of those would just flatter the RMSE
        samples = _arc_samples()
        samples[4] = TrackedPosition(
            frame_idx=samples[4].frame_idx, x=samples[4].x, y=samples[4].y + 1.5,
            interpolated=False,
        )
        arc = _fit_arc(samples, t_ref=self.T_REF)

        self.assertGreater(arc.rmse, 0.1)


if __name__ == "__main__":
    unittest.main()
