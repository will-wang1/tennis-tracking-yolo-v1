import unittest

from src.analysis.flight_segmenter import (
    find_flight_segments,
    find_segment_impacts,
    segment_impacts_as_candidates,
)
from src.tracking.ball_tracker import TrackedPosition


def _flight(start_frame, count, x0, y0, vx, vy, gravity=1.0, step=1):
    """One parabolic flight, sampled every `step` frames."""
    positions = []
    for i in range(count):
        t = i * step
        positions.append(
            TrackedPosition(
                frame_idx=start_frame + t,
                x=x0 + vx * t,
                y=y0 + vy * t + 0.5 * gravity * t * t,
                interpolated=False,
            )
        )
    return positions


def _bounce_trajectory(impact_frame=40, span=16, gap=0, gravity=1.0):
    """Two exact flights meeting at (400, 700) on `impact_frame`, optionally
    with `gap` frames of missing detections centred on the impact.

    Built outward from the impact so both arcs genuinely pass through it -
    back-dating a flight by its linear term alone would leave them meeting
    somewhere else entirely.
    """
    x_impact, y_impact = 400.0, 700.0
    positions = []
    for offset in range(-span, span + 1):
        vx, vy = (8.0, 6.0) if offset < 0 else (6.0, -9.0)
        positions.append(
            TrackedPosition(
                frame_idx=impact_frame + offset,
                x=x_impact + vx * offset,
                y=y_impact + vy * offset + 0.5 * gravity * offset**2,
                interpolated=False,
            )
        )
    if gap:
        low, high = impact_frame - gap // 2, impact_frame + gap // 2
        positions = [p for p in positions if not (low <= p.frame_idx <= high)]
    return positions


class FindFlightSegmentsTest(unittest.TestCase):
    def test_finds_one_segment_for_uninterrupted_flight(self):
        segments = find_flight_segments(_flight(10, 20, 100.0, 300.0, 7.0, -5.0))

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_frame, 10)
        self.assertEqual(segments[0].end_frame, 29)
        self.assertLess(segments[0].rmse, 0.01)

    def test_splits_at_an_impact(self):
        segments = find_flight_segments(_bounce_trajectory())

        self.assertEqual(len(segments), 2)
        # the impact sample lies on BOTH curves and the tolerance is loose
        # enough to carry a couple of samples past it, so the split lands
        # near the impact rather than exactly on it
        self.assertAlmostEqual(segments[0].end_frame, 40, delta=3)
        self.assertAlmostEqual(segments[1].start_frame, 40, delta=3)

    def test_spans_a_dropout_within_one_flight(self):
        positions = [p for p in _flight(10, 24, 100.0, 300.0, 7.0, -5.0) if not 18 <= p.frame_idx <= 23]

        segments = find_flight_segments(positions)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].start_frame, 10)
        self.assertEqual(segments[0].end_frame, 33)

    def test_refuses_to_span_a_very_long_dropout(self):
        positions = _flight(10, 12, 100.0, 300.0, 7.0, -5.0) + _flight(
            60, 12, 500.0, 260.0, 7.0, -5.0
        )

        segments = find_flight_segments(positions, max_gap=12)

        self.assertEqual(len(segments), 2)

    def test_ignores_interpolated_positions(self):
        positions = [
            TrackedPosition(frame_idx=p.frame_idx, x=p.x, y=p.y, interpolated=True)
            for p in _flight(10, 20, 100.0, 300.0, 7.0, -5.0)
        ]

        self.assertEqual(find_flight_segments(positions), [])

    def test_returns_nothing_for_too_few_positions(self):
        self.assertEqual(find_flight_segments(_flight(10, 3, 100.0, 300.0, 7.0, -5.0)), [])

    def test_handles_an_empty_trajectory(self):
        self.assertEqual(find_flight_segments([]), [])


class FindSegmentImpactsTest(unittest.TestCase):
    def test_locates_an_impact_between_two_flights(self):
        segments = find_flight_segments(_bounce_trajectory(impact_frame=40))

        impacts = find_segment_impacts(segments)

        self.assertEqual(len(impacts), 1)
        t, x, y = impacts[0].t, impacts[0].x, impacts[0].y
        self.assertAlmostEqual(t, 40.0, delta=1.5)
        self.assertAlmostEqual(x, 400.0, delta=15.0)
        self.assertAlmostEqual(y, 700.0, delta=15.0)

    def test_locates_an_impact_hidden_by_a_dropout(self):
        # nothing detected for ten frames around the landing - the case the
        # frame-by-frame scan cannot see at all
        segments = find_flight_segments(_bounce_trajectory(impact_frame=40, gap=10))

        impacts = find_segment_impacts(segments)

        self.assertEqual(len(impacts), 1)
        self.assertAlmostEqual(impacts[0].t, 40.0, delta=2.5)

    def test_reports_nothing_for_a_single_flight(self):
        segments = find_flight_segments(_flight(10, 20, 100.0, 300.0, 7.0, -5.0))

        self.assertEqual(find_segment_impacts(segments), [])

    def test_refuses_to_join_flights_across_a_long_dropout(self):
        segments = find_flight_segments(
            _flight(10, 12, 100.0, 300.0, 7.0, -5.0) + _flight(90, 12, 900.0, 260.0, 7.0, -5.0),
            max_gap=12,
        )

        self.assertEqual(find_segment_impacts(segments, max_separation_frames=25), [])

    def test_finds_several_impacts_in_a_rally(self):
        positions = (
            _flight(0, 14, 100.0, 500.0, 9.0, 7.0)
            + _flight(16, 14, 240.0, 600.0, 7.0, -8.0)
            + _flight(32, 14, 350.0, 520.0, 6.0, 6.0)
        )

        impacts = find_segment_impacts(find_flight_segments(positions))

        self.assertEqual(len(impacts), 2)
        self.assertLess(impacts[0].t, impacts[1].t)


class SegmentImpactsAsCandidatesTest(unittest.TestCase):
    def test_recovered_impacts_carry_no_verdict(self):
        # intersecting two flights proves something interrupted the ball, not
        # what did it - none of the court-vs-racket checks have run yet, so
        # claiming "bounce" here would be a marker with no evidence behind it
        candidates = segment_impacts_as_candidates(_bounce_trajectory(impact_frame=40))

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].kind, "unknown")
        self.assertFalse(candidates[0].is_bounce)

    def test_still_carries_the_measurements_needed_to_classify_it(self):
        candidates = segment_impacts_as_candidates(_bounce_trajectory(impact_frame=40))

        self.assertGreater(candidates[0].restitution, 0.0)
        self.assertGreater(candidates[0].horizontal_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
