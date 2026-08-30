import unittest

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.impact_pipeline import ImpactAnalysis, analyze_impacts, merge_impacts
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.detection.ball_detector import Detection


def _candidate(t: float, rmse: float = 1.0, kind: str = "bounce") -> BounceCandidate:
    return BounceCandidate(
        frame_idx=int(round(t)),
        t=t,
        x=500.0,
        y=800.0,
        restitution=0.7,
        horizontal_ratio=0.9,
        speed_ratio=0.8,
        rmse=rmse,
        is_bounce=kind == "bounce",
        kind=kind,
    )


def _impact_detections(
    impact_frame: int = 20,
    x_impact: float = 500.0,
    y_impact: float = 800.0,
    vx_in: float = 6.0,
    vy_in: float = 8.0,
    gravity: float = 1.0,
    restitution: float = 0.7,
    horizontal_factor: float = 0.8,
    span: int = 10,
) -> list[Detection]:
    """Two exact free-flight arcs meeting at `impact_frame`, as per-frame
    detections - the shape the pipeline consumes. Pixel conventions: y grows
    downward, so a positive `vy_in` is the ball descending."""
    detections: list[Detection] = [None] * (impact_frame - span)
    vx_out = vx_in * horizontal_factor
    vy_out = -vy_in * restitution
    for offset in range(-span, span + 1):
        if offset < 0:
            x = x_impact + vx_in * offset
            y = y_impact + vy_in * offset + 0.5 * gravity * offset**2
        else:
            x = x_impact + vx_out * offset
            y = y_impact + vy_out * offset + 0.5 * gravity * offset**2
        detections.append(Detection(x=x, y=y, confidence=0.9))
    return detections


def _identity_calibrations(num_frames: int) -> dict[int, CourtCalibration]:
    """Pixels straight through as metres - enough for the classifier to have
    a projected court position to measure at all."""
    calibration = CourtCalibration(homography=np.eye(3, dtype=np.float64))
    return {i: calibration for i in range(num_frames)}


class MergeImpactsTest(unittest.TestCase):
    def test_keeps_both_when_they_are_far_apart(self):
        merged = merge_impacts([_candidate(10.0)], [_candidate(30.0)], min_frame_gap=5)

        self.assertEqual([c.t for c in merged], [10.0, 30.0])

    def test_collapses_two_descriptions_of_the_same_impact(self):
        merged = merge_impacts([_candidate(10.0)], [_candidate(12.0)], min_frame_gap=5)

        self.assertEqual(len(merged), 1)

    def test_keeps_the_better_fitting_of_a_collapsed_pair(self):
        merged = merge_impacts(
            [_candidate(10.0, rmse=4.0)], [_candidate(12.0, rmse=1.0)], min_frame_gap=5
        )

        self.assertEqual(merged[0].rmse, 1.0)

    def test_prefers_the_better_fit_whichever_search_found_it(self):
        merged = merge_impacts(
            [_candidate(10.0, rmse=1.0)], [_candidate(12.0, rmse=4.0)], min_frame_gap=5
        )

        self.assertEqual(merged[0].rmse, 1.0)

    def test_returns_impacts_in_time_order(self):
        merged = merge_impacts(
            [_candidate(30.0), _candidate(10.0)], [_candidate(50.0)], min_frame_gap=5
        )

        self.assertEqual([c.t for c in merged], [10.0, 30.0, 50.0])


class AnalyzeImpactsTest(unittest.TestCase):
    def test_finds_the_impact_in_a_two_arc_trajectory(self):
        analysis = analyze_impacts(_impact_detections(impact_frame=20), fps=30.0)

        self.assertTrue(analysis.impacts)
        self.assertAlmostEqual(analysis.impacts[0].t, 20.0, delta=1.0)

    def test_leaves_impacts_unattributed_without_a_calibration(self):
        # no calibration means no projected court position, so there is
        # nothing to measure the height effect with and no touchdown verdict
        analysis = analyze_impacts(_impact_detections(), fps=30.0)

        self.assertEqual(analysis.touchdowns, [])

    def test_a_calibration_produces_a_verdict_for_every_impact(self):
        detections = _impact_detections()
        analysis = analyze_impacts(
            detections,
            fps=30.0,
            calibrations_by_frame=_identity_calibrations(len(detections)),
        )

        self.assertEqual(len(analysis.touchdowns), len(analysis.impacts))
        for impact in analysis.impacts:
            self.assertIn(impact.kind, ("bounce", "contact", "unknown"))

    def test_classifier_arguments_reach_the_classifier(self):
        detections = _impact_detections()
        kwargs = dict(fps=30.0, calibrations_by_frame=_identity_calibrations(len(detections)))

        # an edge margin past the impact's screen y disqualifies every impact
        analysis = analyze_impacts(detections, frame_edge_margin=10_000.0, **kwargs)

        self.assertTrue(analysis.impacts)
        self.assertEqual({impact.kind for impact in analysis.impacts}, {"unknown"})

    def test_the_three_verdict_groups_partition_the_impacts(self):
        detections = _impact_detections()
        analysis = analyze_impacts(
            detections,
            fps=30.0,
            calibrations_by_frame=_identity_calibrations(len(detections)),
        )

        self.assertEqual(
            len(analysis.bounces) + len(analysis.contacts) + len(analysis.unattributed),
            len(analysis.impacts),
        )

    def test_reports_no_impacts_for_a_trajectory_that_never_kinks(self):
        straight = [Detection(x=100.0 + 5 * i, y=400.0, confidence=0.9) for i in range(40)]

        self.assertEqual(analyze_impacts(straight, fps=30.0).impacts, [])

    def test_survives_a_video_with_no_detections_at_all(self):
        analysis = analyze_impacts([None] * 30, fps=30.0)

        self.assertIsInstance(analysis, ImpactAnalysis)
        self.assertEqual(analysis.impacts, [])


if __name__ == "__main__":
    unittest.main()
