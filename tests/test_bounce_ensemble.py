import unittest

from src.analysis.bounce_detector import BounceEvent
from src.analysis.bounce_ensemble import (
    detect_bounces_ensemble,
    filter_bounces_near_net,
    filter_bounces_off_court,
    find_speed_drop_candidates,
)
from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition


class _StubCatBoostModel:
    """Duck-types CatBoostBounceDetector.predict_frames without needing a
    real .cbm checkpoint - returns a fixed set of frames, and mimics the
    real model's in-place x_ball/y_ball mutation contract (leaves them
    untouched here since these tests don't exercise gap-filling)."""

    def __init__(self, frames: set[int]):
        self._frames = frames

    def predict_frames(self, x_ball, y_ball, smooth=True):
        return set(self._frames)


def _identity_calibration() -> CourtCalibration:
    import numpy as np

    return CourtCalibration(homography=np.eye(3, dtype=np.float64))


class FilterBouncesOffCourtTest(unittest.TestCase):
    def test_drops_a_position_far_outside_the_court(self):
        bounces = [BounceEvent(frame_idx=1, x=0, y=0, world_x=5.0, world_y=-65.0)]

        result = filter_bounces_off_court(bounces, margin_m=2.0)

        self.assertEqual(result, [])

    def test_keeps_a_position_just_outside_the_lines(self):
        bounces = [BounceEvent(frame_idx=1, x=0, y=0, world_x=-1.0, world_y=12.0)]

        result = filter_bounces_off_court(bounces, margin_m=2.0)

        self.assertEqual(result, bounces)

    def test_keeps_a_position_with_no_world_coordinates(self):
        bounces = [BounceEvent(frame_idx=1, x=0, y=0, world_x=None, world_y=None)]

        result = filter_bounces_off_court(bounces, margin_m=2.0)

        self.assertEqual(result, bounces)


class FilterBouncesNearNetTest(unittest.TestCase):
    def test_drops_a_position_close_to_the_net_line(self):
        bounces = [BounceEvent(frame_idx=1, x=0, y=0, world_x=5.0, world_y=11.885)]  # right at the net

        result = filter_bounces_near_net(bounces, margin_m=8.0)

        self.assertEqual(result, [])

    def test_keeps_a_position_well_clear_of_the_net(self):
        bounces = [BounceEvent(frame_idx=1, x=0, y=0, world_x=5.0, world_y=22.0)]  # deep near the baseline

        result = filter_bounces_near_net(bounces, margin_m=8.0)

        self.assertEqual(result, bounces)

    def test_keeps_a_position_with_no_world_coordinates(self):
        bounces = [BounceEvent(frame_idx=1, x=0, y=0, world_x=None, world_y=None)]

        result = filter_bounces_near_net(bounces, margin_m=8.0)

        self.assertEqual(result, bounces)


def _positions_from_x(xs: list[float]) -> list[TrackedPosition]:
    return [TrackedPosition(frame_idx=i, x=x, y=0.0, interpolated=False) for i, x in enumerate(xs)]


class FindSpeedDropCandidatesTest(unittest.TestCase):
    def test_flags_deceleration_followed_by_reacceleration_in_the_same_direction(self):
        # fast approach (100/frame), sharp drop to 5, then resumes at ~90/frame in the SAME (+x) direction - a bounce
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, {1})

    def test_ignores_deceleration_followed_by_reversal(self):
        # same sharp drop, but the ball then moves back the way it came - a contact, not a bounce
        positions = _positions_from_x([0.0, 100.0, 105.0, 15.0, -75.0, -165.0, -255.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, set())

    def test_ignores_deceleration_with_no_reacceleration(self):
        # sharp drop, then the ball stops completely (absorbed/blocked) rather than resuming speed
        positions = _positions_from_x([0.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, set())

    def test_ignores_a_drop_below_the_minimum_speed(self):
        # 0.1 "meter" (identity calibration) in one frame at 30fps is ~11 km/h - well under the 20 km/h default
        positions = _positions_from_x([0.0, 0.1, 0.11, 0.111, 0.1111, 0.11111, 0.111111])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, set())

    def test_ignores_a_gentle_deceleration(self):
        # 90 vs 100 - only a mild slowdown, not a sharp impact-like drop
        positions = _positions_from_x([0.0, 100.0, 190.0, 280.0, 370.0, 460.0, 550.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, set())

    def test_skips_a_triple_spanning_an_interpolated_position(self):
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        positions[1] = TrackedPosition(frame_idx=1, x=100.0, y=0.0, interpolated=True)
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, set())

    def test_skips_a_frame_with_no_calibration(self):
        positions = _positions_from_x([0.0, 100.0, 105.0, 195.0, 285.0, 375.0, 465.0])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions) - 1)}  # missing the last frame

        candidates = find_speed_drop_candidates(positions, calibrations_by_frame, fps=30.0)

        self.assertEqual(candidates, set())


class DetectBouncesEnsembleTest(unittest.TestCase):
    def test_adds_a_geometric_candidate_catboost_missed(self):
        # a clean local-max-in-y at frame 10, straddled by real detections -
        # CatBoost (stubbed to find nothing) misses it, the geometric scan shouldn't
        positions = [
            TrackedPosition(frame_idx=9, x=140.0, y=130.0, interpolated=False),
            TrackedPosition(frame_idx=10, x=150.0, y=150.0, interpolated=False),
            TrackedPosition(frame_idx=11, x=160.0, y=100.0, interpolated=False),
        ]
        model = _StubCatBoostModel(frames=set())

        bounces = detect_bounces_ensemble(
            positions, model, num_frames=20, min_y_prominence=15.0, min_frame_gap=3
        )

        self.assertEqual([b.frame_idx for b in bounces], [10])

    def test_merges_catboost_and_geometric_candidates_for_the_same_bounce(self):
        positions = [
            TrackedPosition(frame_idx=9, x=140.0, y=100.0, interpolated=False),
            TrackedPosition(frame_idx=10, x=150.0, y=150.0, interpolated=False),
            TrackedPosition(frame_idx=11, x=160.0, y=130.0, interpolated=False),
        ]
        model = _StubCatBoostModel(frames={11})  # 1 frame off from the geometric peak at 10

        bounces = detect_bounces_ensemble(
            positions, model, num_frames=15, min_y_prominence=15.0, min_frame_gap=3
        )

        self.assertEqual(len(bounces), 1)

    def test_resolves_position_via_fallback_when_bounce_frame_itself_is_missing(self):
        positions = [
            # frame 10 (the bounce CatBoost flags) has no tracked position at all -
            # nearest real detection is 2 frames later
            TrackedPosition(frame_idx=12, x=120.0, y=90.0, interpolated=False),
        ]
        model = _StubCatBoostModel(frames={10})

        bounces = detect_bounces_ensemble(
            positions, model, num_frames=15, position_fallback_frames=5
        )

        self.assertEqual(len(bounces), 1)
        self.assertEqual(bounces[0].x, 120.0)
        self.assertEqual(bounces[0].y, 90.0)

    def test_drops_an_off_court_candidate_when_calibration_is_available(self):
        positions = [TrackedPosition(frame_idx=10, x=-1000.0, y=-1000.0, interpolated=False)]
        model = _StubCatBoostModel(frames={10})
        calibrations_by_frame = {10: _identity_calibration()}

        bounces = detect_bounces_ensemble(
            positions, model, num_frames=15, calibrations_by_frame=calibrations_by_frame, court_margin_m=2.0
        )

        self.assertEqual(bounces, [])

    def test_runs_with_only_the_speed_drop_source_and_no_catboost_model(self):
        # x values stay within the court's plausibility gate (identity
        # calibration - pixel coords ARE world coords here): sharp drop at
        # frame 1, then resumes in the same (+x) direction - a bounce
        positions = _positions_from_x([1.0, 3.0, 3.1, 4.0, 4.9, 5.8, 6.7])
        calibrations_by_frame = {i: _identity_calibration() for i in range(len(positions))}

        bounces = detect_bounces_ensemble(
            positions,
            catboost_model=None,
            num_frames=len(positions),
            calibrations_by_frame=calibrations_by_frame,
            fps=30.0,
            sources=frozenset({"speed_drop"}),
        )

        self.assertEqual([b.frame_idx for b in bounces], [1])

    def test_requires_a_catboost_model_when_catboost_is_in_sources(self):
        positions = [TrackedPosition(frame_idx=0, x=0.0, y=0.0, interpolated=False)]

        with self.assertRaises(ValueError):
            detect_bounces_ensemble(positions, catboost_model=None, num_frames=1, sources=frozenset({"catboost"}))

    def test_drops_a_candidate_near_a_player(self):
        positions = [TrackedPosition(frame_idx=10, x=100.0, y=100.0, interpolated=False)]
        model = _StubCatBoostModel(frames={10})
        player_boxes_by_frame = {10: [(80.0, 80.0, 120.0, 120.0)]}

        bounces = detect_bounces_ensemble(
            positions, model, num_frames=15, player_boxes_by_frame=player_boxes_by_frame
        )

        self.assertEqual(bounces, [])


if __name__ == "__main__":
    unittest.main()
