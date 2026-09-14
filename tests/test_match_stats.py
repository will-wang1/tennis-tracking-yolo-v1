import unittest

import numpy as np

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration
from src.analysis.match_stats import (
    attribute_contacts_to_court_half,
    compute_match_stats,
    compute_player_movement,
    compute_serve_speed_trend,
)
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.speed_estimator import ShotSpeed


def _impact(frame_idx, t, kind, x=0.0, y=0.0):
    return BounceCandidate(
        frame_idx=frame_idx,
        t=float(t),
        x=x,
        y=y,
        restitution=0.0,
        horizontal_ratio=0.0,
        speed_ratio=0.0,
        rmse=0.0,
        kind=kind,
    )


def _identity_calibration():
    # pixel_to_world becomes the identity, so a contact's (x, y) IS its
    # world position - lets a test pick world_y directly via the impact's y.
    return CourtCalibration(homography=np.eye(3, dtype=np.float64))


class ComputeMatchStatsTest(unittest.TestCase):
    def test_groups_close_impacts_into_one_rally(self):
        # every gap here is well under the default 4s rally-gap threshold
        fps = 30.0
        impacts = [
            _impact(0, 0, "contact"),
            _impact(30, 30, "bounce"),  # 1.0s later
            _impact(60, 60, "contact"),  # 1.0s later
        ]
        stats = compute_match_stats(impacts, shots=[], bounces=[], fps=fps)

        self.assertEqual(len(stats.rallies), 1)
        self.assertEqual(stats.rallies[0].shot_count, 2)
        self.assertEqual(stats.rallies[0].bounce_count, 1)

    def test_splits_into_separate_rallies_on_a_long_gap(self):
        fps = 30.0
        impacts = [
            _impact(0, 0, "contact"),
            _impact(30, 30, "bounce"),  # 1.0s later - same rally
            _impact(600, 600, "contact"),  # 19s later - a new rally
            _impact(630, 630, "bounce"),
        ]
        stats = compute_match_stats(impacts, shots=[], bounces=[], fps=fps)

        self.assertEqual(len(stats.rallies), 2)
        self.assertEqual(stats.rallies[0].shot_count, 1)
        self.assertEqual(stats.rallies[1].shot_count, 1)

    def test_rally_gap_seconds_is_configurable(self):
        fps = 30.0
        impacts = [_impact(0, 0, "contact"), _impact(150, 150, "bounce")]  # 5.0s apart

        # default (4.0s) splits these into two rallies
        default_stats = compute_match_stats(impacts, shots=[], bounces=[], fps=fps)
        self.assertEqual(len(default_stats.rallies), 2)

        # a wider allowance keeps them together
        wide_stats = compute_match_stats(
            impacts, shots=[], bounces=[], fps=fps, rally_gap_seconds=10.0
        )
        self.assertEqual(len(wide_stats.rallies), 1)

    def test_totals_count_every_impact_kind_regardless_of_rally(self):
        fps = 30.0
        impacts = [
            _impact(0, 0, "contact"),
            _impact(30, 30, "bounce"),
            _impact(60, 60, "unknown"),
        ]
        stats = compute_match_stats(impacts, shots=[], bounces=[], fps=fps)

        self.assertEqual(stats.total_contacts, 1)
        self.assertEqual(stats.total_bounces, 1)
        self.assertEqual(stats.total_unattributed, 1)

    def test_rally_peak_speed_is_the_fastest_overlapping_shot(self):
        fps = 30.0
        impacts = [_impact(0, 0, "contact"), _impact(60, 60, "bounce")]
        shots = [
            ShotSpeed(start_frame=0, end_frame=30, peak_frame=15, peak_speed=80.0, unit="km/h"),
            ShotSpeed(start_frame=31, end_frame=60, peak_frame=45, peak_speed=120.0, unit="km/h"),
        ]
        stats = compute_match_stats(impacts, shots=shots, bounces=[], fps=fps)

        self.assertEqual(len(stats.rallies), 1)
        self.assertAlmostEqual(stats.rallies[0].peak_speed, 120.0)
        self.assertEqual(stats.rallies[0].peak_speed_unit, "km/h")

    def test_bounce_locations_skip_bounces_with_no_world_position(self):
        bounces = [
            BounceEvent(frame_idx=0, x=1.0, y=1.0, world_x=2.5, world_y=6.0),
            BounceEvent(frame_idx=10, x=2.0, y=2.0, world_x=None, world_y=None),
        ]
        stats = compute_match_stats([], shots=[], bounces=bounces, fps=30.0)

        self.assertEqual(stats.bounce_locations, [(2.5, 6.0)])

    def test_stroke_counts_pass_through_unchanged(self):
        near = {"forehand": 3, "backhand": 1, "serve": 2}
        far = {"forehand": 0, "backhand": 4, "serve": 2}
        stats = compute_match_stats(
            [], shots=[], bounces=[], fps=30.0, near_shot_counts=near, far_shot_counts=far
        )

        self.assertEqual(stats.near_shot_counts, near)
        self.assertEqual(stats.far_shot_counts, far)
        # a copy, not the same object - a caller mutating their own dict
        # afterward must not silently change what was already recorded
        near["forehand"] = 999
        self.assertEqual(stats.near_shot_counts["forehand"], 3)

    def test_no_impacts_produces_no_rallies(self):
        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0)
        self.assertEqual(stats.rallies, [])
        self.assertEqual(stats.total_bounces, 0)

    def test_to_dict_round_trips_through_json(self):
        import json

        impacts = [_impact(0, 0, "contact"), _impact(30, 30, "bounce")]
        bounces = [BounceEvent(frame_idx=30, x=1.0, y=1.0, world_x=3.0, world_y=7.5)]
        stats = compute_match_stats(
            impacts, shots=[], bounces=bounces, fps=30.0, near_shot_counts={"forehand": 1}
        )

        # must not raise - every field has to be JSON-serializable
        encoded = json.dumps(stats.to_dict())
        decoded = json.loads(encoded)
        self.assertEqual(decoded["total_bounces"], 1)
        self.assertEqual(decoded["bounce_locations"], [[3.0, 7.5]])


class ComputePlayerMovementTest(unittest.TestCase):
    def test_sums_distance_across_consecutive_frames(self):
        # 3-4-5 triangle each step: 3m distance, 1 frame apart at 30fps
        positions = {0: (0.0, 0.0), 30: (3.0, 4.0), 60: (6.0, 8.0)}
        stats = compute_player_movement(positions, fps=30.0)

        self.assertAlmostEqual(stats.distance_m, 10.0)
        self.assertEqual(stats.tracked_frames, 2)

    def test_rejects_an_implausible_jump_as_a_detector_artifact(self):
        # frame 0->1: 100m in 1/30s (3000 m/s) - a detector artifact, rejected
        # frame 1->2: 0.1m in 1/30s (3 m/s) - an ordinary plausible step
        positions = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (100.1, 0.0)}
        stats = compute_player_movement(positions, fps=30.0)

        # only the second (plausible) step counts
        self.assertAlmostEqual(stats.distance_m, 0.1)
        self.assertEqual(stats.tracked_frames, 1)

    def test_returns_none_with_fewer_than_two_frames(self):
        self.assertIsNone(compute_player_movement({}, fps=30.0))
        self.assertIsNone(compute_player_movement({0: (0.0, 0.0)}, fps=30.0))

    def test_returns_none_when_every_step_is_implausible(self):
        positions = {0: (0.0, 0.0), 1: (500.0, 0.0)}
        self.assertIsNone(compute_player_movement(positions, fps=30.0))

    def test_max_speed_is_the_fastest_plausible_step(self):
        # step 1: 3m in 1s = 3 m/s; step 2: 9m in 1s = 9 m/s
        positions = {0: (0.0, 0.0), 30: (3.0, 0.0), 60: (12.0, 0.0)}
        stats = compute_player_movement(positions, fps=30.0)

        self.assertAlmostEqual(stats.max_speed_mps, 9.0)


class ComputeServeSpeedTrendTest(unittest.TestCase):
    def test_pairs_each_serve_with_its_overlapping_shot_speed(self):
        events = [(10, "serve"), (200, "forehand"), (400, "serve")]
        shots = [
            ShotSpeed(start_frame=0, end_frame=50, peak_frame=10, peak_speed=180.0, unit="km/h"),
            ShotSpeed(start_frame=380, end_frame=420, peak_frame=400, peak_speed=165.0, unit="km/h"),
        ]
        readings = compute_serve_speed_trend(events, shots, fps=30.0)

        self.assertEqual(len(readings), 2)  # the forehand event is skipped
        self.assertEqual(readings[0].frame_idx, 10)
        self.assertAlmostEqual(readings[0].peak_speed, 180.0)
        self.assertEqual(readings[1].frame_idx, 400)
        self.assertAlmostEqual(readings[1].peak_speed, 165.0)

    def test_serve_with_no_overlapping_shot_speed_is_a_missing_reading_not_zero(self):
        readings = compute_serve_speed_trend([(10, "serve")], shots=[], fps=30.0)

        self.assertEqual(len(readings), 1)
        self.assertIsNone(readings[0].peak_speed)
        self.assertIsNone(readings[0].unit)

    def test_no_serves_produces_no_readings(self):
        events = [(10, "forehand"), (50, "backhand")]
        self.assertEqual(compute_serve_speed_trend(events, shots=[], fps=30.0), [])


class ComputeMatchStatsExtrasTest(unittest.TestCase):
    def test_bounce_zone_labels_are_parallel_to_bounce_locations(self):
        bounces = [
            BounceEvent(frame_idx=0, x=1.0, y=1.0, world_x=2.0, world_y=2.0),
            BounceEvent(frame_idx=10, x=2.0, y=2.0, world_x=2.0, world_y=22.0),
        ]
        stats = compute_match_stats([], shots=[], bounces=bounces, fps=30.0)

        self.assertEqual(len(stats.bounce_zone_labels), 2)
        self.assertEqual(stats.bounce_zone_labels[0], "far_deuce_deep")
        self.assertEqual(stats.bounce_zone_labels[1], "near_ad_deep")

    def test_bounce_zone_counts_tallies_the_labels(self):
        bounces = [
            BounceEvent(frame_idx=0, x=0.0, y=0.0, world_x=2.0, world_y=2.0),
            BounceEvent(frame_idx=1, x=0.0, y=0.0, world_x=2.0, world_y=2.0),
            BounceEvent(frame_idx=2, x=0.0, y=0.0, world_x=2.0, world_y=22.0),
        ]
        stats = compute_match_stats([], shots=[], bounces=bounces, fps=30.0)

        self.assertEqual(stats.bounce_zone_counts, {"far_deuce_deep": 2, "near_ad_deep": 1})

    def test_player_movement_is_none_when_no_positions_given(self):
        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0)
        self.assertIsNone(stats.near_player_movement)
        self.assertIsNone(stats.far_player_movement)

    def test_player_movement_is_computed_when_positions_given(self):
        near = {0: (0.0, 0.0), 30: (3.0, 4.0)}
        far = {0: (0.0, 0.0), 30: (6.0, 8.0)}
        stats = compute_match_stats(
            [], shots=[], bounces=[], fps=30.0,
            near_player_positions_by_frame=near, far_player_positions_by_frame=far,
        )

        self.assertAlmostEqual(stats.near_player_movement.distance_m, 5.0)
        self.assertAlmostEqual(stats.far_player_movement.distance_m, 10.0)

    def test_serve_speeds_are_computed_when_events_given(self):
        events = [(10, "serve")]
        shots = [ShotSpeed(start_frame=0, end_frame=20, peak_frame=10, peak_speed=190.0, unit="km/h")]
        stats = compute_match_stats(
            [], shots=shots, bounces=[], fps=30.0, near_shot_events=events
        )

        self.assertEqual(len(stats.near_serve_speeds), 1)
        self.assertAlmostEqual(stats.near_serve_speeds[0].peak_speed, 190.0)
        self.assertEqual(stats.far_serve_speeds, [])

    def test_to_dict_round_trips_the_new_fields_through_json(self):
        import json

        bounces = [BounceEvent(frame_idx=0, x=1.0, y=1.0, world_x=2.0, world_y=2.0)]
        near = {0: (0.0, 0.0), 30: (3.0, 4.0)}
        events = [(10, "serve")]
        shots = [ShotSpeed(start_frame=0, end_frame=20, peak_frame=10, peak_speed=190.0, unit="km/h")]
        stats = compute_match_stats(
            [], shots=shots, bounces=bounces, fps=30.0,
            near_player_positions_by_frame=near, near_shot_events=events,
        )

        decoded = json.loads(json.dumps(stats.to_dict()))
        self.assertEqual(decoded["bounce_zone_labels"], ["far_deuce_deep"])
        self.assertEqual(decoded["near_player_movement"]["distance_m"], 5.0)
        self.assertEqual(len(decoded["near_serve_speeds"]), 1)


class AttributeContactsToCourtHalfTest(unittest.TestCase):
    def test_splits_contacts_by_world_y_against_the_net(self):
        contacts = [
            _impact(0, 0, "contact", x=5.0, y=2.0),  # far half
            _impact(30, 30, "contact", x=5.0, y=22.0),  # near half
        ]
        calibrations = {0: _identity_calibration(), 30: _identity_calibration()}

        results = attribute_contacts_to_court_half(contacts, calibrations)

        self.assertEqual([half for _, half in results], ["far", "near"])
        self.assertEqual(results[0][0], (5.0, 2.0))

    def test_skips_contacts_with_no_calibration_for_their_frame(self):
        contacts = [_impact(0, 0, "contact", x=5.0, y=2.0)]
        results = attribute_contacts_to_court_half(contacts, calibrations_by_frame={})
        self.assertEqual(results, [])


class ComputeMatchStatsContactAttributionTest(unittest.TestCase):
    def test_only_contacts_are_attributed_not_bounces_or_unknowns(self):
        impacts = [
            _impact(0, 0, "contact", x=5.0, y=2.0),  # far
            _impact(30, 30, "bounce", x=5.0, y=2.0),
            _impact(60, 60, "unknown", x=5.0, y=2.0),
        ]
        calibrations = {f: _identity_calibration() for f in (0, 30, 60)}
        stats = compute_match_stats(
            impacts, shots=[], bounces=[], fps=30.0, calibrations_by_frame=calibrations
        )

        self.assertEqual(stats.contact_locations, [(5.0, 2.0)])
        self.assertEqual(stats.contact_sides, ["far"])

    def test_contact_side_counts_tallies_both_players(self):
        impacts = [
            _impact(0, 0, "contact", x=5.0, y=2.0),  # far
            _impact(30, 30, "contact", x=5.0, y=22.0),  # near
            _impact(60, 60, "contact", x=5.0, y=1.0),  # far
        ]
        calibrations = {f: _identity_calibration() for f in (0, 30, 60)}
        stats = compute_match_stats(
            impacts, shots=[], bounces=[], fps=30.0, calibrations_by_frame=calibrations
        )

        self.assertEqual(stats.contact_side_counts, {"far": 2, "near": 1})

    def test_no_calibrations_leaves_contact_fields_empty(self):
        impacts = [_impact(0, 0, "contact", x=5.0, y=2.0)]
        stats = compute_match_stats(impacts, shots=[], bounces=[], fps=30.0)

        self.assertEqual(stats.contact_locations, [])
        self.assertEqual(stats.contact_sides, [])
        self.assertEqual(stats.contact_side_counts, {})

    def test_to_dict_round_trips_contact_attribution_through_json(self):
        import json

        impacts = [_impact(0, 0, "contact", x=5.0, y=2.0)]
        calibrations = {0: _identity_calibration()}
        stats = compute_match_stats(
            impacts, shots=[], bounces=[], fps=30.0, calibrations_by_frame=calibrations
        )

        decoded = json.loads(json.dumps(stats.to_dict()))
        self.assertEqual(decoded["contact_locations"], [[5.0, 2.0]])
        self.assertEqual(decoded["contact_sides"], ["far"])
        self.assertEqual(decoded["contact_side_counts"], {"far": 1})


class ComputeMatchStatsSceneCutsTest(unittest.TestCase):
    def test_defaults_to_no_scene_cuts(self):
        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0)
        self.assertEqual(stats.scene_cuts, [])

    def test_scene_cuts_pass_through(self):
        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0, scene_cuts=[120, 340])
        self.assertEqual(stats.scene_cuts, [120, 340])

    def test_scene_cuts_round_trip_through_json(self):
        import json

        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0, scene_cuts=[42])
        decoded = json.loads(json.dumps(stats.to_dict()))
        self.assertEqual(decoded["scene_cuts"], [42])

    def test_match_log_defaults_to_none(self):
        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0)
        self.assertIsNone(stats.match_log)
        self.assertIsNone(stats.to_dict()["match_log"])

    def test_match_log_passes_through_and_serializes(self):
        import json

        from src.analysis.match_log import build_match_log
        from src.analysis.serve_sequences import ServeAttempt

        log = build_match_log(
            [ServeAttempt(frame_idx=0, t_s=0.0, side="near", identity="player_a", outcome="unreturned", shots_before_next_serve=0)]
        )
        stats = compute_match_stats([], shots=[], bounces=[], fps=30.0, match_log=log)

        self.assertEqual(stats.match_log.points[0].server_identity, "player_a")
        decoded = json.loads(json.dumps(stats.to_dict()))
        self.assertEqual(decoded["match_log"]["point_count"], 1)


if __name__ == "__main__":
    unittest.main()
