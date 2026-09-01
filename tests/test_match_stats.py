import unittest

from src.analysis.bounce_detector import BounceEvent
from src.analysis.match_stats import compute_match_stats
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.speed_estimator import ShotSpeed


def _impact(frame_idx, t, kind):
    return BounceCandidate(
        frame_idx=frame_idx,
        t=float(t),
        x=0.0,
        y=0.0,
        restitution=0.0,
        horizontal_ratio=0.0,
        speed_ratio=0.0,
        rmse=0.0,
        kind=kind,
    )


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


if __name__ == "__main__":
    unittest.main()
