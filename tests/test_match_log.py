import unittest

from src.analysis.match_log import build_match_log
from src.analysis.serve_sequences import ServeAttempt


def _attempt(t_s, identity, side, outcome, shots_before_next=0, frame_idx=None):
    return ServeAttempt(
        frame_idx=frame_idx if frame_idx is not None else int(t_s * 30),
        t_s=t_s,
        side=side,
        identity=identity,
        outcome=outcome,
        shots_before_next_serve=shots_before_next,
    )


class BuildMatchLogTest(unittest.TestCase):
    def test_a_single_rally_developed_serve_is_one_point(self):
        attempts = [_attempt(0.0, "player_a", "near", "rally_developed", shots_before_next=4)]
        log = build_match_log(attempts)

        self.assertEqual(len(log.points), 1)
        point = log.points[0]
        self.assertEqual(point.server_identity, "player_a")
        self.assertEqual(point.fault_count, 0)
        self.assertFalse(point.double_fault)
        self.assertEqual(point.rally_shot_count, 4)
        self.assertEqual(point.outcome, "rally_developed")

    def test_a_fault_followed_by_its_retry_is_ONE_point_not_two(self):
        attempts = [
            _attempt(0.0, "player_a", "near", "fault"),
            _attempt(5.0, "player_a", "near", "unreturned"),
        ]
        log = build_match_log(attempts)

        self.assertEqual(len(log.points), 1)
        point = log.points[0]
        self.assertEqual(point.fault_count, 1)
        self.assertFalse(point.double_fault)
        # the point is attributed to when the FIRST (faulted) serve
        # happened, not the retry
        self.assertEqual(point.start_t_s, 0.0)
        self.assertEqual(point.outcome, "unreturned")

    def test_two_faults_in_a_row_is_a_double_fault_and_the_third_attempt_is_a_new_point(self):
        # tennis allows only two serves per point - the third attempt here
        # can't belong to player_a's point at all, whatever
        # classify_serve_sequences' structural read of it happened to be;
        # it must start a new point of its own
        attempts = [
            _attempt(0.0, "player_a", "near", "fault"),
            _attempt(5.0, "player_a", "near", "fault"),
            _attempt(10.0, "player_b", "far", "unreturned"),
        ]
        log = build_match_log(attempts)

        self.assertEqual(len(log.points), 2)
        self.assertEqual(log.points[0].fault_count, 2)
        self.assertTrue(log.points[0].double_fault)
        self.assertEqual(log.points[0].server_identity, "player_a")
        self.assertEqual(log.points[0].outcome, "double_fault")
        self.assertEqual(log.points[0].rally_shot_count, 0)
        self.assertEqual(log.points[1].server_identity, "player_b")
        self.assertEqual(log.points[1].outcome, "unreturned")

    def test_multiple_points_are_all_captured(self):
        attempts = [
            _attempt(0.0, "player_a", "near", "unreturned"),
            _attempt(10.0, "player_a", "near", "rally_developed", shots_before_next=6),
            _attempt(30.0, "player_b", "far", "unreturned"),
        ]
        log = build_match_log(attempts)
        self.assertEqual(len(log.points), 3)

    def test_points_served_by_tallies_who_served_each_point(self):
        attempts = [
            _attempt(0.0, "player_a", "near", "unreturned"),
            _attempt(10.0, "player_a", "near", "unreturned"),
            _attempt(20.0, "player_b", "far", "unreturned"),
        ]
        log = build_match_log(attempts)
        self.assertEqual(log.points_served_by, {"player_a": 2, "player_b": 1})

    def test_fault_and_double_fault_counts_by_server(self):
        attempts = [
            _attempt(0.0, "player_a", "near", "fault"),
            _attempt(5.0, "player_a", "near", "fault"),
            _attempt(10.0, "player_a", "near", "unreturned"),  # double fault point
            _attempt(20.0, "player_b", "far", "fault"),
            _attempt(25.0, "player_b", "far", "rally_developed", shots_before_next=2),  # single fault
        ]
        log = build_match_log(attempts)

        self.assertEqual(log.fault_counts_by, {"player_a": 2, "player_b": 1})
        self.assertEqual(log.double_fault_counts_by, {"player_a": 1})
        self.assertNotIn("player_b", log.double_fault_counts_by)

    def test_first_serve_in_rate(self):
        attempts = [
            _attempt(0.0, "player_a", "near", "unreturned"),  # first serve in
            _attempt(10.0, "player_a", "near", "fault"),
            _attempt(15.0, "player_a", "near", "unreturned"),  # first serve was a fault
        ]
        log = build_match_log(attempts)

        self.assertAlmostEqual(log.first_serve_in_rate_by["player_a"], 0.5)

    def test_first_serve_in_rate_is_none_for_a_server_with_no_points(self):
        log = build_match_log([])
        self.assertEqual(log.first_serve_in_rate_by, {})

    def test_average_rally_shot_count_excludes_unknown_outcomes(self):
        attempts = [
            _attempt(0.0, "player_a", "near", "rally_developed", shots_before_next=4),
            _attempt(20.0, "player_b", "far", "rally_developed", shots_before_next=6),
            _attempt(40.0, "player_a", "near", "unknown", shots_before_next=0),
        ]
        log = build_match_log(attempts)
        self.assertAlmostEqual(log.average_rally_shot_count, 5.0)

    def test_average_rally_shot_count_is_none_with_no_decided_points(self):
        attempts = [_attempt(0.0, "player_a", "near", "unknown")]
        log = build_match_log(attempts)
        self.assertIsNone(log.average_rally_shot_count)

    def test_empty_input_produces_an_empty_log(self):
        log = build_match_log([])
        self.assertEqual(log.points, [])
        self.assertEqual(log.points_served_by, {})

    def test_to_dict_round_trips_through_json(self):
        import json

        attempts = [
            _attempt(0.0, "player_a", "near", "fault"),
            _attempt(5.0, "player_a", "near", "unreturned"),
        ]
        log = build_match_log(attempts)

        decoded = json.loads(json.dumps(log.to_dict()))
        self.assertEqual(decoded["point_count"], 1)
        self.assertEqual(decoded["points"][0]["fault_count"], 1)
        self.assertEqual(decoded["points_served_by"], {"player_a": 1})


if __name__ == "__main__":
    unittest.main()
