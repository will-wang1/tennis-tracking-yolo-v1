import unittest

import numpy as np

from src.session.insights import Targets, build_summary, likely_coach


def _coverage(off_court_share):
    # 17 x 37 grid over the court plus margins; put time on court or beside it
    grid = np.zeros((37, 17))
    grid[18, 8] = 1 - off_court_share  # on court
    grid[18, 0] = off_court_share  # 3m beside the left sideline
    return {"origin_m": [-3.0, -6.4], "cell_m": 1.0, "seconds": grid.tolist()}


def _player(n, moving=0.66, dist=140.0, on=100.0, queue=0.03, net=0.15, longest=3.0):
    return {"player": n, "moving_share": moving, "distance_m": dist, "on_camera_s": on,
            "queue_share": queue, "longest_queue_s": longest,
            "zone_shares": {"net": net, "baseline": 0.8 - net, "deep": 0.1, "off_court": 0.1}}


def _report(players, bip=0.79, coverage=None, intensity=None):
    return {"metrics": {"ball_in_play": {"value": bip}, "drills": {"breaks": 0}},
            "players": players,
            "raw": {"player_coverage": coverage or {str(p["player"]): _coverage(0.2) for p in players},
                    "intensity": intensity or []}}


class CoachTest(unittest.TestCase):
    def test_least_mobile_and_most_off_court_is_the_coach(self):
        players = [_player(1, moving=0.38)] + [_player(n) for n in range(2, 6)]
        cov = {"1": _coverage(0.74), **{str(n): _coverage(0.3) for n in range(2, 6)}}
        self.assertEqual(likely_coach(players, cov), 1)

    def test_no_coach_when_nobody_stands_out(self):
        players = [_player(n, moving=0.6 + 0.02 * n) for n in range(1, 6)]
        self.assertIsNone(likely_coach(players, {str(n): _coverage(0.3) for n in range(1, 6)}))

    def test_override_wins(self):
        s = build_summary(_report([_player(n) for n in range(1, 6)]), coach_override=3)
        self.assertEqual((s["coach_player"], s["coach_source"]), (3, "set"))


class BallsHitTest(unittest.TestCase):
    def _players(self, rates):
        out = []
        for n, r in enumerate(rates, start=1):
            p = _player(n)
            p["shots"], p["shots_per_minute"] = round(r * 2), r
            out.append(p)
        return out

    def test_a_player_hitting_under_half_the_group_is_flagged(self):
        s = build_summary(_report(self._players([7, 7.5, 3, 8, 7])))
        self.assertIn("Player 3 hit far fewer balls", [f["title"] for f in s["to_improve"]])
        self.assertIn("hit 6 balls", next(c["line"] for c in s["players"] if c["player"] == 3))

    def test_a_gap_short_of_half_is_not_flagged_nor_called_even(self):
        s = build_summary(_report(self._players([7, 7.5, 4.2, 8, 7])))
        titles = [f["title"] for f in s["to_improve"] + s["went_well"]]
        self.assertFalse(any("balls" in t.lower() for t in titles))

    def test_a_tight_spread_is_balls_shared_evenly(self):
        s = build_summary(_report(self._players([7, 7.5, 6.5, 8, 7])))
        self.assertIn("Balls shared evenly", [f["title"] for f in s["went_well"]])


class FindingsTest(unittest.TestCase):
    def test_a_busy_evenly_shared_session_has_nothing_to_fix(self):
        s = build_summary(_report([_player(n) for n in range(1, 6)]))
        self.assertEqual(s["to_improve"], [])
        self.assertIn("nothing stands out", s["headline"])

    def test_the_player_who_waits_is_called_out_in_the_headline(self):
        players = [_player(n) for n in range(1, 6)]
        players[2] = _player(3, queue=0.15, net=0.02, dist=90.0)
        s = build_summary(_report(players))
        self.assertEqual(s["to_improve"][0]["title"], "Player 3 waited much longer than the others")
        self.assertIn("Player 3", s["headline"])
        self.assertIn("net", s["headline"])

    def test_dead_time_is_judged_against_the_target_and_the_target_is_changeable(self):
        low = build_summary(_report([_player(n) for n in range(1, 6)], bip=0.45))
        self.assertTrue(any(f["title"] == "A lot of dead time" for f in low["to_improve"]))
        lenient = build_summary(_report([_player(n) for n in range(1, 6)], bip=0.45),
                                Targets(ball_in_play_poor=0.40, ball_in_play_good=0.44))
        self.assertTrue(any(f["title"] == "Very little dead time" for f in lenient["went_well"]))

    def test_a_late_drop_in_intensity_is_flagged(self):
        windows = [{"ball_in_play_share": 0.8, "metres_per_minute": {str(n): 90.0 for n in range(1, 6)}},
                   {"ball_in_play_share": 0.8, "metres_per_minute": {str(n): 60.0 for n in range(1, 6)}}]
        s = build_summary(_report([_player(n) for n in range(1, 6)], intensity=windows))
        self.assertTrue(any(f["title"] == "Intensity dropped towards the end" for f in s["to_improve"]))

    def test_every_player_gets_a_plain_sentence(self):
        s = build_summary(_report([_player(n) for n in range(1, 6)]))
        self.assertEqual(len(s["players"]), 5)
        self.assertTrue(all(c["line"].endswith(".") for c in s["players"]))


if __name__ == "__main__":
    unittest.main()
