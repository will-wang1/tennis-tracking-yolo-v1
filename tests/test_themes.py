import unittest

from src.session.feeding import feeding_summary, rallies
from src.session.themes import consistency, evaluate, netplay


def sw(t, player, end, at_net=False, fps=25):
    return {"frame": int(t * fps), "player": player, "end": end, "at_net": at_net}


class PartnerRalliesTest(unittest.TestCase):
    def test_two_pairs_side_by_side_stay_separate(self):
        # Pair A (1 near, 2 far) and pair B (3 near, 4 far) rally at the same time.
        swings = []
        for k in range(4):
            t = 10 + 3 * k
            swings += [sw(t, 1, "near"), sw(t + 0.3, 3, "near"), sw(t + 1.5, 2, "far"), sw(t + 1.8, 4, "far")]
        rs = rallies(swings, 25.0, coach=None)
        self.assertEqual(sorted((r.starter, r.shots) for r in rs), [(1, 8), (3, 8)])

    def test_same_end_twice_is_a_new_rally(self):
        rs = rallies([sw(1, 1, "near"), sw(2, 2, "near")], 25.0, coach=None)
        self.assertEqual([r.shots for r in rs], [1, 1])

    def test_without_ends_falls_back_to_time(self):
        rs = rallies([{"frame": 25, "player": 1}, {"frame": 50, "player": 2}], 25.0, coach=None)
        self.assertEqual([r.shots for r in rs], [2])


class NetplayTest(unittest.TestCase):
    def test_more_time_at_the_net_late_is_improved(self):
        report = {"raw": {"net_windows": [
            {"net_share": {"1": 0.05, "2": 0.1, "3": 0.08, "9": 0.9}},
            {"net_share": {"1": 0.1, "2": 0.12, "3": 0.1}},
            {"net_share": {"1": 0.25, "2": 0.3, "3": 0.2, "9": 0.0}},
        ]}}
        swings = [sw(t, 1 + t % 3, "far", at_net=t > 20) for t in range(0, 40)]
        out = netplay(report, swings, coach=9)
        self.assertEqual(out["verdict"], "improved")
        self.assertIn("8% of the time early, 25% late", out["headline"])
        self.assertEqual(out["measures"][1]["name"], "shots from the net")
        self.assertNotIn("9", out["per_player"])

    def test_nothing_to_compare(self):
        self.assertEqual(netplay({"raw": {}}, [], None)["verdict"], "not enough")


class ConsistencyTest(unittest.TestCase):
    def test_longer_rallies_late_is_improved(self):
        rl = [{"start_s": i, "shots": 1 if i < 8 else 5} for i in range(16)]
        out = consistency({"rally_list": rl})
        self.assertEqual(out["verdict"], "improved")
        self.assertEqual(out["measures"][0]["early"], 1.0)

    def test_too_few_rallies(self):
        self.assertEqual(consistency({"rally_list": [{"start_s": 0, "shots": 2}]})["verdict"], "not enough")


class EvaluateTest(unittest.TestCase):
    def test_stroke_themes_say_not_measured_and_unknown_themes_fail(self):
        out = evaluate(["adjustment", "around_backhand"], {"raw": {}}, [], None)
        self.assertEqual([t["verdict"] for t in out], ["not measured", "not measured"])
        with self.assertRaises(ValueError):
            evaluate(["footwork"], {"raw": {}}, [], None)


class FeedingStillWorksTest(unittest.TestCase):
    def test_coach_started_chain_is_fed(self):
        swings = [sw(1, 9, "far"), sw(2.2, 3, "near"), sw(3.4, 9, "far"), sw(10, 3, "near"), sw(11.2, 4, "far")]
        f = feeding_summary(swings, 25.0, coach=9)
        self.assertEqual((f["fed_rallies"], f["live_rallies"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
