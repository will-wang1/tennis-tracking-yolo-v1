import unittest

from src.session.feeding import feeding_summary, rallies
from src.session.insights import play_note


def sw(t, player, fps=25):
    return {"frame": int(t * fps), "player": player}


class RalliesTest(unittest.TestCase):
    def test_swings_split_on_gaps_and_the_first_swing_starts_it(self):
        swings = [sw(1, 9), sw(2, 3), sw(3.5, 4), sw(10, 3), sw(11, 4), sw(12.5, 3)]
        rs = rallies(swings, 25.0, coach=9)
        self.assertEqual([(r.starter, r.shots, r.fed(9)) for r in rs], [(9, 3, True), (3, 3, False)])
        self.assertEqual(rs[0].coach_shots, 1)


class FeedingSummaryTest(unittest.TestCase):
    def test_counts_and_shares(self):
        swings = [sw(1, 9), sw(2, 3), sw(10, 9), sw(11, 4), sw(20, 3), sw(21, 4), sw(22, 3), sw(23, 4)]
        f = feeding_summary(swings, 25.0, coach=9)
        self.assertEqual((f["rallies"], f["fed_rallies"], f["live_rallies"]), (3, 2, 1))
        self.assertEqual(f["shots_per_rally"], {"fed": 2.0, "live": 4.0, "all": 2.7})
        self.assertAlmostEqual(f["coach_shot_share"], 0.25)

    def test_no_coach_means_nothing_counts_as_fed(self):
        f = feeding_summary([sw(1, 9), sw(2, 3)], 25.0, coach=None)
        self.assertEqual(f["fed_rallies"], 0)
        self.assertTrue(any("No coach" in c for c in f["caveats"]))


class PlayNoteTest(unittest.TestCase):
    def test_wording(self):
        live = feeding_summary([sw(t, 3 if i % 2 else 4) for i, t in enumerate(range(0, 40, 5))], 25.0, coach=9)
        self.assertEqual(play_note(live)["title"], "Mostly live play")
        fed = feeding_summary([sw(t, 9) for t in range(0, 40, 5)], 25.0, coach=9)
        self.assertEqual(play_note(fed)["title"], "Mostly fed")
        self.assertEqual(play_note(feeding_summary([sw(1, 3)], 25.0, coach=None))["title"], "Fed or live: unknown")
        self.assertIsNone(play_note(feeding_summary([], 25.0, coach=9)))


if __name__ == "__main__":
    unittest.main()
