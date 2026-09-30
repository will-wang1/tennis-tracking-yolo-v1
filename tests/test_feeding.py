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


class PlaySegmentsTest(unittest.TestCase):
    def rally(self, t, fed, shots=3):
        return {"start_s": t, "end_s": t + 2, "fed": fed, "shots": shots}

    def test_a_fed_block_then_live_rallying(self):
        from src.session.feeding import play_segments, play_time

        rallies = [self.rally(t, True, 2) for t in range(0, 40, 4)] + [self.rally(t, False, 6) for t in range(40, 80, 5)]
        segs = play_segments(rallies, [{"start_s": 0, "end_s": 80}])
        self.assertEqual([(s["kind"], s["start_s"], s["end_s"]) for s in segs], [("fed", 0, 40.0), ("live", 40.0, 80)])
        self.assertEqual(segs[0]["rallies"], 10)
        self.assertEqual(play_time(segs)["shares"]["fed"], 0.5)

    def test_one_coach_swing_in_live_play_is_not_a_fed_stretch(self):
        from src.session.feeding import play_segments

        rallies = [self.rally(t, False) for t in range(0, 30, 3)] + [self.rally(35, True)] + [self.rally(t, False) for t in range(40, 60, 3)]
        self.assertEqual([s["kind"] for s in play_segments(rallies, [{"start_s": 0, "end_s": 60}])], ["live"])

    def test_both_at_once_is_mixed_and_long_silence_is_not_play(self):
        from src.session.feeding import play_segments

        rallies = []
        for t in range(0, 20, 2):
            rallies.append(self.rally(t, t % 4 == 0))
        segs = play_segments(rallies, [{"start_s": 0, "end_s": 60}])
        self.assertEqual(segs[0]["kind"], "mixed")
        self.assertLess(segs[-1]["end_s"], 60)  # nothing started after 20s: not counted as play past the carry
