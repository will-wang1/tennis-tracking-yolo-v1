import unittest

import numpy as np

from src.session.structure import drill_segments, players_gathered, queue_and_idle, still_mask

FPS = 25.0


def _mask(n, *ranges_s):
    m = np.zeros(n, dtype=bool)
    for a, b in ranges_s:
        m[int(a * FPS):int(b * FPS)] = True
    return m


class DrillSegmentsTest(unittest.TestCase):
    def test_short_gaps_stay_inside_one_drill(self):
        n = int(60 * FPS)
        segs = drill_segments(_mask(n, (0, 20), (25, 60)), np.ones(n, bool), FPS)  # 5s gap
        self.assertEqual([s.kind for s in segs], ["drill"])

    def test_a_long_gap_is_a_break_between_two_drills(self):
        n = int(100 * FPS)
        segs = drill_segments(_mask(n, (0, 30), (60, 100)), np.ones(n, bool), FPS)  # 30s gap
        self.assertEqual([s.kind for s in segs], ["drill", "break", "drill"])
        self.assertAlmostEqual(segs[1].to_dict(FPS)["duration_s"], 30.0, delta=0.1)

    def test_a_camera_cutaway_does_not_split_a_drill(self):
        # 20s gap, but 18s of it is a cutaway: only 2s of ANALYSED gap
        n = int(80 * FPS)
        analysed = ~_mask(n, (31, 49))
        segs = drill_segments(_mask(n, (0, 30), (50, 80)), analysed, FPS)
        self.assertEqual([s.kind for s in segs], ["drill"])

    def test_no_play_no_segments(self):
        self.assertEqual(drill_segments(np.zeros(100, bool), np.ones(100, bool), FPS), [])


class StillAndQueueTest(unittest.TestCase):
    def test_far_court_jitter_still_counts_as_standing(self):
        rng = np.random.default_rng(0)
        pos = {f: (5.0 + rng.normal(0, 0.1), -2.0 + rng.normal(0, 0.1)) for f in range(250)}
        self.assertGreater(still_mask(pos, FPS, 250)[25:225].mean(), 0.9)

    def test_walking_is_not_still(self):
        pos = {f: (0.04 * f, 10.0) for f in range(250)}  # 1 m/s
        self.assertFalse(still_mask(pos, FPS, 250).any())

    def test_waiting_behind_the_baseline_is_queue_standing_on_court_is_only_idle(self):
        n = 500
        drill = np.ones(n, bool)
        behind = {f: (5.0, -2.0) for f in range(n)}  # 2m behind the far baseline
        on_court = {f: (5.0, 3.0) for f in range(n)}
        q = queue_and_idle(behind, FPS, drill)
        # the first and last second can't be judged: stillness is measured
        # over a 2s window centred on each frame
        self.assertGreaterEqual(q["queue_share"], 0.85)
        c = queue_and_idle(on_court, FPS, drill)
        self.assertEqual(c["queue_s"], 0.0)
        self.assertGreater(c["idle_s"], 15.0)

    def test_standing_still_only_counts_during_drills(self):
        n = 500
        q = queue_and_idle({f: (5.0, -2.0) for f in range(n)}, FPS, np.zeros(n, bool))
        self.assertEqual((q["idle_s"], q["queue_s"]), (0.0, 0.0))

    def test_short_stops_are_not_idle(self):
        n = 500
        # stops for 2s at a time between 1 m/s walks
        pos, x = {}, 0.0
        for f in range(n):
            if (f // 50) % 2 == 0:
                x += 0.04
            pos[f] = (x, 10.0)
        self.assertEqual(queue_and_idle(pos, FPS, np.ones(n, bool))["idle_s"], 0.0)


class GatheredTest(unittest.TestCase):
    def test_huddle_vs_spread_out(self):
        frames = range(100)
        huddle = {p: {f: (5.0 + p * 0.5, 12.0) for f in frames} for p in range(5)}
        spread = {p: {f: (p * 3.0, p * 5.0) for f in frames} for p in range(5)}
        self.assertTrue(players_gathered(huddle, frames, FPS))
        self.assertFalse(players_gathered(spread, frames, FPS))


if __name__ == "__main__":
    unittest.main()
