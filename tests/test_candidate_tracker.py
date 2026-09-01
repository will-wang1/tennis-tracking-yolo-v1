import unittest

from src.detection.ball_detector import Detection
from src.tracking.candidate_tracker import track_candidates


def _d(x, y, confidence=0.9):
    return Detection(x=float(x), y=float(y), confidence=confidence)


def _straight(count=12, x0=100.0, y0=100.0, step=10.0, confidence=0.9):
    """A ball moving steadily, one candidate per frame."""
    return [[_d(x0 + step * i, y0 + step * i, confidence)] for i in range(count)]


class TrackCandidatesTest(unittest.TestCase):
    def test_follows_a_single_clean_candidate_per_frame(self):
        chosen = track_candidates(_straight())

        self.assertEqual([c is not None for c in chosen], [True] * 12)
        self.assertEqual(chosen[0].x, 100.0)
        self.assertEqual(chosen[-1].x, 210.0)

    def test_prefers_the_smooth_path_over_the_confident_one(self):
        # a decoy sits still and scores HIGHER every frame - a net cord or a
        # line lighting up the heatmap. Only the trajectory tells them apart.
        frames = []
        for i in range(12):
            ball = _d(100 + 10 * i, 100 + 10 * i, confidence=0.55)
            decoy = _d(600, 600, confidence=0.95)
            frames.append([decoy, ball])

        chosen = track_candidates(frames)

        self.assertTrue(all(c.x != 600 for c in chosen if c is not None))
        self.assertEqual(chosen[-1].x, 210.0)

    def test_recovers_the_ball_when_it_is_not_the_strongest_peak(self):
        # mid-flight the ball blurs and drops to second place for 3 frames
        frames = []
        for i in range(14):
            ball = _d(100 + 10 * i, 100, confidence=0.3 if 5 <= i <= 7 else 0.9)
            if 5 <= i <= 7:
                frames.append([_d(700, 400, confidence=0.8), ball])
            else:
                frames.append([ball])

        chosen = track_candidates(frames)

        self.assertEqual([c.x for c in chosen[5:8]], [150.0, 160.0, 170.0])

    def test_skips_frames_where_the_ball_is_genuinely_absent(self):
        frames = _straight(14)
        for i in (6, 7, 8):
            frames[i] = []  # occluded by a player

        chosen = track_candidates(frames)

        self.assertEqual([chosen[i] for i in (6, 7, 8)], [None, None, None])
        self.assertIsNotNone(chosen[9])
        self.assertEqual(chosen[9].x, 190.0)

    def test_bridges_a_gap_rather_than_restarting_after_it(self):
        # the path either side of a dropout should be one track, so the
        # frames after it continue the same line
        frames = _straight(16)
        for i in (7, 8, 9, 10):
            frames[i] = []

        chosen = track_candidates(frames)

        self.assertEqual(chosen[6].x, 160.0)
        self.assertEqual(chosen[11].x, 210.0)

    def test_refuses_a_physically_impossible_jump(self):
        # nothing plausible after the gap: the far candidate would need to
        # travel further than a ball can, so it is not joined to the track
        frames = _straight(6) + [[]] * 2 + [[_d(20000, 20000)]]

        chosen = track_candidates(frames, max_pixels_per_frame=150.0)

        self.assertIsNone(chosen[-1])

    def test_still_follows_a_real_bounce(self):
        # a bounce is a sharp direction change, so the turn penalty must not
        # be so strong that the track refuses to follow one
        frames = [[_d(100 + 10 * i, 100 + 20 * i)] for i in range(6)]
        frames += [[_d(160 + 10 * i, 200 - 20 * i)] for i in range(1, 7)]

        chosen = track_candidates(frames)

        self.assertTrue(all(c is not None for c in chosen))

    def test_handles_an_empty_sequence(self):
        self.assertEqual(track_candidates([]), [])

    def test_handles_a_sequence_with_no_candidates_at_all(self):
        self.assertEqual(track_candidates([[], [], []]), [None, None, None])

    def test_returns_one_entry_per_frame(self):
        frames = _straight(9)
        frames[4] = []

        self.assertEqual(len(track_candidates(frames)), 9)


if __name__ == "__main__":
    unittest.main()
