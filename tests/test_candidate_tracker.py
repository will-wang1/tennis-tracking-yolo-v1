import unittest

from src.detection.ball_detector import Detection
from src.tracking.candidate_tracker import (
    BallTrack,
    count_simultaneous_frames,
    track_ball_paths,
    track_candidates,
)


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


def _two_balls(count=14, separation=400.0):
    """Two balls in flight at once, well apart - the coaching case
    (`dingles_serve_volley` is a two-ball drill)."""
    frames = []
    for i in range(count):
        a = _d(100 + 10 * i, 100 + 8 * i)
        b = _d(100 + 10 * i + separation, 500 - 6 * i)
        frames.append([a, b])
    return frames


class BallTrackTest(unittest.TestCase):
    def test_overlaps_is_true_only_when_a_frame_is_shared(self):
        a = BallTrack({0: _d(0, 0), 1: _d(1, 1), 2: _d(2, 2)})
        b = BallTrack({2: _d(9, 9), 3: _d(8, 8)})
        c = BallTrack({7: _d(5, 5), 8: _d(6, 6)})

        self.assertTrue(a.overlaps(b))  # share frame 2 - cannot be one ball
        self.assertFalse(a.overlaps(c))  # disjoint - may be one ball, gapped

    def test_as_frame_list_places_detections_at_their_own_frames(self):
        track = BallTrack({1: _d(10, 10), 3: _d(30, 30)})

        out = track.as_frame_list(5)

        self.assertEqual(len(out), 5)
        self.assertIsNone(out[0])
        self.assertEqual(out[1].x, 10.0)
        self.assertIsNone(out[2])
        self.assertEqual(out[3].x, 30.0)

    def test_as_frame_list_drops_frames_outside_the_clip(self):
        track = BallTrack({0: _d(1, 1), 9: _d(9, 9)})
        self.assertEqual([d is not None for d in track.as_frame_list(3)], [True, False, False])


class TrackBallPathsTest(unittest.TestCase):
    def test_finds_both_balls_when_two_are_in_flight_at_once(self):
        tracks = track_ball_paths(_two_balls())

        self.assertGreaterEqual(len(tracks), 2)
        self.assertTrue(tracks[0].overlaps(tracks[1]))
        xs = {round(t.detections[0].x) for t in tracks[:2]}
        self.assertEqual(xs, {100, 500})

    def test_reports_the_frames_where_two_balls_coexist(self):
        multi = count_simultaneous_frames(track_ball_paths(_two_balls()))

        self.assertEqual(len(multi), 14)
        self.assertTrue(all(n >= 2 for n in multi.values()))

    def test_a_path_hugging_another_is_one_ball_seen_twice_not_two(self):
        # Same flight, with a second blob 6px away every frame - the shadow
        # that leaving frames open would otherwise admit as a second ball.
        frames = [
            [_d(100 + 10 * i, 100 + 10 * i), _d(106 + 10 * i, 104 + 10 * i)] for i in range(14)
        ]

        tracks = track_ball_paths(frames)

        self.assertEqual(len(tracks), 1)

    def test_balls_that_cross_closely_both_survive(self):
        # They pass within a few px mid-flight but are far apart either
        # side, so the MEDIAN separation - not the minimum - is what keeps
        # both. See _is_shadow.
        frames = []
        for i in range(16):
            frames.append([_d(100 + 30 * i, 300), _d(550 - 30 * i, 300 + 2 * i)])

        tracks = track_ball_paths(frames)

        self.assertGreaterEqual(len(tracks), 2)

    def test_single_ball_footage_yields_one_track_and_no_overlap(self):
        tracks = track_ball_paths(_straight(count=14))

        self.assertEqual(len(tracks), 1)
        self.assertEqual(count_simultaneous_frames(tracks), {})

    def test_no_candidates_yields_no_tracks(self):
        self.assertEqual(track_ball_paths([]), [])
        self.assertEqual(track_ball_paths([[], [], []]), [])

    def test_max_tracks_bounds_the_search(self):
        tracks = track_ball_paths(_two_balls(), max_tracks=1)
        self.assertEqual(len(tracks), 1)

    def test_track_candidates_still_returns_one_detection_per_frame(self):
        # The flattening wrapper must keep its original contract even though
        # it now shares an extractor with the multi-ball path.
        chosen = track_candidates(_two_balls())

        self.assertEqual(len(chosen), 14)
        self.assertTrue(all(c is None or isinstance(c, Detection) for c in chosen))
