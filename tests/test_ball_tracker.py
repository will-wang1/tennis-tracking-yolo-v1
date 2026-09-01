import unittest

from src.detection.ball_detector import Detection
from src.tracking.ball_tracker import BallTracker


class ImplausibleReappearanceTest(unittest.TestCase):
    """A detection that reappears after a gap suspiciously fast and then
    immediately goes suspiciously still - see BallTracker's module
    docstring and _reject_implausible_reappearances for why this specific
    combination, not speed or duration alone, is what's safe to act on."""

    def setUp(self):
        # smoothing_window=0 so positions can be checked exactly; the
        # reappearance check itself is tested directly on raw Detections
        self.tracker = BallTracker(max_pixels_per_frame=150.0, smoothing_window=0)

    def _run(self, detections):
        return self.tracker._reject_implausible_reappearances(detections)

    def test_rejects_a_fast_reappearance_that_immediately_freezes(self):
        # real ball approaching, then a 2-frame gap, then a detection 148px
        # away (near the 150 ceiling) that barely moves for several frames -
        # exactly the zverev artifact's own numbers
        detections = (
            [Detection(x=500.0 - 10.0 * i, y=500.0, confidence=0.9) for i in range(5)]
            + [None, None]
            + [Detection(x=796.0 + i, y=500.0, confidence=0.8) for i in range(6)]
        )

        cleaned = self._run(detections)

        self.assertTrue(all(d is None for d in cleaned[7:13]))
        self.assertIsNotNone(cleaned[4])  # the real ball beforehand is untouched

    def test_does_not_reject_a_fast_reappearance_that_keeps_moving(self):
        # a legitimate hard-hit continuation: reappears far away, but keeps
        # covering real ground afterward rather than freezing - nothing
        # about this is unphysical, so it must survive
        detections = (
            [Detection(x=500.0 - 10.0 * i, y=500.0, confidence=0.9) for i in range(5)]
            + [None, None]
            + [Detection(x=796.0 + 40.0 * i, y=500.0, confidence=0.8) for i in range(6)]
        )

        cleaned = self._run(detections)

        self.assertTrue(all(d is not None for d in cleaned[7:13]))

    def test_does_not_reject_a_long_slow_run_with_no_preceding_jump(self):
        # duration ALONE is unsafe to act on - measured against hand labels,
        # a real ball can sit under 10px/frame for 8-12 frames near a real
        # bounce, so a plain "many slow frames" rule would break those too
        detections = [Detection(x=500.0 + 0.5 * i, y=500.0, confidence=0.9) for i in range(15)]

        cleaned = self._run(detections)

        self.assertTrue(all(d is not None for d in cleaned))

    def test_a_small_gradual_gap_does_not_trigger_the_check(self):
        # a modest, physically ordinary jump across a short gap must not be
        # mistaken for the artifact just because a gap preceded it
        detections = (
            [Detection(x=500.0 + 10.0 * i, y=500.0, confidence=0.9) for i in range(5)]
            + [None, None]
            + [Detection(x=560.0 + 2.0 * i, y=500.0, confidence=0.9) for i in range(6)]
        )

        cleaned = self._run(detections)

        # unchanged from the input - the pre-existing gap at 5,6 stays, and
        # nothing else is newly rejected
        self.assertEqual(cleaned, detections)

    def test_rejection_is_capped_at_max_reappearance_reject(self):
        tracker = BallTracker(
            max_pixels_per_frame=150.0, smoothing_window=0, max_reappearance_reject=3
        )
        detections = (
            [Detection(x=500.0 - 10.0 * i, y=500.0, confidence=0.9) for i in range(5)]
            + [None, None]
            + [Detection(x=796.0 + i, y=500.0, confidence=0.8) for i in range(10)]
        )

        cleaned = tracker._reject_implausible_reappearances(detections)

        # only up to the cap is rejected - the rest of the (still-frozen)
        # run is left for a human/other mechanism, not silently erased
        # forever
        self.assertIsNotNone(cleaned[11])

    def test_the_reference_point_survives_a_rejection_unchanged(self):
        # after rejecting a run, the NEXT comparison must be judged against
        # the true last-good point, not against the rejected run's own
        # (fake) position. Proven the direct way: a real continuation from
        # the TRUE prior trajectory (not from the fake cluster's position,
        # which sits far off to one side) must survive untouched. If `last`
        # had been wrongly updated to the fake cluster instead, this same
        # detection would look like another huge, suspicious jump - FROM
        # the wrong place - and could be rejected in turn.
        detections = (
            [Detection(x=500.0 - 10.0 * i, y=500.0, confidence=0.9) for i in range(5)]  # 0-4, real
            + [None, None]  # 5-6
            + [Detection(x=796.0 + i, y=500.0, confidence=0.8) for i in range(4)]  # 7-10, the artifact
            + [None, None]  # 11-12
            # continues the REAL trajectory from frame 4 (x=460, -10/frame),
            # not from the artifact's own position (x=799) - only correct
            # if `last` still points at frame 4
            + [Detection(x=460.0 - 10.0 * i, y=500.0, confidence=0.9) for i in range(1, 4)]  # 13-15
        )

        cleaned = self._run(detections)

        self.assertTrue(all(d is None for d in cleaned[7:11]))  # the artifact itself: rejected
        self.assertTrue(all(d is not None for d in cleaned[13:16]))  # the real continuation: kept
        self.assertAlmostEqual(cleaned[13].x, 450.0)

    def test_survives_no_detections_at_all(self):
        self.assertEqual(self._run([None] * 5), [None] * 5)

    def test_survives_a_single_detection(self):
        detections = [Detection(x=1.0, y=1.0, confidence=0.9)]
        self.assertEqual(self._run(detections), detections)

    def test_end_to_end_the_gap_is_left_missing_not_interpolated_across(self):
        # combined with track()'s own max_interpolation_gap, a rejected run
        # this long is left as a genuine gap rather than papered over with
        # a straight line through where the artifact used to be
        tracker = BallTracker(max_pixels_per_frame=150.0, max_interpolation_gap=8, smoothing_window=0)
        detections = (
            [Detection(x=500.0 - 10.0 * i, y=500.0, confidence=0.9) for i in range(5)]
            + [None, None]
            + [Detection(x=796.0 + i, y=500.0, confidence=0.8) for i in range(8)]
            + [Detection(x=1000.0, y=500.0, confidence=0.9)]
        )

        positions = tracker.track(detections)
        frames = {p.frame_idx for p in positions}

        self.assertNotIn(9, frames)  # inside the rejected run
        self.assertIn(4, frames)
        self.assertIn(15, frames)  # the real detection right after the run


class BallTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = BallTracker(max_pixels_per_frame=150.0, max_interpolation_gap=8)

    def test_fills_short_gap_by_interpolation(self):
        detections = [
            Detection(x=0.0, y=0.0, confidence=0.9),
            None,
            None,
            Detection(x=30.0, y=0.0, confidence=0.9),
        ]
        positions = self.tracker.track(detections)

        self.assertEqual([p.frame_idx for p in positions], [0, 1, 2, 3])
        self.assertAlmostEqual(positions[1].x, 10.0)
        self.assertAlmostEqual(positions[2].x, 20.0)
        self.assertTrue(positions[1].interpolated)
        self.assertTrue(positions[2].interpolated)
        self.assertFalse(positions[0].interpolated)
        self.assertFalse(positions[3].interpolated)

    def test_leaves_long_gap_unfilled(self):
        detections = (
            [Detection(x=0.0, y=0.0, confidence=0.9)]
            + [None] * 10
            + [Detection(x=1000.0, y=0.0, confidence=0.9)]
        )
        positions = self.tracker.track(detections)

        frame_indices = {p.frame_idx for p in positions}
        self.assertIn(0, frame_indices)
        self.assertIn(11, frame_indices)
        # the 10-frame gap exceeds max_interpolation_gap=8, so none of the
        # in-between frames should have been fabricated
        self.assertTrue(all(1 <= i <= 10 and i not in frame_indices for i in range(1, 11)))

    def test_does_not_interpolate_before_first_or_after_last_detection(self):
        detections = [None, None, Detection(x=5.0, y=5.0, confidence=0.9), None, None]
        positions = self.tracker.track(detections)

        self.assertEqual([p.frame_idx for p in positions], [2])

    def test_rejects_implausible_jump_as_outlier(self):
        detections = [
            Detection(x=0.0, y=0.0, confidence=0.9),
            Detection(x=5000.0, y=5000.0, confidence=0.9),  # impossible jump -> false positive
            Detection(x=10.0, y=0.0, confidence=0.9),
        ]
        positions = self.tracker.track(detections)
        by_frame = {p.frame_idx: p for p in positions}

        # frame 1's outlier should be discarded and interpolated between
        # frame 0 and frame 2 instead of trusted as-is
        self.assertAlmostEqual(by_frame[1].x, 5.0)
        self.assertTrue(by_frame[1].interpolated)

    def test_rejects_persistent_static_lockon(self):
        # simulates YOLO locking onto something fixed on screen (a graphic,
        # a line, the net cord) and re-detecting it near the same spot every
        # frame - unlike outlier rejection, nothing here looks like a big
        # jump, so only the "hasn't gone anywhere in N frames" check
        # catches it. Default: static_lockon_frames=10, radius=20px.
        detections = [Detection(x=500.0, y=500.0, confidence=0.9) for _ in range(25)]
        positions = self.tracker.track(detections)

        frame_indices = [p.frame_idx for p in positions]
        # accepted up to the point the lock-on window fills, then rejected
        # for the rest of the (still-static) sequence
        self.assertEqual(frame_indices, list(range(9)))

    def test_does_not_reject_jitter_within_a_brief_pause(self):
        # a short run of positions that wander a little (sub-pixel/
        # localization noise) but stay confined - shorter than
        # static_lockon_frames - is plausible ball behavior (e.g. near the
        # apex of a lob) and should be trusted, not treated as a lock-on
        detections = [
            Detection(x=500.0 + (i % 2) * 3, y=500.0, confidence=0.9) for i in range(8)
        ]
        positions = self.tracker.track(detections)

        self.assertEqual([p.frame_idx for p in positions], list(range(8)))

    def test_empty_input(self):
        self.assertEqual(self.tracker.track([]), [])

    def test_all_missing(self):
        self.assertEqual(self.tracker.track([None, None, None]), [])


if __name__ == "__main__":
    unittest.main()
