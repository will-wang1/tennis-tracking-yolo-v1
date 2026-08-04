import unittest

from src.detection.ball_detector import Detection
from src.tracking.ball_tracker import BallTracker


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
