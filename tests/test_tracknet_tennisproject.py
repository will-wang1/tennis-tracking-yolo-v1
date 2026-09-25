import unittest

import numpy as np

from src.detection.tracknet_ball_detector import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    tennisproject_postprocess,
    track_ball_tennisproject,
)


def _disc(value, cx=300, cy=150, radius=4):
    """An argmax map shaped like TrackNet's output (1, H*W) with one blob."""
    yy, xx = np.mgrid[0:MODEL_HEIGHT, 0:MODEL_WIDTH]
    heat = np.where((xx - cx) ** 2 + (yy - cy) ** 2 <= radius * radius, value, 0)
    return heat.astype(np.int64).reshape(1, -1)


class TennisProjectPostprocessTest(unittest.TestCase):
    def test_finds_a_blob_and_scales_it_to_1280x720(self):
        x, y = tennisproject_postprocess(_disc(50), [None, None])
        self.assertAlmostEqual(float(x), 600.0, delta=4.0)
        self.assertAlmostEqual(float(y), 300.0, delta=4.0)

    def test_the_upstream_overflow_is_reproduced_not_fixed(self):
        # Upstream multiplies an argmax already in 0..255 by 255 and casts to
        # uint8, which wraps: 255 -> 1, below the 127 threshold. So a blob at
        # the model's HIGHEST class vanishes. Almost certainly a bug upstream,
        # kept on purpose - the bounce model was trained on tracks made this
        # way. If this test ever fails, the method is no longer TennisProject's.
        self.assertEqual(tennisproject_postprocess(_disc(255), [None, None]), (None, None))

    def test_a_blob_far_from_the_previous_ball_is_rejected(self):
        self.assertEqual(tennisproject_postprocess(_disc(50), [100.0, 100.0]), (None, None))

    def test_a_blob_near_the_previous_ball_is_kept(self):
        x, _ = tennisproject_postprocess(_disc(50), [590.0, 290.0])
        self.assertIsNotNone(x)

    def test_an_empty_map_finds_nothing(self):
        empty = np.zeros((1, MODEL_HEIGHT * MODEL_WIDTH), dtype=np.int64)
        self.assertEqual(tennisproject_postprocess(empty, [None, None]), (None, None))


class TrackBallTennisProjectTest(unittest.TestCase):
    def test_refuses_frames_that_are_not_1280x720(self):
        # Upstream hardcodes x2 scaling from 640x360, so any other size gives
        # coordinates in the wrong units for the bounce model.
        frames = iter([np.zeros((1080, 1920, 3), dtype=np.uint8)])
        with self.assertRaises(ValueError):
            track_ball_tennisproject(frames, "weights/tracknet_pretrained.pt", device="cpu")


if __name__ == "__main__":
    unittest.main()
