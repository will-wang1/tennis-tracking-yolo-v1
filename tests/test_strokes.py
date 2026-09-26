import unittest

import numpy as np

from src.analysis.strokes import BodyFrame, classify_swing, resting_lateral, swing_peaks

FPS = 25.0
NEAR_BASELINE = (5.0, 23.0)   # near half: player faces AWAY from the camera
FAR_BASELINE = (5.0, 0.5)     # far half: player faces the camera
NEAR_NET_FAR_SIDE = (5.0, 6.0)


def _frame(f, rw=(0.0, -0.5), lw=(-0.3, -0.5), nose=(0.0, -1.3)):
    return BodyFrame(
        frame=f, left_wrist=np.array(lw, float), right_wrist=np.array(rw, float),
        nose=np.array(nose, float), shoulder_mid=np.array([0.0, -1.0]),
        torso_px=100.0, hip_px=np.array([0.0, 0.0]), wrist_conf=(0.9, 0.9),
    )


def _swing(peak=50, rw_path=None, n=100, rest=(0.3, -0.5)):
    """Frames at rest, with the right wrist following rw_path(offset_frames)
    in the 0.4s either side of `peak`."""
    frames = {}
    for f in range(n):
        o = f - peak
        rw = rw_path(o) if rw_path is not None and -12 <= o <= 12 else rest
        frames[f] = _frame(f, rw=rw)
    return frames


class ServeTest(unittest.TestCase):
    def test_racket_arm_above_the_head_at_a_baseline_is_a_serve(self):
        frames = _swing(rw_path=lambda o: (0.2, -2.3) if -6 <= o <= 0 else (0.2, -0.5))
        stroke, side, ev = classify_swing(frames, 50, FPS, "right", FAR_BASELINE)
        self.assertEqual(stroke, "serve")

    def test_the_same_motion_away_from_the_baseline_is_an_overhead(self):
        frames = _swing(rw_path=lambda o: (0.2, -2.3) if -6 <= o <= 0 else (0.2, -0.5))
        self.assertEqual(classify_swing(frames, 50, FPS, "right", (5.0, 17.0))[0], "overhead")

    def test_a_high_follow_through_after_contact_is_not_a_serve(self):
        # Topspin groundstrokes finish over the head; only BEFORE contact counts.
        frames = _swing(rw_path=lambda o: (-0.5, -2.3) if o > 0 else (0.6 - 0.1 * (o + 12), -0.6))
        self.assertNotEqual(classify_swing(frames, 50, FPS, "right", NEAR_BASELINE)[0], "serve")


class SideTest(unittest.TestCase):
    def test_right_hander_seen_from_behind_forehand_sweeps_right_to_left(self):
        # facing away: player's right is image right; forehand crosses toward image left
        frames = _swing(rw_path=lambda o: (0.8 - 0.1 * (o + 12), -0.6))
        stroke, side, _ = classify_swing(frames, 50, FPS, "right", NEAR_BASELINE)
        self.assertEqual(side, "forehand")

    def test_right_hander_seen_from_behind_backhand_sweeps_left_to_right(self):
        frames = _swing(rw_path=lambda o: (-0.8 + 0.1 * (o + 12), -0.6))
        self.assertEqual(classify_swing(frames, 50, FPS, "right", NEAR_BASELINE)[1], "backhand")

    def test_the_same_image_motion_means_the_opposite_for_a_player_facing_the_camera(self):
        frames = _swing(rw_path=lambda o: (-0.8 + 0.1 * (o + 12), -0.6))
        self.assertEqual(classify_swing(frames, 50, FPS, "right", FAR_BASELINE)[1], "forehand")

    def test_a_left_hander_flips_it_again(self):
        frames = {f: _frame(f, lw=b.right_wrist.tolist(), rw=(0.0, -0.5)) for f, b in
                  _swing(rw_path=lambda o: (0.8 - 0.1 * (o + 12), -0.6)).items()}
        self.assertEqual(classify_swing(frames, 50, FPS, "left", NEAR_BASELINE)[1], "backhand")


class VolleyAndSliceTest(unittest.TestCase):
    def test_near_the_net_is_a_volley_judged_against_the_resting_hand(self):
        # Far half, so the player FACES the camera: image-right is their LEFT.
        # Racket hand rests at image x +0.3; contact further to image-right
        # (+0.7) is further to their left - a BACKHAND volley - even though
        # nothing crossed the body's midline.
        frames = _swing(rw_path=lambda o: (0.7, -0.6), rest=(0.3, -0.5))
        rest = resting_lateral(frames, "right")
        stroke, _, _ = classify_swing(frames, 50, FPS, "right", NEAR_NET_FAR_SIDE, rest_lateral=rest)
        self.assertEqual(stroke, "backhand_volley")

    def test_contact_toward_the_racket_side_is_a_forehand_volley(self):
        frames = _swing(rw_path=lambda o: (-0.1, -0.6), rest=(0.3, -0.5))  # toward image-left = their right
        rest = resting_lateral(frames, "right")
        stroke, _, _ = classify_swing(frames, 50, FPS, "right", NEAR_NET_FAR_SIDE, rest_lateral=rest)
        self.assertEqual(stroke, "forehand_volley")

    def test_high_to_low_groundstroke_is_a_slice(self):
        frames = _swing(rw_path=lambda o: (-0.8 + 0.1 * (o + 12), -1.2 + 0.08 * (o + 12)))
        self.assertEqual(classify_swing(frames, 50, FPS, "right", NEAR_BASELINE)[0], "slice")

    def test_low_to_high_groundstroke_is_not(self):
        frames = _swing(rw_path=lambda o: (-0.8 + 0.1 * (o + 12), -0.2 - 0.08 * (o + 12)))
        self.assertEqual(classify_swing(frames, 50, FPS, "right", NEAR_BASELINE)[0], "backhand")


class SwingPeaksTest(unittest.TestCase):
    def test_one_fast_swing_is_one_peak_and_standing_still_is_none(self):
        still = {f: _frame(f) for f in range(100)}
        self.assertEqual(swing_peaks(still, FPS), [])
        swing = _swing(rw_path=lambda o: (0.8 - 0.15 * (o + 12), -0.6))
        peaks = swing_peaks(swing, FPS)
        self.assertEqual(len(peaks), 1)
        self.assertLess(abs(peaks[0][0] - 50), 14)


if __name__ == "__main__":
    unittest.main()
