import unittest

import numpy as np

from src.analysis.scene_cuts import continuous_ranges, detect_scene_cuts, frame_distances


def _gradient_frame(offset: int, size: int = 64) -> np.ndarray:
    """A smoothly shifting colour gradient - stands in for an ordinary
    pan/zoom within one continuous shot: it changes frame to frame, but
    gradually, never a full-frame jump."""
    ramp = (np.arange(size) + offset) % 256
    frame = np.tile(ramp, (size, 1)).astype(np.uint8)
    return np.stack([frame, np.roll(frame, 20), np.roll(frame, 40)], axis=-1)


def _solid_frame(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def _panning_sequence(count: int = 30) -> list[np.ndarray]:
    return [_gradient_frame(offset=i * 2) for i in range(count)]


class DetectSceneCutsTest(unittest.TestCase):
    def test_a_smooth_pan_has_no_cuts(self):
        frames = _panning_sequence(30)
        self.assertEqual(detect_scene_cuts(frames), [])

    def test_a_static_scene_has_no_cuts(self):
        frames = [_solid_frame((40, 90, 60)) for _ in range(30)]
        self.assertEqual(detect_scene_cuts(frames), [])

    def test_a_hard_colour_swap_is_detected_at_the_right_frame(self):
        before = _panning_sequence(20)
        # a totally different, saturated colour stands in for a cut to a
        # different camera/scene entirely - held long enough that cutting
        # BACK to the pan afterward doesn't need judging here too
        cut_to = [_solid_frame((0, 0, 255)) for _ in range(20)]
        frames = before + cut_to

        cuts = detect_scene_cuts(frames)

        self.assertEqual(cuts, [20])

    def test_two_cuts_are_both_found(self):
        a = _panning_sequence(20)
        b = [_solid_frame((0, 255, 0)) for _ in range(20)]
        c = _panning_sequence(20)
        frames = a + b + c

        cuts = detect_scene_cuts(frames)

        self.assertEqual(cuts, [20, 40])

    def test_fewer_than_two_frames_returns_no_cuts(self):
        self.assertEqual(detect_scene_cuts([]), [])
        self.assertEqual(detect_scene_cuts([_solid_frame((1, 2, 3))]), [])

    def test_too_short_a_sequence_to_judge_local_context_returns_no_cuts(self):
        # detect_scene_cuts needs enough neighbours to know what "surprising"
        # means locally - a handful of frames isn't enough to judge from,
        # so it declines rather than guessing off a near-empty baseline
        frames = _panning_sequence(3) + [_solid_frame((0, 0, 255))]
        self.assertEqual(detect_scene_cuts(frames), [])

    def test_a_noisy_stretch_that_crosses_the_threshold_several_times_collapses_to_one(self):
        # Reproduces a real false-positive found on actual broadcast footage
        # (see detect_scene_cuts's docstring): an animated on-screen graphic
        # disturbed several nearby frames enough that each independently
        # crossed the cut threshold, none of them a real camera change.
        # A handful of moderately-elevated frames close together should be
        # ONE reported cut, not several near-empty "shots" a few frames apart.
        before = _panning_sequence(20)
        noisy = [_solid_frame((80, 80, 200)), _solid_frame((90, 70, 190)), _solid_frame((70, 90, 210))]
        after = _panning_sequence(20)
        frames = before + noisy + after

        cuts = detect_scene_cuts(frames)

        self.assertEqual(len(cuts), 1)

    def test_two_genuinely_separate_cuts_further_apart_than_the_gap_both_survive(self):
        a = _panning_sequence(20)
        b = [_solid_frame((0, 255, 0)) for _ in range(20)]
        c = _panning_sequence(20)
        frames = a + b + c

        cuts = detect_scene_cuts(frames, min_gap_frames=10)

        self.assertEqual(cuts, [20, 40])


class FrameDistancesTest(unittest.TestCase):
    def test_one_value_per_consecutive_pair(self):
        frames = _panning_sequence(5)
        self.assertEqual(len(frame_distances(frames)), 4)

    def test_identical_frames_have_zero_distance(self):
        frames = [_solid_frame((10, 20, 30))] * 5
        self.assertTrue(all(d == 0.0 for d in frame_distances(frames)))

    def test_a_hard_swap_produces_a_much_larger_distance_than_a_pan(self):
        pan_distances = frame_distances(_panning_sequence(10))
        swap_distances = frame_distances([_solid_frame((10, 20, 30)), _solid_frame((0, 0, 255))])

        self.assertGreater(swap_distances[0], max(pan_distances))


class ContinuousRangesTest(unittest.TestCase):
    def test_no_cuts_is_one_range_covering_everything(self):
        self.assertEqual(continuous_ranges([], num_frames=50), [(0, 49)])

    def test_cuts_split_into_the_right_ranges(self):
        self.assertEqual(continuous_ranges([20, 40], num_frames=60), [(0, 19), (20, 39), (40, 59)])

    def test_duplicate_and_unordered_cuts_are_handled(self):
        self.assertEqual(continuous_ranges([40, 20, 20], num_frames=60), [(0, 19), (20, 39), (40, 59)])

    def test_a_cut_at_frame_zero_is_ignored(self):
        # frame 0 can't be a cut FROM anything - there's no prior frame -
        # and treating it as one would produce a zero-length leading range
        self.assertEqual(continuous_ranges([0, 30], num_frames=60), [(0, 29), (30, 59)])

    def test_zero_frames_is_no_ranges(self):
        self.assertEqual(continuous_ranges([], num_frames=0), [])


if __name__ == "__main__":
    unittest.main()
