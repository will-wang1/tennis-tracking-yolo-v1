import unittest

import numpy as np

from src.analysis.stroke_tcn import (
    CLASSES, N_FEATURES, SEQ_LEN, build_model, class_of_label, features, mirror_hand, window_times,
)


def _window(wrist_x=10.0):
    kp = np.zeros((SEQ_LEN, 17, 2))
    kp[:, :, 0], kp[:, :, 1] = 100.0, 100.0
    kp[:, 11], kp[:, 12] = (95.0, 150.0), (105.0, 150.0)  # hips
    kp[:, 10, 0] = 100.0 + wrist_x  # right wrist
    conf = np.ones((SEQ_LEN, 17))
    box = np.tile([80.0, 50.0, 120.0, 250.0], (SEQ_LEN, 1))  # 200px tall
    return kp, conf, box


class FeaturesTest(unittest.TestCase):
    def test_hip_centred_height_scaled_and_far_players_mirrored(self):
        kp, conf, box = _window()
        near = features(kp, conf, box, far=False)
        far = features(kp, conf, box, far=True)
        self.assertEqual(near.shape, (SEQ_LEN, N_FEATURES))
        self.assertAlmostEqual(near[0, 20], 10 / 200)  # right wrist x, joint 10 -> column 20
        self.assertAlmostEqual(far[0, 20], -10 / 200)

    def test_missing_frames_are_zero(self):
        kp, conf, box = _window()
        kp[5], conf[5], box[5] = np.nan, 0, np.nan
        f = features(kp, conf, box, far=False)
        self.assertTrue(np.all(f[5] == 0))
        self.assertTrue(np.isfinite(f).all())

    def test_mirroring_the_hand_swaps_sides_and_is_its_own_inverse(self):
        kp, conf, box = _window()
        f = features(kp, conf, box, far=False)
        m = mirror_hand(f)
        self.assertAlmostEqual(m[0, 18], -f[0, 20])  # right wrist becomes the left wrist, mirrored
        np.testing.assert_allclose(mirror_hand(m), f)


class LabelsTest(unittest.TestCase):
    def test_our_labels_map_to_classes(self):
        self.assertEqual(class_of_label({"stroke": "backhand", "slice": True}), "backhand_slice")
        self.assertEqual(class_of_label({"stroke": "forehand", "volley": True, "slice": True}), "forehand_volley")
        self.assertEqual(class_of_label({"stroke": "not a swing"}), "not_a_swing")
        self.assertIsNone(class_of_label({"stroke": "skip"}))

    def test_window_is_51_samples_around_the_swing(self):
        t = window_times(10.0)
        self.assertEqual(len(t), 51)
        self.assertAlmostEqual(t[0], 8.8)
        self.assertAlmostEqual(t[-1], 10.8)


class ModelTest(unittest.TestCase):
    def test_forward_shape(self):
        import torch

        model = build_model().eval()
        out = model(torch.zeros(3, SEQ_LEN, N_FEATURES), torch.tensor([True, False, True]))
        self.assertEqual(tuple(out.shape), (3, len(CLASSES)))


if __name__ == "__main__":
    unittest.main()
