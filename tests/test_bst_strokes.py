import unittest
from pathlib import Path

import numpy as np

from src.analysis.bst_strokes import (
    SEQ_LEN, PersonFrame, bones, build_input, fold_serve_tosses, normalize_joints, stroke_name,
)

WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "bst" / "bst_AP_JnB_bone.pt"


class NormalizeTest(unittest.TestCase):
    def test_box_centre_is_origin_and_unit_is_the_diagonal(self):
        box = (100.0, 200.0, 130.0, 240.0)  # 30 x 40, diagonal 50
        kp = np.array([[115.0, 220.0], [130.0, 240.0]] + [[0.0, 0.0]] * 15)
        out = normalize_joints(kp, box)
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[1], [0.3, 0.4])

    def test_bones_are_zero_where_a_joint_is_missing(self):
        joints = np.zeros((17, 2))
        joints[5] = (1.0, 1.0)
        joints[7] = (2.0, 3.0)
        b = bones(joints)
        self.assertEqual(b.shape, (19, 2))
        np.testing.assert_allclose(b[7], [1.0, 2.0])  # (5, 7) is the 8th pair
        self.assertTrue(np.all(b[0] == 0))


class BuildInputTest(unittest.TestCase):
    def test_shapes_padding_and_far_player_first(self):
        box = (0.0, 0.0, 30.0, 40.0)
        kp = np.full((17, 2), 10.0)
        far = [PersonFrame(kp, box, (5.485, 0.0))] * 5
        near = [PersonFrame(None, None, None)] * 5
        jnb, pos, shuttle, length = build_input(far, near, [(960.0, 540.0)] * 4 + [None], (1920, 1080))
        self.assertEqual((jnb.shape, pos.shape, shuttle.shape, length), ((SEQ_LEN, 2, 72), (SEQ_LEN, 2, 2), (SEQ_LEN, 2), 5))
        np.testing.assert_allclose(pos[0, 0], [0.5, 0.0])
        self.assertTrue(np.all(jnb[0, 1] == 0) and np.all(jnb[5:] == 0))
        np.testing.assert_allclose(shuttle[0], [0.5, 0.5])
        self.assertTrue(np.all(shuttle[4] == 0))


class StrokeNameTest(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(stroke_name("HNR", 0.9, 0.1, True), "forehand")
        self.assertEqual(stroke_name("HFL", 0.9, 0.1, True), "forehand")
        self.assertEqual(stroke_name("HFR", 0.9, 0.1, True), "backhand")
        self.assertEqual(stroke_name("HFR", 0.9, 0.1, True, left_handed=True), "forehand")
        self.assertEqual(stroke_name("HNR", 0.9, 0.85, True), "serve")
        self.assertEqual(stroke_name("HNR", 0.9, 0.1, False), "unsure")
        self.assertEqual(stroke_name("HNR", 0.5, 0.1, True), "unsure")

    def test_a_middling_serve_score_counts_only_from_the_baseline(self):
        self.assertEqual(stroke_name("HFL", 0.8, 0.6, True, at_baseline=True), "serve")
        self.assertEqual(stroke_name("HFL", 0.8, 0.6, True, at_baseline=False), "forehand")


class FoldServeTossesTest(unittest.TestCase):
    def test_the_toss_before_a_serve_is_folded_into_it(self):
        hits = [
            {"frame": 900, "player": 6, "tracklet": 1, "stroke": "forehand", "at_baseline": True},   # toss
            {"frame": 925, "player": 6, "tracklet": 1, "stroke": "serve", "at_baseline": True},      # toss, as serve
            {"frame": 950, "player": 6, "tracklet": 1, "stroke": "serve", "at_baseline": True},      # the hit
            {"frame": 930, "player": 2, "tracklet": 2, "stroke": "forehand", "at_baseline": True},   # someone else
            {"frame": 800, "player": 6, "tracklet": 1, "stroke": "forehand", "at_baseline": True},   # 6s earlier
        ]
        kept = fold_serve_tosses(hits, 25.0)
        self.assertEqual(sorted(h["frame"] for h in kept), [800, 930, 950])


@unittest.skipUnless(WEIGHTS.exists(), "BST weights not downloaded")
class ModelTest(unittest.TestCase):
    def test_weights_load_and_give_probabilities(self):
        from src.analysis.bst_strokes import classify, load_model

        model = load_model(str(WEIGHTS))
        empty = [PersonFrame(None, None, None)] * 20
        probs = classify(model, [build_input(empty, empty, [None] * 20, (1920, 1080))])
        self.assertEqual(probs.shape, (1, 6))
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
