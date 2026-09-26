import unittest

import numpy as np

from src.analysis.identity import box_iou, cluster_tracklets, conflicts, pairwise_scores

RED, BLUE = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0])


def _boxes(frames, x):
    return {f: (x, 100.0, x + 40.0, 200.0, 0.9) for f in frames}


class ConflictTest(unittest.TestCase):
    def test_two_people_on_screen_together_in_different_places_conflict(self):
        boxes = {1: _boxes(range(100), 100.0), 2: _boxes(range(100), 600.0)}
        self.assertEqual(conflicts(boxes), {frozenset((1, 2))})

    def test_a_short_tracker_handover_is_not_a_conflict(self):
        boxes = {1: _boxes(range(0, 50), 100.0), 2: _boxes(range(45, 100), 600.0)}  # 5 shared frames
        self.assertEqual(conflicts(boxes), set())

    def test_one_person_boxed_twice_is_not_a_conflict(self):
        # the coach carried two ids at once for 2.8s on Dingles
        boxes = {1: _boxes(range(100), 100.0), 2: _boxes(range(100), 104.0)}
        self.assertEqual(conflicts(boxes), set())

    def test_iou(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


class ClusterTest(unittest.TestCase):
    def test_same_clothes_at_different_times_merge_different_clothes_do_not(self):
        fp = {1: RED, 2: RED * 0.9 + BLUE * 0.05, 3: BLUE}
        boxes = {1: _boxes(range(0, 50), 100.0), 2: _boxes(range(60, 100), 300.0), 3: _boxes(range(0, 100), 600.0)}
        a = cluster_tracklets(fp, boxes, merge_distance=0.3)
        self.assertEqual(a[1], a[2])
        self.assertNotEqual(a[1], a[3])

    def test_identical_clothes_on_screen_together_are_kept_apart(self):
        # the rule that protects same-kit players: co-occurrence forbids the merge
        fp = {1: RED, 2: RED}
        boxes = {1: _boxes(range(100), 100.0), 2: _boxes(range(100), 600.0)}
        a = cluster_tracklets(fp, boxes, merge_distance=0.9)
        self.assertNotEqual(a[1], a[2])

    def test_scores(self):
        s = pairwise_scores({1: 0, 2: 0, 3: 1}, {1: "a", 2: "a", 3: "b"})
        self.assertEqual((s["precision"], s["recall"], s["clusters"]), (1.0, 1.0, 2))


if __name__ == "__main__":
    unittest.main()
