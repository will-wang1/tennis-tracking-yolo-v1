import unittest

import numpy as np

from src.analysis.player_identity import PLAYER_A, PLAYER_B, PlayerIdentityTracker

NEAR_BOX = (10, 10, 40, 60)
FAR_BOX = (60, 10, 90, 60)


def _frame(near_color, far_color, size=100):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    x1, y1, x2, y2 = NEAR_BOX
    frame[y1:y2, x1:x2] = near_color
    x1, y1, x2, y2 = FAR_BOX
    frame[y1:y2, x1:x2] = far_color
    return frame


BLUE = (200, 40, 20)   # BGR - stands in for one player's kit
GREEN = (20, 180, 40)  # BGR - stands in for the other's


class PlayerIdentityTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = PlayerIdentityTracker()

    def test_first_double_sighting_bootstraps_the_labels(self):
        result = self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        self.assertEqual(result, {"near": PLAYER_A, "far": PLAYER_B})

    def test_identity_carries_forward_without_reassignment(self):
        self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        # continuity trusted by default, even though nothing here re-checks colour
        result = self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        self.assertEqual(result, {"near": PLAYER_A, "far": PLAYER_B})

    def test_a_missing_box_produces_no_entry_for_that_role(self):
        self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        result = self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, None)
        self.assertEqual(result, {"near": PLAYER_A})
        self.assertNotIn("far", result)

    def test_without_force_reassign_a_positional_swap_is_not_corrected(self):
        self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        # the two players have physically swapped ends, but nothing told
        # the tracker to re-check - it keeps trusting position
        result = self.tracker.update(_frame(GREEN, BLUE), NEAR_BOX, FAR_BOX)
        self.assertEqual(result, {"near": PLAYER_A, "far": PLAYER_B})

    def test_force_reassign_corrects_a_positional_swap_by_appearance(self):
        self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        # a "cut" (or an end-change) swaps who's on which side; colour says
        # what actually happened even though position doesn't
        result = self.tracker.update(_frame(GREEN, BLUE), NEAR_BOX, FAR_BOX, force_reassign=True)
        self.assertEqual(result, {"near": PLAYER_B, "far": PLAYER_A})

    def test_force_reassign_keeps_the_mapping_when_nothing_actually_swapped(self):
        self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        # a cut to a DIFFERENT angle of the same, unchanged arrangement -
        # force_reassign should reach the same conclusion, not flip for no reason
        result = self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX, force_reassign=True)
        self.assertEqual(result, {"near": PLAYER_A, "far": PLAYER_B})

    def test_force_reassign_on_the_bootstrap_frame_still_just_bootstraps(self):
        # nothing to match against yet - force_reassign can't do anything
        # useful here and must not raise
        result = self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX, force_reassign=True)
        self.assertEqual(result, {"near": PLAYER_A, "far": PLAYER_B})

    def test_no_boxes_before_bootstrap_produces_an_empty_result(self):
        result = self.tracker.update(_frame(BLUE, GREEN), None, None)
        self.assertEqual(result, {})

    def test_repeated_reassignment_stays_stable_across_several_cuts(self):
        self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX)
        for _ in range(5):
            result = self.tracker.update(_frame(BLUE, GREEN), NEAR_BOX, FAR_BOX, force_reassign=True)
            self.assertEqual(result, {"near": PLAYER_A, "far": PLAYER_B})


if __name__ == "__main__":
    unittest.main()
