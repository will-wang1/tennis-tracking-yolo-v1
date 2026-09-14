import unittest

from src.analysis.court_zones import classify_court_half, classify_landing_zone


class ClassifyLandingZoneTest(unittest.TestCase):
    def test_far_baseline_corner_is_far_deep(self):
        zone = classify_landing_zone(world_x=1.0, world_y=1.0)
        self.assertEqual(zone.half, "far")
        self.assertEqual(zone.depth, "deep")

    def test_near_baseline_corner_is_near_deep(self):
        zone = classify_landing_zone(world_x=1.0, world_y=23.0)
        self.assertEqual(zone.half, "near")
        self.assertEqual(zone.depth, "deep")

    def test_just_past_the_net_on_the_far_side_is_far_short(self):
        zone = classify_landing_zone(world_x=5.485, world_y=10.0)
        self.assertEqual(zone.half, "far")
        self.assertEqual(zone.depth, "short")

    def test_just_past_the_net_on_the_near_side_is_near_short(self):
        zone = classify_landing_zone(world_x=5.485, world_y=13.77)
        self.assertEqual(zone.half, "near")
        self.assertEqual(zone.depth, "short")

    def test_deuce_ad_split_flips_between_halves(self):
        # left of center (low x): deuce on the far side, ad on the near side
        far_left = classify_landing_zone(world_x=2.0, world_y=2.0)
        near_left = classify_landing_zone(world_x=2.0, world_y=22.0)
        self.assertEqual(far_left.side, "deuce")
        self.assertEqual(near_left.side, "ad")

        # right of center (high x): ad on the far side, deuce on the near side
        far_right = classify_landing_zone(world_x=9.0, world_y=2.0)
        near_right = classify_landing_zone(world_x=9.0, world_y=22.0)
        self.assertEqual(far_right.side, "ad")
        self.assertEqual(near_right.side, "deuce")

    def test_inside_singles_lines_is_singles(self):
        zone = classify_landing_zone(world_x=5.485, world_y=2.0)
        self.assertEqual(zone.bounds, "singles")

    def test_between_singles_and_doubles_lines_is_doubles_alley(self):
        zone = classify_landing_zone(world_x=0.5, world_y=2.0)
        self.assertEqual(zone.bounds, "doubles_alley")

    def test_beyond_doubles_lines_is_out(self):
        zone = classify_landing_zone(world_x=-1.0, world_y=2.0)
        self.assertEqual(zone.bounds, "out")

    def test_label_combines_all_three_fields(self):
        zone = classify_landing_zone(world_x=2.0, world_y=2.0)
        self.assertEqual(zone.label(), "far_deuce_deep")


class ClassifyCourtHalfTest(unittest.TestCase):
    def test_agrees_with_landing_zone_half(self):
        for world_y in (-6.5, 1.0, 10.0, 13.77, 23.0, 30.0):
            self.assertEqual(
                classify_court_half(world_y), classify_landing_zone(world_x=5.0, world_y=world_y).half
            )

    def test_negative_y_beyond_the_far_baseline_is_still_far(self):
        # An elevated point (a racket contact, not a bounce) projects through
        # the ground-plane homography as pushed away from the camera - see
        # parabolic_bounce_detector's module docstring - which can land past
        # y=0, behind the far baseline. Still unambiguously the far half.
        self.assertEqual(classify_court_half(-7.0), "far")


if __name__ == "__main__":
    unittest.main()
