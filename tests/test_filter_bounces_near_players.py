import unittest

from src.analysis.bounce_detector import BounceEvent
from src.analysis.catboost_bounce_detector import filter_bounces_near_players


class FilterBouncesNearPlayersTest(unittest.TestCase):
    def test_drops_bounce_inside_a_player_box(self):
        bounces = [BounceEvent(frame_idx=5, x=100.0, y=200.0)]
        player_boxes_by_frame = {5: [(80.0, 150.0, 120.0, 250.0)]}

        result = filter_bounces_near_players(bounces, player_boxes_by_frame)

        self.assertEqual(result, [])

    def test_drops_bounce_within_the_reach_margin_of_a_player_box(self):
        bounces = [BounceEvent(frame_idx=5, x=135.0, y=200.0)]
        player_boxes_by_frame = {5: [(80.0, 150.0, 120.0, 250.0)]}  # 15px outside the box on x

        result = filter_bounces_near_players(bounces, player_boxes_by_frame, reach_margin=20.0)

        self.assertEqual(result, [])

    def test_keeps_bounce_far_from_any_player(self):
        bounces = [BounceEvent(frame_idx=5, x=500.0, y=500.0)]
        player_boxes_by_frame = {5: [(80.0, 150.0, 120.0, 250.0)]}

        result = filter_bounces_near_players(bounces, player_boxes_by_frame, reach_margin=20.0)

        self.assertEqual(result, bounces)

    def test_keeps_bounce_on_a_frame_with_no_player_boxes_recorded(self):
        bounces = [BounceEvent(frame_idx=9, x=500.0, y=500.0)]

        result = filter_bounces_near_players(bounces, player_boxes_by_frame={})

        self.assertEqual(result, bounces)


if __name__ == "__main__":
    unittest.main()
