import unittest

import numpy as np

from src.analysis.striker import estimate_ground_y, select_players, select_striker
from src.detection.pose_detector import PersonPose
from src.tracking.ball_tracker import TrackedPosition


def make_pose(center_x, center_y, keypoints=None, keypoint_confidence=None):
    return PersonPose(
        keypoints=np.zeros((17, 2)) if keypoints is None else keypoints,
        keypoint_confidence=np.zeros(17) if keypoint_confidence is None else keypoint_confidence,
        bbox_confidence=0.9,
        center_x=center_x,
        center_y=center_y,
    )


class StrikerTest(unittest.TestCase):
    def test_nearer_pose_wins(self):
        near = make_pose(100, 100)
        far = make_pose(500, 500)
        ball = TrackedPosition(frame_idx=0, x=110, y=110, interpolated=False)

        self.assertIs(select_striker([far, near], ball), near)

    def test_none_ball_returns_none(self):
        pose = make_pose(100, 100)
        self.assertIsNone(select_striker([pose], None))

    def test_no_poses_returns_none(self):
        ball = TrackedPosition(frame_idx=0, x=0, y=0, interpolated=False)
        self.assertIsNone(select_striker([], ball))

    def test_exact_tie_first_in_list_wins(self):
        first = make_pose(0, 0)
        second = make_pose(0, 0)
        ball = TrackedPosition(frame_idx=0, x=0, y=0, interpolated=False)

        self.assertIs(select_striker([first, second], ball), first)


class SelectPlayersTest(unittest.TestCase):
    def test_keeps_two_nearest_frame_center(self):
        near_player = make_pose(900, 400)
        far_player = make_pose(1000, 200)
        umpire = make_pose(1550, 300)
        ball_kid = make_pose(100, 500)

        result = select_players([umpire, near_player, ball_kid, far_player], frame_width=1920)

        self.assertEqual(len(result), 2)
        self.assertTrue(any(p is near_player for p in result))
        self.assertTrue(any(p is far_player for p in result))

    def test_fewer_poses_than_count_returns_all(self):
        pose = make_pose(950, 400)
        result = select_players([pose], frame_width=1920)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], pose)

    def test_no_poses_returns_empty(self):
        self.assertEqual(select_players([], frame_width=1920), [])

    def test_respects_custom_count(self):
        poses = [make_pose(x, 400) for x in (960, 900, 1020, 500, 1400)]
        result = select_players(poses, frame_width=1920, count=3)
        self.assertEqual(len(result), 3)


class EstimateGroundYTest(unittest.TestCase):
    def test_uses_max_of_confident_ankles(self):
        keypoints = np.zeros((17, 2))
        keypoints[15] = (100, 800)  # left ankle
        keypoints[16] = (110, 820)  # right ankle, slightly lower on screen
        confidence = np.zeros(17)
        confidence[15] = 0.9
        confidence[16] = 0.9
        pose = make_pose(100, 400, keypoints=keypoints, keypoint_confidence=confidence)
        ball = TrackedPosition(frame_idx=0, x=105, y=790, interpolated=False)

        self.assertAlmostEqual(estimate_ground_y([pose], ball), 820.0)

    def test_falls_back_to_lowest_confident_keypoint_when_ankles_unreliable(self):
        keypoints = np.zeros((17, 2))
        keypoints[15] = (100, 800)  # left ankle - present but low confidence
        keypoints[13] = (100, 700)  # left knee - confidently detected, next lowest
        confidence = np.zeros(17)
        confidence[15] = 0.1  # below MIN_KEYPOINT_CONFIDENCE
        confidence[13] = 0.9
        pose = make_pose(100, 400, keypoints=keypoints, keypoint_confidence=confidence)
        ball = TrackedPosition(frame_idx=0, x=105, y=690, interpolated=False)

        self.assertAlmostEqual(estimate_ground_y([pose], ball), 700.0)

    def test_none_when_no_ball_or_poses(self):
        pose = make_pose(100, 400)
        ball = TrackedPosition(frame_idx=0, x=0, y=0, interpolated=False)

        self.assertIsNone(estimate_ground_y([], ball))
        self.assertIsNone(estimate_ground_y([pose], None))

    def test_none_when_no_keypoints_confident_enough(self):
        pose = make_pose(100, 400)  # all-zero confidence from the default helper
        ball = TrackedPosition(frame_idx=0, x=100, y=400, interpolated=False)

        self.assertIsNone(estimate_ground_y([pose], ball))


if __name__ == "__main__":
    unittest.main()
