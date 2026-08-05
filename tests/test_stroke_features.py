import unittest

import numpy as np

from src.analysis.stroke_features import FEATURE_NAMES, extract_features
from src.detection.pose_detector import PersonPose


def make_window(scale=1.0, num_frames=7):
    """A standing figure with the right arm extending outward over time -
    the right wrist moves, everything else stays fixed, so the right side
    should be picked as the active (swinging) arm."""
    window = []
    for t in range(num_frames):
        keypoints = np.zeros((17, 2))
        keypoints[5] = (-20, 0)  # left_shoulder
        keypoints[6] = (0, 0)  # right_shoulder
        keypoints[7] = (-30, 0)  # left_elbow (stationary)
        keypoints[8] = (10, 0)  # right_elbow
        keypoints[9] = (-40, 0)  # left_wrist (stationary)
        keypoints[10] = (20 + 5 * t, 0)  # right_wrist, moving +5/frame
        keypoints[11] = (-15, 40)  # left_hip
        keypoints[12] = (15, 40)  # right_hip
        keypoints *= scale
        window.append(
            PersonPose(
                keypoints=keypoints,
                keypoint_confidence=np.ones(17),
                bbox_confidence=0.9,
                center_x=0.0,
                center_y=0.0,
            )
        )
    return window


class StrokeFeaturesTest(unittest.TestCase):
    def test_output_length_matches_feature_names(self):
        features = extract_features(make_window())
        self.assertEqual(len(features), len(FEATURE_NAMES))

    def test_hand_computed_values_for_known_pose(self):
        features = extract_features(make_window())
        by_name = dict(zip(FEATURE_NAMES, features))

        # right arm is fully extended and colinear at the center frame -> straight elbow
        self.assertAlmostEqual(by_name["elbow_angle"], np.pi, places=4)
        # shoulders level -> 0
        self.assertAlmostEqual(by_name["shoulder_angle"], 0.0, places=4)
        # wrist at same height as shoulder -> 0
        self.assertAlmostEqual(by_name["wrist_height_rel_shoulder"], 0.0, places=4)
        # wrist moves +5px/frame in x, shoulder width is 20px -> 0.25/frame normalized
        self.assertAlmostEqual(by_name["wrist_dx_per_frame"], 0.25, places=4)
        self.assertAlmostEqual(by_name["wrist_dy_per_frame"], 0.0, places=4)

    def test_scale_invariance(self):
        # the same stroke performed twice as close to the camera should
        # produce an (almost) identical feature vector, not a scaled one
        original = extract_features(make_window(scale=1.0))
        scaled = extract_features(make_window(scale=2.0))

        np.testing.assert_allclose(original, scaled, atol=1e-4)

    def test_requires_at_least_two_frames(self):
        with self.assertRaises(ValueError):
            extract_features(make_window(num_frames=1))


if __name__ == "__main__":
    unittest.main()
