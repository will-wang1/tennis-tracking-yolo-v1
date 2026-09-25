import unittest

import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.analysis.person_tracks import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    coverage_grid,
    foot_positions_by_track,
    is_on_this_court,
    metres_per_pixel_at,
    smooth_positions,
    summarise_movement,
)

FPS = 25.0


def _perspective_calibration():
    """A behind-the-baseline view: far baseline narrow and high in frame,
    near baseline wide and low - so metres-per-pixel really does differ
    between the two ends, as on the real fence camera."""
    corners = {
        "baseline_far_left": (700.0, 200.0),
        "baseline_far_right": (1220.0, 200.0),
        "baseline_near_right": (1900.0, 850.0),
        "baseline_near_left": (20.0, 850.0),
    }
    return CourtCalibration.from_keypoints(
        corners, world_points={k: FULL_COURT_REFERENCE_POINTS[k] for k in corners}
    )


class OnThisCourtTest(unittest.TestCase):
    def test_inside_and_just_behind_the_baseline_count(self):
        self.assertTrue(is_on_this_court((5.0, 12.0)))
        self.assertTrue(is_on_this_court((5.0, -3.0)))  # queueing behind far baseline

    def test_the_next_court_over_does_not(self):
        self.assertFalse(is_on_this_court((-7.6, 10.0)))
        self.assertFalse(is_on_this_court((COURT_WIDTH_M + 7.0, 10.0)))


class SmoothingTest(unittest.TestCase):
    def test_jitter_on_a_standing_person_does_not_become_distance(self):
        rng = np.random.default_rng(0)
        standing = {f: (5.0 + rng.normal(0, 0.15), 20.0 + rng.normal(0, 0.15)) for f in range(250)}
        calibration = _perspective_calibration()

        raw = summarise_movement({1: standing}, FPS, calibration, smoothing_frames=1)[0]
        smoothed = summarise_movement({1: standing}, FPS, calibration)[0]

        self.assertGreater(raw.distance_m, 20.0)  # 10s of pure jitter reads as metres
        self.assertLess(smoothed.distance_m, raw.distance_m / 3)

    def test_real_movement_survives_smoothing(self):
        walking = {f: (1.0 + 0.04 * f, 20.0) for f in range(250)}  # 1 m/s for 10s
        m = summarise_movement({1: walking}, FPS, _perspective_calibration())[0]
        self.assertAlmostEqual(m.distance_m, 9.96, delta=0.3)

    def test_one_wild_box_does_not_drag_the_path(self):
        path = {f: (5.0, 20.0) for f in range(30)}
        path[15] = (9.0, 2.0)
        self.assertEqual(smooth_positions(path)[15], (5.0, 20.0))


class ConfidenceTest(unittest.TestCase):
    def test_far_end_is_estimated_near_end_is_measured(self):
        calibration = _perspective_calibration()
        far = {f: (5.0, 0.0) for f in range(50)}
        near = {f: (5.0, 23.0) for f in range(50)}
        by_id = {m.track_id: m for m in summarise_movement({1: far, 2: near}, FPS, calibration)}

        self.assertGreater(metres_per_pixel_at(calibration, (5.0, 0.0)), metres_per_pixel_at(calibration, (5.0, 23.0)))
        self.assertEqual(by_id[1].distance_confidence, "estimated")
        self.assertEqual(by_id[2].distance_confidence, "measured")


class FootPositionsTest(unittest.TestCase):
    def test_uses_the_foot_point_and_skips_cutaway_frames(self):
        calibration = _perspective_calibration()
        people = [
            [(7, 900.0, 600.0, 1000.0, 850.0, 0.9)],
            [(7, 900.0, 600.0, 1000.0, 850.0, 0.9)],
            [(None, 0.0, 0.0, 10.0, 10.0, 0.4)],
        ]
        tracks = foot_positions_by_track(people, lambda f: calibration, skip_frames={1})

        self.assertEqual(set(tracks), {7})
        self.assertEqual(set(tracks[7]), {0})  # frame 1 skipped, untracked box ignored
        self.assertAlmostEqual(tracks[7][0][1], COURT_LENGTH_M, delta=0.05)  # feet on near baseline


class CoverageTest(unittest.TestCase):
    def test_time_lands_in_the_right_cell_and_off_court_is_ignored(self):
        tracks = {1: {f: (5.5, 12.5) for f in range(25)}, 2: {f: (-20.0, 12.0) for f in range(25)}}
        grid = coverage_grid(tracks, FPS)
        cells = np.array(grid.seconds)

        self.assertAlmostEqual(cells.sum(), 1.0, places=6)  # one second, and only person 1's
        r = int((12.5 - grid.origin_m[1]) // grid.cell_m)
        c = int((5.5 - grid.origin_m[0]) // grid.cell_m)
        self.assertAlmostEqual(cells[r, c], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()


class BoxesByFrameTest(unittest.TestCase):
    def test_every_box_is_kept_tracked_or_not_and_empty_frames_are_skipped(self):
        from src.analysis.person_tracks import boxes_by_frame

        people = [[(3, 1.0, 2.0, 3.0, 4.0, 0.9), (None, 5.0, 6.0, 7.0, 8.0, 0.3)], [], [(3, 1.0, 2.0, 3.0, 4.0, 0.9)]]
        self.assertEqual(boxes_by_frame(people), {0: [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)], 2: [(1.0, 2.0, 3.0, 4.0)]})


class MovingShareTest(unittest.TestCase):
    def test_standing_still_is_not_moving_and_jogging_is(self):
        from src.analysis.person_tracks import moving_share

        standing = {f: (5.0, 20.0) for f in range(100)}
        jogging = {f: (1.0 + 0.08 * f, 20.0) for f in range(100)}  # 2 m/s
        self.assertEqual(moving_share(standing, FPS), 0.0)
        self.assertEqual(moving_share(jogging, FPS), 1.0)

    def test_half_and_half(self):
        from src.analysis.person_tracks import moving_share

        path = {f: (1.0 + 0.08 * f, 20.0) if f < 100 else (9.0, 20.0) for f in range(200)}
        self.assertAlmostEqual(moving_share(path, FPS), 0.5, delta=0.1)

    def test_a_gap_in_tracking_is_not_bridged(self):
        from src.analysis.person_tracks import moving_share

        # Two stationary stretches 5m apart with nothing in between: a window
        # across the gap would read as movement that was never seen.
        path = {**{f: (1.0, 20.0) for f in range(50)}, **{f: (6.0, 20.0) for f in range(200, 250)}}
        self.assertEqual(moving_share(path, FPS), 0.0)

    def test_too_short_to_say(self):
        from src.analysis.person_tracks import moving_share

        self.assertIsNone(moving_share({f: (1.0, 1.0) for f in range(20)}, FPS))


class AttributeContactsTest(unittest.TestCase):
    def test_credits_the_person_in_reach_and_nobody_otherwise(self):
        from src.analysis.person_tracks import attribute_contacts

        people = [
            [(1, 100.0, 100.0, 140.0, 200.0, 0.9), (2, 800.0, 100.0, 840.0, 200.0, 0.9)],
            [(1, 100.0, 100.0, 140.0, 200.0, 0.9)],
        ]
        # frame 0: ball beside person 2; frame 1: ball far from everyone
        credited, unattributed = attribute_contacts([(0, 850.0, 150.0), (1, 500.0, 150.0)], people)

        self.assertEqual(credited, {2: 1})
        self.assertEqual(unattributed, 1)

    def test_untracked_boxes_are_never_credited(self):
        from src.analysis.person_tracks import attribute_contacts

        people = [[(None, 100.0, 100.0, 140.0, 200.0, 0.4)]]
        self.assertEqual(attribute_contacts([(0, 120.0, 150.0)], people), ({}, 1))
