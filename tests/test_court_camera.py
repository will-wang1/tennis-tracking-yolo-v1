import unittest

import numpy as np

from src.analysis.court_camera import (
    GRAVITY,
    CameraPose,
    fit_height_curve,
    initial_pose_from_homography,
    pose_from_upright_objects,
    upright_observations_from_boxes,
)
from src.analysis.court_calibration import CourtCalibration

FPS = 60.0
POSE = CameraPose(nadir=(5.27, 63.11), height=13.04)  # the pose measured on video_input2


def _project(point: tuple[float, float], height: float, pose: CameraPose = POSE) -> tuple[float, float]:
    """Where the camera ray through a ball at (`point`, `height`) meets the
    court - i.e. what a ground-plane homography reports for it."""
    nadir = np.asarray(pose.nadir)
    projected = nadir + (np.asarray(point) - nadir) * pose.height / (pose.height - height)
    return float(projected[0]), float(projected[1])


def _flight(
    ground_start=(4.0, 18.0), velocity=(1.0, -12.0), height=1.0, rise=5.0, frames=48, pose=POSE
):
    """One free flight, as (times, ground projections, true heights)."""
    times, projections, heights = [], [], []
    for i in range(frames):
        t = i / FPS
        h = height + rise * t - 0.5 * GRAVITY * t * t
        point = (ground_start[0] + velocity[0] * t, ground_start[1] + velocity[1] * t)
        times.append(t)
        projections.append(_project(point, h, pose))
        heights.append(h)
    return times, projections, heights


class FitHeightCurveTest(unittest.TestCase):
    def test_recovers_the_true_height_of_a_clean_flight(self):
        times, projections, heights = _flight()

        curve = fit_height_curve(times, projections, POSE)

        self.assertIsNotNone(curve)
        for t, expected in zip(times, heights):
            self.assertAlmostEqual(curve.height(t), expected, places=3)

    def test_recovers_the_true_vertical_velocity(self):
        times, projections, _ = _flight(height=1.0, rise=5.0)

        curve = fit_height_curve(times, projections, POSE)

        # dh/dt = rise - g*t, in metres per second
        for t in (0.1, 0.3, 0.5):
            self.assertAlmostEqual(curve.vertical_velocity(t), 5.0 - GRAVITY * t, places=2)

    def test_a_clean_flight_fits_with_almost_no_residual(self):
        times, projections, _ = _flight()

        self.assertLess(fit_height_curve(times, projections, POSE).residual, 1e-6)

    def test_needs_more_samples_than_unknowns(self):
        times, projections, _ = _flight(frames=3)

        self.assertIsNone(fit_height_curve(times, projections, POSE))

    def test_rejects_a_camera_at_or_below_court_level(self):
        times, projections, _ = _flight()

        self.assertIsNone(
            fit_height_curve(times, projections, CameraPose(nadir=(5.0, 60.0), height=0.0))
        )

    def test_a_short_window_cannot_pin_down_height(self):
        # gravity is the only thing fixing the height scale and its effect
        # grows with the square of the window, so a fifth of a second of
        # noisy samples leaves the ball's height essentially unconstrained.
        # This is the measured reason the module isn't wired into the
        # pipeline - see its docstring.
        rng = np.random.default_rng(0)
        times, projections, heights = _flight(frames=12)
        noisy = [(x + rng.normal(0, 0.05), y + rng.normal(0, 0.05)) for x, y in projections]

        curve = fit_height_curve(times, noisy, POSE)

        middle = len(times) // 2
        self.assertGreater(abs(curve.height(times[middle]) - heights[middle]), 2.0)

    def test_a_long_window_survives_the_same_noise(self):
        # four times the window against identical noise: the error falls
        # from metres to well under one. Still not the ~0.2m a bounce needs,
        # which is why 0.75s is the floor rather than a comfortable margin.
        rng = np.random.default_rng(0)
        times, projections, heights = _flight(frames=48)  # 0.8s
        noisy = [(x + rng.normal(0, 0.05), y + rng.normal(0, 0.05)) for x, y in projections]

        curve = fit_height_curve(times, noisy, POSE)

        middle = len(times) // 2
        self.assertLess(abs(curve.height(times[middle]) - heights[middle]), 1.0)


class PoseFromUprightObjectsTest(unittest.TestCase):
    def _sightings(self, count=200, person_height=1.78, noise=0.0, seed=1):
        rng = np.random.default_rng(seed)
        sightings = []
        for _ in range(count):
            point = (rng.uniform(0.0, 10.97), rng.uniform(-1.0, 25.0))
            base = np.array(_project(point, 0.0))
            top = np.array(_project(point, person_height))
            if noise:
                base = base + rng.normal(0, noise, 2)
                top = top + rng.normal(0, noise, 2)
            sightings.append((tuple(base), tuple(top)))
        return sightings

    def test_recovers_the_camera_from_people_of_known_height(self):
        pose = pose_from_upright_objects(self._sightings(), object_height=1.78)

        self.assertAlmostEqual(pose.height, POSE.height, delta=0.1)
        self.assertAlmostEqual(pose.nadir[0], POSE.nadir[0], delta=0.1)
        self.assertAlmostEqual(pose.nadir[1], POSE.nadir[1], delta=0.5)

    def test_tolerates_realistic_measurement_noise(self):
        pose = pose_from_upright_objects(self._sightings(noise=0.05), object_height=1.78)

        self.assertAlmostEqual(pose.height, POSE.height, delta=1.5)

    def test_returns_none_without_enough_sightings(self):
        self.assertIsNone(pose_from_upright_objects(self._sightings(count=4)))

    def test_returns_none_when_every_sighting_points_the_same_way(self):
        # people all at the same court position give parallel lines, which
        # never intersect at a nadir
        base, top = _project((5.0, 10.0), 0.0), _project((5.0, 10.0), 1.78)

        self.assertIsNone(pose_from_upright_objects([(base, top)] * 20))

    def test_ignores_a_sighting_whose_top_is_not_elevated(self):
        # a box with zero height projects base and top to the same place and
        # carries no direction at all
        flat = [(_project((3.0, 8.0), 0.0), _project((3.0, 8.0), 0.0))] * 20

        self.assertIsNone(pose_from_upright_objects(flat))


class UprightObservationsFromBoxesTest(unittest.TestCase):
    def test_uses_the_bottom_and_top_of_each_box(self):
        calibration = CourtCalibration(homography=np.eye(3, dtype=np.float64))
        boxes = {7: [(100.0, 200.0, 140.0, 400.0)]}

        observations = upright_observations_from_boxes(boxes, {7: calibration})

        # identity calibration, so world == pixels: feet at the box bottom
        # centre, head at the top centre
        self.assertEqual(observations, [((120.0, 400.0), (120.0, 200.0))])

    def test_skips_frames_without_a_calibration(self):
        boxes = {7: [(100.0, 200.0, 140.0, 400.0)]}

        self.assertEqual(upright_observations_from_boxes(boxes, {}), [])


class InitialPoseFromHomographyTest(unittest.TestCase):
    def test_recovers_a_synthetic_camera_from_its_own_homography(self):
        # build a real camera, project the court corners through it, and fit
        # the homography those correspondences imply
        focal = 2000.0
        intrinsics = np.array([[focal, 0, 960.0], [0, focal, 540.0], [0, 0, 1.0]])
        pitch = np.deg2rad(20.0)
        rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        center = np.array([5.5, 40.0, 12.0])
        translation = -rotation @ center
        ground_to_image = intrinsics @ np.column_stack(
            [rotation[:, 0], rotation[:, 1], translation]
        )

        pose = initial_pose_from_homography(np.linalg.inv(ground_to_image), 1920, 1080)

        self.assertAlmostEqual(pose.height, 12.0, delta=0.5)
        self.assertAlmostEqual(pose.nadir[0], 5.5, delta=0.5)
        self.assertAlmostEqual(pose.nadir[1], 40.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()
