import unittest

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.analysis.speed_estimator import (
    estimate_net_crossing_speeds,
    estimate_shot_speeds,
    instantaneous_speeds,
    net_pixel_y_range,
    segment_shots,
)
from src.tracking.ball_tracker import TrackedPosition


def pos(frame_idx, x, y, interpolated=False):
    return TrackedPosition(frame_idx=frame_idx, x=x, y=y, interpolated=interpolated)


def scaled_calibration(scale=100.0):
    # pixel = world * scale for the 4 doubles-baseline corners - a pure
    # uniform scale with no perspective distortion, so reprojected pixel
    # positions are exactly predictable in tests.
    names = ["baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right"]
    pixel_points = {
        name: (wx * scale, wy * scale)
        for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items()
        if name in names
    }
    return CourtCalibration.from_keypoints(pixel_points)


class SpeedEstimatorTest(unittest.TestCase):
    def test_instantaneous_speed_px_per_second_no_calibration(self):
        # 10 px/frame at 30fps -> 300 px/s, exactly
        positions = [pos(0, 0, 0), pos(1, 10, 0), pos(2, 20, 0)]
        speeds = instantaneous_speeds(positions, fps=30.0)

        self.assertAlmostEqual(speeds[1], 300.0)
        self.assertAlmostEqual(speeds[2], 300.0)
        self.assertNotIn(0, speeds)  # no speed for the first frame - nothing to diff against

    def test_instantaneous_speed_accounts_for_frame_gap(self):
        # same total displacement but spread over 2 frames -> half the speed
        positions = [pos(0, 0, 0), pos(2, 20, 0)]
        speeds = instantaneous_speeds(positions, fps=30.0)

        self.assertAlmostEqual(speeds[2], 300.0)

    def test_wider_window_smooths_single_frame_jitter(self):
        # a real, steady 10px/frame drift with one single-frame jitter spike
        # (frame 2 briefly jumps far off-trend) - window=1 is fooled by the
        # spike, a wider window averages it away since it's a single outlier
        # amid otherwise-consistent motion
        positions = [
            pos(0, 0, 0), pos(1, 10, 0), pos(2, 80, 0), pos(3, 30, 0),
            pos(4, 40, 0), pos(5, 50, 0),
        ]
        narrow = instantaneous_speeds(positions, fps=30.0, window=1)
        wide = instantaneous_speeds(positions, fps=30.0, window=5)

        self.assertGreater(narrow[2], 1000)  # the 70px single-frame spike dominates
        self.assertAlmostEqual(wide[5], 300.0)  # 50px over 5 frames at 30fps = steady 10px/frame trend

    def test_window_larger_than_history_yields_no_speeds_for_early_frames(self):
        positions = [pos(0, 0, 0), pos(1, 10, 0)]
        speeds = instantaneous_speeds(positions, fps=30.0, window=5)

        self.assertEqual(speeds, {})

    def test_instantaneous_speed_km_per_hour_with_calibration(self):
        pixel_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        world_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        calibration = CourtCalibration.from_points(pixel_points, world_points)

        # 100px displacement in 1 frame @ 10fps = 10m in 0.1s = 100 m/s = 360 km/h
        positions = [pos(0, 0, 50), pos(1, 100, 50)]
        speeds = instantaneous_speeds(positions, fps=10.0, calibration=calibration)

        self.assertAlmostEqual(speeds[1], 360.0, places=1)

    def test_segment_shots_splits_at_breakpoint_frames(self):
        positions = [pos(i, i, 0) for i in range(11)]  # frames 0..10

        segments = segment_shots(positions, [5])

        self.assertEqual(segments, [(0, 5), (5, 10)])

    def test_segment_shots_with_no_breakpoints_is_one_segment(self):
        positions = [pos(i, i, 0) for i in range(5)]
        self.assertEqual(segment_shots(positions, []), [(0, 4)])

    def test_segment_shots_empty_positions(self):
        self.assertEqual(segment_shots([], []), [])

    def test_estimate_shot_speeds_reports_peak_not_average_or_last(self):
        # speed rises then falls within one segment (no breakpoints): 10, 50, 10 px/frame
        positions = [
            pos(0, 0, 0),
            pos(1, 10, 0),
            pos(2, 60, 0),
            pos(3, 70, 0),
        ]
        shots = estimate_shot_speeds(positions, breakpoint_frames=[], fps=30.0)

        self.assertEqual(len(shots), 1)
        shot = shots[0]
        self.assertEqual(shot.peak_frame, 2)  # the 50px/frame jump
        self.assertAlmostEqual(shot.peak_speed, 50.0 * 30.0)
        self.assertEqual(shot.unit, "px/s")

    def test_estimate_shot_speeds_unit_is_kmh_with_calibration(self):
        pixel_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        world_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        calibration = CourtCalibration.from_points(pixel_points, world_points)

        positions = [pos(0, 0, 50), pos(1, 10, 50), pos(2, 20, 50)]
        shots = estimate_shot_speeds(positions, breakpoint_frames=[], fps=30.0, calibration=calibration)

        self.assertEqual(shots[0].unit, "km/h")

    def test_implausible_kmh_reading_is_excluded_from_peak_selection(self):
        # a coarse calibration (large real-world distance per pixel) turns
        # an individually-small, tracker-accepted pixel jump into a
        # physically impossible speed - that frame must not win "peak"
        pixel_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        world_points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]  # 10 m/px
        calibration = CourtCalibration.from_points(pixel_points, world_points)

        positions = [
            pos(0, 0, 5),
            pos(1, 1, 5),  # 1px -> 10m in 1/30s = 300 m/s = 1080 km/h - implausible, must be excluded
            pos(2, 1.05, 5),  # tiny, plausible jump
        ]
        shots = estimate_shot_speeds(positions, breakpoint_frames=[], fps=30.0, calibration=calibration)

        self.assertEqual(len(shots), 1)
        self.assertNotEqual(shots[0].peak_frame, 1)
        self.assertLess(shots[0].peak_speed, 300.0)

    def test_px_per_second_readings_are_never_excluded_as_implausible(self):
        # px/s has no physical ceiling to check against - the cap only
        # applies when a calibration converts to a real-world unit
        positions = [pos(0, 0, 0), pos(1, 100000, 0)]
        shots = estimate_shot_speeds(positions, breakpoint_frames=[], fps=30.0)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].unit, "px/s")


    def test_net_pixel_y_range_matches_known_scale(self):
        calibration = scaled_calibration(scale=100.0)
        # net world-y is the court length's midpoint (23.77 / 2 = 11.885);
        # a pure uniform scale means both sideline edges reproject to the
        # same pixel y, with no perspective spread
        min_y, max_y = net_pixel_y_range(calibration)

        self.assertAlmostEqual(min_y, 1188.5, places=1)
        self.assertAlmostEqual(max_y, 1188.5, places=1)

    def test_net_crossing_speed_ignores_static_lockon_near_net(self):
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5

        positions = [
            # a real, moving ball crossing near the net
            pos(0, 500.0, net_y - 10),
            pos(1, 550.0, net_y + 5),
            # a static false-positive lock-on sitting right at the net -
            # barely moves frame to frame, must NOT be read as a real shot
            pos(10, 700.0, net_y),
            pos(11, 701.0, net_y + 1),
            pos(12, 700.5, net_y - 1),
        ]
        calibrations = {p.frame_idx: calibration for p in positions}
        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        # the pair (frame 0 -> frame 1) is recorded at frame 1, matching
        # instantaneous_speeds' convention of keying by the later frame
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].start_frame, 1)
        self.assertEqual(shots[0].end_frame, 1)
        self.assertEqual(shots[0].unit, "km/h")

    def test_net_crossing_speed_ignores_positions_away_from_net(self):
        calibration = scaled_calibration(scale=100.0)
        # far from the net line (net_y ~= 1188.5), even with real motion
        positions = [pos(0, 0.0, 50.0), pos(1, 100.0, 50.0)]
        calibrations = {p.frame_idx: calibration for p in positions}

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(shots, [])

    def test_net_crossing_speed_ignores_interpolated_positions(self):
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [
            pos(0, 500.0, net_y, interpolated=True),
            pos(1, 550.0, net_y, interpolated=True),
        ]
        calibrations = {p.frame_idx: calibration for p in positions}

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(shots, [])

    def test_net_crossing_speed_reports_peak_within_group(self):
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [
            pos(0, 0.0, net_y),
            pos(1, 10.0, net_y),  # small step
            pos(2, 60.0, net_y),  # big step - the peak
            pos(3, 70.0, net_y),  # small step
        ]
        calibrations = {p.frame_idx: calibration for p in positions}

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].peak_frame, 2)

    def test_net_crossing_speed_ignores_frames_missing_a_calibration(self):
        # the court detector lost the court on this frame (e.g. motion
        # blur) - that frame's pair must be excluded, not guessed at
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [pos(0, 0.0, net_y), pos(1, 50.0, net_y)]

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame={0: calibration})

        self.assertEqual(shots, [])

    def test_net_crossing_speed_uses_each_frames_own_calibration(self):
        # a panning camera: two calibrations at different scales, each with
        # its OWN correctly-located net line in pixel space (a larger scale
        # reprojects the net line further down the frame too) - the same
        # pixel displacement must map to a different real-world distance
        # depending on which frame's calibration is picked, proving the
        # per-frame lookup is actually used and not just a shared one
        calib_1x = scaled_calibration(scale=100.0)
        calib_2x = scaled_calibration(scale=200.0)
        net_y_1x, _ = net_pixel_y_range(calib_1x)
        net_y_2x, _ = net_pixel_y_range(calib_2x)

        positions_1x = [pos(0, 0.0, net_y_1x), pos(1, 50.0, net_y_1x)]
        positions_2x = [pos(0, 0.0, net_y_2x), pos(1, 50.0, net_y_2x)]

        shots_1x = estimate_net_crossing_speeds(positions_1x, fps=30.0, calibrations_by_frame={0: calib_1x, 1: calib_1x})
        shots_2x = estimate_net_crossing_speeds(positions_2x, fps=30.0, calibrations_by_frame={0: calib_2x, 1: calib_2x})

        self.assertEqual(len(shots_1x), 1)
        self.assertEqual(len(shots_2x), 1)
        # double the scale (pixel = world*100 vs *200) -> the same 50px
        # displacement maps to HALF the real-world distance -> half the speed
        self.assertAlmostEqual(shots_2x[0].peak_speed, shots_1x[0].peak_speed / 2, places=1)


if __name__ == "__main__":
    unittest.main()
