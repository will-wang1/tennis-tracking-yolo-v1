import unittest

from src.analysis.flight_segmenter import find_flight_segments
from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.analysis.speed_estimator import (
    estimate_flight_net_speeds,
    net_pixel_line,
    ShotSpeed,
    estimate_net_crossing_speeds,
    estimate_shot_speeds,
    instantaneous_speeds,
    merge_with_net_crossing_speeds,
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
            # a real, moving ball crossing near the net over several frames
            pos(0, 500.0, net_y - 10),
            pos(1, 520.0, net_y - 5),
            pos(2, 540.0, net_y),
            pos(3, 560.0, net_y + 5),
            # a static false-positive lock-on sitting right at the net -
            # barely moves frame to frame, must NOT be read as a real shot
            # even though it spans enough frames to pass min_window_frames
            pos(10, 700.0, net_y),
            pos(11, 701.0, net_y + 1),
            pos(12, 700.5, net_y - 1),
            pos(13, 700.2, net_y + 0.5),
        ]
        calibrations = {p.frame_idx: calibration for p in positions}
        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].start_frame, 0)
        self.assertEqual(shots[0].end_frame, 3)
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

    def test_net_crossing_speed_averages_net_displacement_over_window(self):
        # net displacement is first->last (0 -> 70px), NOT the summed
        # per-step path length (10 + 50 + 10 = 70px here too, coincidentally
        # equal - the point is this is an average over the whole window,
        # not a peak instantaneous reading picking out the 50px step alone)
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [
            pos(0, 0.0, net_y),
            pos(1, 10.0, net_y),
            pos(2, 60.0, net_y),
            pos(3, 70.0, net_y),
        ]
        calibrations = {p.frame_idx: calibration for p in positions}

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].start_frame, 0)
        self.assertEqual(shots[0].end_frame, 3)
        self.assertEqual(shots[0].peak_frame, 3)
        # 70px / 100(scale) = 0.7m over 3 frames @ 30fps = 0.1s -> 7 m/s -> 25.2 km/h
        self.assertAlmostEqual(shots[0].peak_speed, 25.2, places=1)

    def test_net_crossing_window_shorter_than_min_frames_is_dropped(self):
        # only 2 qualifying frames - a real, large displacement that would
        # have produced a (noisy) reading under the old pairwise logic, but
        # min_window_frames=4 by default requires more frames than this
        # before trusting an average
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [pos(0, 0.0, net_y), pos(1, 50.0, net_y)]
        calibrations = {p.frame_idx: calibration for p in positions}

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(shots, [])

    def test_net_crossing_min_window_frames_is_tunable(self):
        # the same short window that test_net_crossing_window_shorter_than_min_frames_is_dropped
        # rejects at the default threshold is accepted once min_window_frames is lowered to match
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [pos(0, 0.0, net_y), pos(1, 50.0, net_y)]
        calibrations = {p.frame_idx: calibration for p in positions}

        shots = estimate_net_crossing_speeds(
            positions, fps=30.0, calibrations_by_frame=calibrations, min_window_frames=2
        )

        self.assertEqual(len(shots), 1)

    def test_net_crossing_speed_ignores_frames_missing_a_calibration(self):
        # the court detector lost the court on frame 2 (e.g. motion blur),
        # splitting what would otherwise be one 5-frame window into two
        # 2-frame pieces, both too short to pass min_window_frames - the
        # missing frame must not be silently skipped over as if it were
        # still part of one continuous window
        calibration = scaled_calibration(scale=100.0)
        net_y = 1188.5
        positions = [pos(i, i * 10.0, net_y) for i in range(5)]  # frames 0..4
        calibrations = {i: calibration for i in range(5) if i != 2}

        shots = estimate_net_crossing_speeds(positions, fps=30.0, calibrations_by_frame=calibrations)

        self.assertEqual(shots, [])

    def test_net_crossing_speed_uses_each_frames_own_calibration(self):
        # a panning camera: two calibrations at different scales, each with
        # its OWN correctly-located net line in pixel space (a larger scale
        # reprojects the net line further down the frame too) - the same
        # pixel displacement must map to a different real-world distance
        # depending on which calibration is picked, proving the per-frame
        # lookup is actually used and not just a shared one
        calib_1x = scaled_calibration(scale=100.0)
        calib_2x = scaled_calibration(scale=200.0)
        net_y_1x, _ = net_pixel_y_range(calib_1x)
        net_y_2x, _ = net_pixel_y_range(calib_2x)

        positions_1x = [pos(i, i * 15.0, net_y_1x) for i in range(4)]
        positions_2x = [pos(i, i * 15.0, net_y_2x) for i in range(4)]

        shots_1x = estimate_net_crossing_speeds(
            positions_1x, fps=30.0, calibrations_by_frame={i: calib_1x for i in range(4)}
        )
        shots_2x = estimate_net_crossing_speeds(
            positions_2x, fps=30.0, calibrations_by_frame={i: calib_2x for i in range(4)}
        )

        self.assertEqual(len(shots_1x), 1)
        self.assertEqual(len(shots_2x), 1)
        # double the scale (pixel = world*100 vs *200) -> the same 45px net
        # displacement maps to HALF the real-world distance -> half the speed
        self.assertAlmostEqual(shots_2x[0].peak_speed, shots_1x[0].peak_speed / 2, places=1)

    def test_instantaneous_speed_uses_per_frame_calibration_over_static(self):
        calib_1x = scaled_calibration(scale=100.0)
        calib_2x = scaled_calibration(scale=200.0)
        positions = [pos(0, 0.0, 0.0), pos(1, 10.0, 0.0)]

        # a static `calibration` is passed too, but calibrations_by_frame
        # must win - proving the per-frame lookup takes priority
        speeds = instantaneous_speeds(
            positions, fps=30.0, calibration=calib_1x, calibrations_by_frame={1: calib_2x}
        )

        expected = instantaneous_speeds(positions, fps=30.0, calibration=calib_2x)
        self.assertAlmostEqual(speeds[1], expected[1])

    def test_estimate_shot_speeds_unit_is_kmh_with_calibrations_by_frame(self):
        calibration = scaled_calibration(scale=100.0)
        positions = [pos(0, 0, 0), pos(1, 10, 0), pos(2, 20, 0)]

        shots = estimate_shot_speeds(
            positions, breakpoint_frames=[], fps=30.0, calibrations_by_frame={0: calibration, 1: calibration, 2: calibration}
        )

        self.assertEqual(shots[0].unit, "km/h")

    def test_merge_prefers_net_crossing_speed_when_it_overlaps(self):
        fallback = [ShotSpeed(start_frame=0, end_frame=20, peak_frame=5, peak_speed=80.0, unit="km/h")]
        net = [ShotSpeed(start_frame=10, end_frame=11, peak_frame=10, peak_speed=150.0, unit="km/h")]

        merged = merge_with_net_crossing_speeds(fallback, net)

        self.assertEqual(len(merged), 1)
        # the fallback shot's own (start, end) window is kept...
        self.assertEqual(merged[0].start_frame, 0)
        self.assertEqual(merged[0].end_frame, 20)
        # ...but the more accurate net-crossing peak is used instead
        self.assertEqual(merged[0].peak_speed, 150.0)
        self.assertEqual(merged[0].peak_frame, 10)

    def test_merge_keeps_fallback_speed_when_no_net_crossing_overlaps(self):
        # a shot the ball was never tracked crossing the net during (e.g.
        # occlusion) - must still get a reading, just the less precise one
        fallback = [ShotSpeed(start_frame=0, end_frame=20, peak_frame=5, peak_speed=80.0, unit="km/h")]
        net = [ShotSpeed(start_frame=50, end_frame=51, peak_frame=50, peak_speed=150.0, unit="km/h")]

        merged = merge_with_net_crossing_speeds(fallback, net)

        self.assertEqual(merged, fallback)

    def test_merge_does_not_duplicate_a_reading_across_a_shot_boundary(self):
        # a net_shot's own WINDOW can straddle a shot boundary even when its
        # peak_frame - the actual crossing instant - sits cleanly inside one
        # shot: estimate_flight_net_speeds fits on the unsmoothed trajectory,
        # whose flight endpoints land a frame or so from the bounce
        # segment_shots cuts shots at. Measured on the zverev clip this
        # windows a net_shot into TWO adjacent shots and overwrites the
        # second with the first's exact reading - the displayed speed
        # appeared not to update between two consecutive shots because it
        # had silently been copied, not because nothing new was measured.
        fallback = [
            ShotSpeed(start_frame=0, end_frame=20, peak_frame=10, peak_speed=80.0, unit="km/h"),
            ShotSpeed(start_frame=20, end_frame=40, peak_frame=30, peak_speed=90.0, unit="km/h"),
        ]
        # this net_shot's window (17-22) crosses the boundary at 20, but its
        # peak_frame (18) belongs to the FIRST shot only
        net = [ShotSpeed(start_frame=17, end_frame=22, peak_frame=18, peak_speed=150.0, unit="km/h")]

        merged = merge_with_net_crossing_speeds(fallback, net)

        self.assertEqual(merged[0].peak_speed, 150.0)  # upgraded, correctly
        self.assertEqual(merged[1].peak_speed, 90.0)  # untouched - not this shot's reading

    def test_merge_uses_the_shots_half_open_boundary_convention(self):
        # matches segment_shots: start_frame < frame <= end_frame, so a
        # crossing exactly AT a boundary belongs to the shot ending there,
        # not the one beginning there
        fallback = [
            ShotSpeed(start_frame=0, end_frame=20, peak_frame=10, peak_speed=80.0, unit="km/h"),
            ShotSpeed(start_frame=20, end_frame=40, peak_frame=30, peak_speed=90.0, unit="km/h"),
        ]
        net = [ShotSpeed(start_frame=20, end_frame=20, peak_frame=20, peak_speed=150.0, unit="km/h")]

        merged = merge_with_net_crossing_speeds(fallback, net)

        self.assertEqual(merged[0].peak_speed, 150.0)
        self.assertEqual(merged[1].peak_speed, 90.0)

    def test_merge_gives_every_fallback_shot_a_reading(self):
        # the core guarantee this function exists for: every fallback shot
        # survives the merge (nothing dropped), regardless of net coverage
        fallback = [
            ShotSpeed(start_frame=0, end_frame=10, peak_frame=5, peak_speed=60.0, unit="km/h"),
            ShotSpeed(start_frame=10, end_frame=20, peak_frame=15, peak_speed=70.0, unit="km/h"),
            ShotSpeed(start_frame=20, end_frame=30, peak_frame=25, peak_speed=90.0, unit="km/h"),
        ]
        net = [ShotSpeed(start_frame=15, end_frame=16, peak_frame=15, peak_speed=140.0, unit="km/h")]

        merged = merge_with_net_crossing_speeds(fallback, net)

        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[1].peak_speed, 140.0)  # overlapped, upgraded
        self.assertEqual(merged[0].peak_speed, 60.0)  # no overlap, kept as-is
        self.assertEqual(merged[2].peak_speed, 90.0)  # no overlap, kept as-is


class FlightNetSpeedTest(unittest.TestCase):
    """Speed read off a fitted flight where it crosses the net. See
    speed_estimator.estimate_flight_net_speeds."""

    FPS = 30.0

    def _calibration(self, scale=20.0):
        """Pixels = world metres * scale, so a known world speed produces a
        known pixel speed and the arithmetic can be checked by hand."""
        from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS

        names = [
            "baseline_far_left",
            "baseline_far_right",
            "baseline_near_left",
            "baseline_near_right",
        ]
        pixel_points = {
            name: (wx * scale, wy * scale)
            for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items()
            if name in names
        }
        return CourtCalibration.from_keypoints(pixel_points)

    def _straight_flight(self, metres_per_second, scale=20.0, frames=40, start_frame=10):
        """A ball crossing the net down the middle at a known ground speed."""
        court_length = max(y for _, y in FULL_COURT_REFERENCE_POINTS.values())
        net_y = court_length / 2
        per_frame = metres_per_second / self.FPS
        positions = [
            TrackedPosition(
                frame_idx=start_frame + i,
                x=5.0 * scale,
                y=(net_y - (frames / 2 - i) * per_frame) * scale,
                interpolated=False,
            )
            for i in range(frames)
        ]
        return find_flight_segments(positions)

    def _speeds(self, segments, scale=20.0, **kwargs):
        calibrations = {i: self._calibration(scale) for i in range(0, 200)}
        return estimate_flight_net_speeds(segments, self.FPS, calibrations, **kwargs)

    def test_measures_a_known_ground_speed(self):
        segments = self._straight_flight(metres_per_second=30.0)
        shots = self._speeds(segments)

        self.assertEqual(len(shots), 1)
        self.assertAlmostEqual(shots[0].peak_speed, 30.0 * 3.6, delta=1.0)
        self.assertEqual(shots[0].unit, "km/h")

    def test_reports_the_crossing_frame(self):
        segments = self._straight_flight(metres_per_second=30.0, frames=40, start_frame=10)
        shots = self._speeds(segments)

        # the fixture is built to cross the net at its midpoint
        self.assertAlmostEqual(shots[0].peak_frame, 30, delta=2)

    def test_ignores_a_flight_that_never_reaches_the_net(self):
        court_length = max(y for _, y in FULL_COURT_REFERENCE_POINTS.values())
        scale = 20.0
        positions = [
            TrackedPosition(
                frame_idx=10 + i,
                x=5.0 * scale,
                y=(court_length * 0.9 - i * 0.1) * scale,
                interpolated=False,
            )
            for i in range(40)
        ]

        self.assertEqual(self._speeds(find_flight_segments(positions)), [])

    def test_ignores_a_flight_too_short_to_measure(self):
        segments = self._straight_flight(metres_per_second=30.0, frames=40)
        self.assertTrue(self._speeds(segments))

        self.assertEqual(self._speeds(segments, min_flight_frames=100), [])

    def test_rejects_an_impossible_speed(self):
        segments = self._straight_flight(metres_per_second=30.0)

        self.assertEqual(self._speeds(segments, max_speed_kmh=10.0), [])

    def test_needs_no_detection_at_the_crossing_itself(self):
        # the whole point of measuring off the fitted flight: the ball is
        # small, fast and often occluded exactly at the net, and the curve
        # is defined there whether or not the detector saw it
        court_length = max(y for _, y in FULL_COURT_REFERENCE_POINTS.values())
        scale, per_frame = 20.0, 30.0 / self.FPS
        positions = [
            TrackedPosition(
                frame_idx=10 + i,
                x=5.0 * scale,
                y=(court_length / 2 - (20 - i) * per_frame) * scale,
                interpolated=False,
            )
            for i in range(40)
            if not 17 <= i <= 23  # nothing detected across the crossing
        ]
        segments = find_flight_segments(positions)
        self.assertTrue(segments, "a dropout should not break the flight")
        shots = self._speeds(segments)

        self.assertEqual(len(shots), 1)
        self.assertAlmostEqual(shots[0].peak_speed, 108.0, delta=2.0)

    def test_returns_nothing_without_a_calibration(self):
        segments = self._straight_flight(metres_per_second=30.0)

        self.assertEqual(estimate_flight_net_speeds(segments, self.FPS, {}), [])


class NetPixelLineTest(unittest.TestCase):
    def test_the_net_line_separates_the_two_halves(self):
        from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS

        scale = 20.0
        names = [
            "baseline_far_left",
            "baseline_far_right",
            "baseline_near_left",
            "baseline_near_right",
        ]
        calibration = CourtCalibration.from_keypoints(
            {
                name: (wx * scale, wy * scale)
                for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items()
                if name in names
            }
        )
        a, b, c = net_pixel_line(calibration)
        court_length = max(y for _, y in FULL_COURT_REFERENCE_POINTS.values())

        far = a * (5 * scale) + b * (1.0 * scale) + c
        near = a * (5 * scale) + b * ((court_length - 1.0) * scale) + c

        self.assertLess(far * near, 0.0, "the halves must land on opposite sides")


if __name__ == "__main__":
    unittest.main()
