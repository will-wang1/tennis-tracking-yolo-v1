import unittest

import numpy as np

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.analysis.flight_segmenter import find_flight_segments
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.tracking.ball_tracker import TrackedPosition
from src.analysis.speed_estimator import ShotSpeed
from src.visualize.draw import (
    ARC_COLOR,
    BounceMarkerDrawer,
    CourtOverlayDrawer,
    ImpactMarkerDrawer,
    ShotArcDrawer,
    StatsPanelDrawer,
    TrailDrawer,
)


def scaled_calibration(scale):
    # simple, exactly-known pixel<->world relationship: pixel = world * scale
    names = ["baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right"]
    pixel_points = {
        name: (wx * scale, wy * scale)
        for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items()
        if name in names
    }
    return CourtCalibration.from_keypoints(pixel_points)


class CourtOverlayDrawerTest(unittest.TestCase):
    def setUp(self):
        self.calibration = scaled_calibration(scale=1.0)
        self.drawer = CourtOverlayDrawer()

    def test_draw_mutates_and_returns_same_frame(self):
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        result = self.drawer.draw(frame, self.calibration)

        self.assertIs(result, frame)
        self.assertTrue((frame != 0).any())  # something got drawn

    def test_none_calibration_leaves_frame_untouched(self):
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        result = self.drawer.draw(frame, None)

        self.assertIs(result, frame)
        self.assertTrue((frame == 0).all())

    def test_draws_a_corner_marker_at_the_reprojected_pixel_location(self):
        # pixel = world * 1.0, so baseline_far_left (world (0, 0)) reprojects
        # to pixel (0, 0) - a corner marker (circle, radius 8) should light
        # up pixels right around there
        frame = np.zeros((30, 30, 3), dtype=np.uint8)

        self.drawer.draw(frame, self.calibration)

        self.assertTrue((frame[0:5, 0:5] != 0).any())

    def test_recomputes_reprojection_fresh_each_call_for_a_different_calibration(self):
        # a "panning camera" scenario: two different calibrations passed to
        # the SAME drawer instance across two calls must each draw at their
        # own reprojected locations - proving there's no stale cached state
        # from a previous call
        small_scale = scaled_calibration(scale=1.0)
        large_scale = scaled_calibration(scale=2.0)

        frame_small = np.zeros((60, 60, 3), dtype=np.uint8)
        self.drawer.draw(frame_small, small_scale)

        frame_large = np.zeros((60, 60, 3), dtype=np.uint8)
        self.drawer.draw(frame_large, large_scale)

        # baseline_near_right reprojects further out under the 2x
        # calibration than the 1x one, so the two frames' drawn pixels
        # must differ
        self.assertFalse(np.array_equal(frame_small, frame_large))


def _flight_positions(start_frame, count, x0, y0, vx, vy, gravity=1.2):
    return [
        TrackedPosition(
            frame_idx=start_frame + i,
            x=x0 + vx * i,
            y=y0 + vy * i + 0.5 * gravity * i * i,
            interpolated=False,
        )
        for i in range(count)
    ]


def _blank(height=400, width=640):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _painted(frame):
    """How many pixels the drawer touched."""
    return int(np.count_nonzero(frame.any(axis=2)))


class ShotArcDrawerTest(unittest.TestCase):
    def setUp(self):
        positions = _flight_positions(10, 30, 100.0, 300.0, 12.0, -14.0)
        self.segments = find_flight_segments(positions)
        self.assertTrue(self.segments, "fixture should contain one clean flight")
        self.drawer = ShotArcDrawer(self.segments)

    def test_draws_nothing_before_the_first_flight_starts(self):
        # extrapolating the curve backwards would draw a path the ball never
        # took, so there is simply no shot to show yet
        frame = _blank()
        self.drawer.draw(frame, 2)

        self.assertEqual(_painted(frame), 0)

    def test_draws_the_arc_once_the_flight_is_under_way(self):
        frame = _blank()
        self.drawer.draw(frame, 25)

        self.assertGreater(_painted(frame), 0)

    def test_the_arc_grows_as_the_ball_travels(self):
        early, late = _blank(), _blank()
        self.drawer.draw(early, 18)
        self.drawer.draw(late, 32)

        self.assertGreater(_painted(late), _painted(early))

    def test_stops_growing_when_the_flight_ends(self):
        # both are past the end, so neither carries a ball marker - what is
        # left is the finished arc, and it must not have moved
        end = self.segments[0].end_frame
        just_after, well_after = _blank(), _blank()
        self.drawer.draw(just_after, end + 5)
        self.drawer.draw(well_after, end + 40)

        self.assertTrue(np.array_equal(just_after, well_after))

    def test_holds_the_finished_arc_until_the_next_flight(self):
        frame = _blank()
        self.drawer.draw(frame, self.segments[0].end_frame + 15)

        self.assertGreater(_painted(frame), 0)

    def test_switches_to_the_new_flight_once_it_starts(self):
        second = _flight_positions(120, 30, 500.0, 300.0, -12.0, -14.0)
        drawer = ShotArcDrawer(self.segments + find_flight_segments(second))

        self.assertEqual(drawer.segment_for(30), self.segments[0])
        self.assertEqual(drawer.segment_for(135).start_frame, 120)

    def test_draws_in_the_arc_colour(self):
        frame = _blank()
        self.drawer.draw(frame, 30)
        painted = frame[frame.any(axis=2)]

        # anti-aliasing dims the line, so check the hue rather than equality
        self.assertTrue((painted[:, 0] <= painted[:, 1]).all())
        self.assertEqual(ARC_COLOR, (0, 255, 255))

    def test_marks_the_ball_on_its_own_fitted_curve(self):
        frame = _blank()
        self.drawer.draw(frame, 30)
        x, y = self.segments[0].position(30)

        self.assertTrue(frame[int(round(y)), int(round(x))].any())

    def test_a_false_positive_cannot_move_the_ball_marker(self):
        # the whole point: the marker is read off the fitted flight, so a
        # detection in the crowd has nothing to move
        strays = _flight_positions(10, 30, 100.0, 300.0, 12.0, -14.0)
        strays[14] = TrackedPosition(frame_idx=strays[14].frame_idx, x=20.0, y=20.0, interpolated=False)
        drawer = ShotArcDrawer(find_flight_segments(strays))
        frame = _blank()
        drawer.draw(frame, 30)

        self.assertFalse(frame[15:26, 15:26].any())

    def test_no_ball_marker_between_two_flights(self):
        # at a bounce or a strike there is no flight to read a position off,
        # and inventing one from a curve that has ended would be a guess
        end = self.segments[0].end_frame
        with_ball, without = _blank(), _blank()
        self.drawer.draw(with_ball, end)
        self.drawer.draw(without, end + 6)

        self.assertGreater(_painted(with_ball), _painted(without))

    def test_survives_having_no_flights_at_all(self):
        frame = _blank()
        ShotArcDrawer([]).draw(frame, 10)

        self.assertEqual(_painted(frame), 0)

    def test_keeps_the_arc_inside_the_frame(self):
        # a fitted curve can leave the image; drawing it must not raise or
        # write out of bounds
        positions = _flight_positions(10, 30, 600.0, 380.0, 40.0, -40.0)
        frame = _blank()
        ShotArcDrawer(find_flight_segments(positions)).draw(frame, 39)

        self.assertEqual(frame.shape, (400, 640, 3))


def _impact(frame_idx, kind, x=0.0, y=0.0):
    return BounceCandidate(
        frame_idx=frame_idx, t=float(frame_idx), x=x, y=y,
        restitution=0.5, horizontal_ratio=0.8, speed_ratio=0.7, rmse=1.0,
        is_bounce=kind == "bounce", kind=kind,
    )


class ShotArcGroupingTest(unittest.TestCase):
    """A shot is racket to racket, so it spans the bounce in the middle -
    two fitted flights, drawn as one path. See ShotArcDrawer."""

    def setUp(self):
        # flight one lands at frame 41; flight two runs on from there
        self.first = find_flight_segments(_flight_positions(10, 30, 100.0, 300.0, 12.0, -14.0))
        self.second = find_flight_segments(_flight_positions(45, 30, 470.0, 300.0, 12.0, -14.0))
        self.assertEqual(len(self.first), 1)
        self.assertEqual(len(self.second), 1)
        self.segments = self.first + self.second
        self.boundary = (self.first[0].end_frame + self.second[0].start_frame) // 2

    def _painted_at(self, drawer, frame_idx):
        frame = _blank(600, 1100)
        drawer.draw(frame, frame_idx)
        return _painted(frame)

    def _first_flight_shown(self, drawer, frame_idx=70, radius=6):
        """Is the EARLIER flight still on screen? Probed at a point on its
        own curve, well away from the later flight - comparing pixel counts
        between two drawers no longer works now that a flight's drawn extent
        depends on the neighbour it meets."""
        frame = _blank(600, 1100)
        drawer.draw(frame, frame_idx)
        x, y = (int(round(v)) for v in self.first[0].position(20))
        return bool(frame[y - radius:y + radius, x - radius:x + radius].any())

    def test_a_bounce_between_two_flights_keeps_them_on_one_shot(self):
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "bounce")])

        self.assertTrue(self._first_flight_shown(drawer))

    def test_a_contact_between_them_starts_a_new_shot(self):
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "contact")])

        self.assertFalse(self._first_flight_shown(drawer))

    def test_an_unattributed_impact_also_starts_a_new_shot(self):
        # the classifier declined to say what happened; drawing through it
        # would assert a continuity nothing supports
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "unknown")])

        self.assertFalse(self._first_flight_shown(drawer))

    def test_the_two_flights_of_a_shot_are_drawn_as_one_connected_path(self):
        # separate polylines leave the gap between the flights visible; one
        # line closes it, and the join reads as the corner a bounce is
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "bounce")])
        frame = _blank(600, 1100)
        drawer.draw(frame, 70)

        first_end = self.first[0].position(self.first[0].end_frame)
        second_start = self.second[0].position(self.second[0].start_frame)
        midpoint = (
            int(round((first_end[0] + second_start[0]) / 2)),
            int(round((first_end[1] + second_start[1]) / 2)),
        )
        painted = frame[midpoint[1] - 6:midpoint[1] + 6, midpoint[0] - 6:midpoint[0] + 6]

        self.assertTrue(painted.any(), "the gap between the two flights should be bridged")

    def test_a_flight_is_drawn_over_its_own_samples_only(self):
        # extrapolating a fitted curve past its data to meet its neighbour
        # makes the path double back on itself - see ShotArcDrawer
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "bounce")])
        points = drawer._arc_points(
            self.first[0], float(self.first[0].start_frame), float(self.first[0].end_frame)
        )

        self.assertEqual(points[0], tuple(int(round(v)) for v in self.first[0].position(self.first[0].start_frame)))
        self.assertEqual(points[-1], tuple(int(round(v)) for v in self.first[0].position(self.first[0].end_frame)))

    def test_a_split_flight_with_no_impact_between_is_rejoined(self):
        # consecutive frames, nothing detected between them: one flight the
        # segmenter happened to cut in two
        halves = find_flight_segments(_flight_positions(10, 22, 100.0, 300.0, 12.0, -14.0))
        halves += find_flight_segments(_flight_positions(33, 22, 376.0, 60.0, 12.0, 6.0))
        drawer = ShotArcDrawer(halves)

        self.assertEqual(drawer._shot_start, [0, 0])

    def test_a_long_unexplained_gap_is_not_drawn_through(self):
        far = find_flight_segments(_flight_positions(200, 30, 100.0, 300.0, 12.0, -14.0))
        drawer = ShotArcDrawer(self.first + far)

        self.assertEqual(drawer._shot_start, [0, 1])

    def test_the_earlier_flight_is_drawn_whole_while_the_later_one_grows(self):
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "bounce")])
        early = self._painted_at(drawer, 50)
        later = self._painted_at(drawer, 70)

        self.assertGreater(later, early)

    def test_the_ball_marker_rides_the_current_flight_not_the_first(self):
        drawer = ShotArcDrawer(self.segments, [_impact(self.boundary, "bounce")])
        frame = _blank(600, 1100)
        drawer.draw(frame, 60)
        x, y = self.second[0].position(60)

        self.assertTrue(frame[int(round(y)), int(round(x))].any())


class WorldToPixelTest(unittest.TestCase):
    def test_round_trips_with_pixel_to_world(self):
        calibration = scaled_calibration(scale=37.0)

        world = calibration.pixel_to_world(120.0, 340.0)
        pixel = calibration.world_to_pixel(*world)

        self.assertAlmostEqual(pixel[0], 120.0, places=4)
        self.assertAlmostEqual(pixel[1], 340.0, places=4)

    def test_a_fixed_court_point_reprojects_differently_under_a_different_calibration(self):
        # this is the whole mechanism a panning/zooming camera relies on -
        # the same court point must NOT reproject to the same pixel once
        # the calibration (camera pose) has changed
        narrow = scaled_calibration(scale=1.0)
        wide = scaled_calibration(scale=2.0)

        self.assertNotEqual(narrow.world_to_pixel(5.0, 10.0), wide.world_to_pixel(5.0, 10.0))


def _impact_for_trail(t, frame_idx=None, kind="contact"):
    return BounceCandidate(
        frame_idx=frame_idx if frame_idx is not None else int(round(t)),
        t=float(t), x=0.0, y=0.0,
        restitution=0.0, horizontal_ratio=0.0, speed_ratio=0.0, rmse=0.0,
        kind=kind,
    )


class TrailDrawerTest(unittest.TestCase):
    def setUp(self):
        self.drawer = TrailDrawer(trail_length=15, fps=30.0)

    def _position(self, frame_idx, x=100.0, y=100.0):
        return TrackedPosition(frame_idx=frame_idx, x=x, y=y, interpolated=False)

    def test_a_position_is_drawn(self):
        frame = _blank(300, 300)
        self.drawer.draw(frame, self._position(0))
        self.assertTrue(frame.any())

    def test_no_impact_data_behaves_exactly_as_before(self):
        # fps=None means impact-based clearing is inert - the plain sliding
        # trail keeps working with no impact argument at all
        drawer = TrailDrawer(trail_length=15)
        frame = _blank(300, 300)
        drawer.draw(frame, self._position(0))
        self.assertTrue(frame.any())

    def test_a_rally_gap_clears_the_trail(self):
        # 30fps, default rally_gap_seconds=4.0 -> anything past 120 frames
        self.drawer.draw(_blank(300, 300), self._position(0), _impact_for_trail(0, frame_idx=0))
        far_frame = _blank(300, 300)
        # no new position this frame (tracking hasn't resumed yet) - only
        # the new rally's impact arrives
        self.drawer.draw(far_frame, None, _impact_for_trail(500, frame_idx=500))

        self.assertEqual(len(self.drawer.trail), 0)
        self.assertFalse(far_frame.any())

    def test_within_the_rally_gap_the_trail_is_not_cleared(self):
        self.drawer.draw(_blank(300, 300), self._position(0), _impact_for_trail(0, frame_idx=0))
        self.drawer.draw(_blank(300, 300), self._position(30), _impact_for_trail(30, frame_idx=30))

        self.assertEqual(len(self.drawer.trail), 2)

    def test_the_new_rallys_position_still_draws_after_clearing(self):
        self.drawer.draw(_blank(300, 300), self._position(0, x=50.0, y=50.0), _impact_for_trail(0, frame_idx=0))
        frame = _blank(300, 300)
        self.drawer.draw(frame, self._position(500, x=200.0, y=200.0), _impact_for_trail(500, frame_idx=500))

        self.assertEqual(len(self.drawer.trail), 1)
        self.assertTrue(frame[200, 200].any())
        self.assertFalse(frame[50, 50].any())


class BounceMarkerDrawerTest(unittest.TestCase):
    """A bounce is a fixed spot on the COURT, so its marker must track the
    court as the camera pans/zooms, not stay fixed on screen."""

    def setUp(self):
        self.narrow = scaled_calibration(scale=1.0)
        self.wide = scaled_calibration(scale=2.0)
        self.bounce = BounceEvent(frame_idx=10, x=999.0, y=999.0, world_x=5.0, world_y=10.0)
        self.drawer = BounceMarkerDrawer(fps=30.0)

    def test_draws_at_the_reprojected_world_position_not_the_stored_pixel(self):
        frame = _blank(1200, 1200)
        self.drawer.draw(frame, self.bounce, self.narrow)
        x, y = self.narrow.world_to_pixel(5.0, 10.0)

        self.assertTrue(frame[int(y), int(x)].any())
        # the marker's own stored (x, y) = (999, 999) - the pixel field it
        # was captured at, not where a court-relative marker belongs
        self.assertFalse(frame[989:1010, 989:1010].any())

    def test_the_marker_moves_when_the_calibration_changes(self):
        # this is the fix itself: the same marker, redrawn under a different
        # calibration, must land somewhere else - never at a fixed pixel
        narrow_frame, wide_frame = _blank(600, 1200), _blank(600, 1200)
        self.drawer.draw(narrow_frame, self.bounce, self.narrow)
        second = BounceMarkerDrawer(fps=30.0)
        second.draw(wide_frame, self.bounce, self.wide)

        self.assertFalse(np.array_equal(narrow_frame, wide_frame))

    def test_the_marker_persists_and_still_tracks_on_later_calls(self):
        frame = _blank(600, 1200)
        self.drawer.draw(frame, self.bounce, self.narrow)
        later = _blank(600, 1200)
        self.drawer.draw(later, None, self.wide)  # no new bounce this frame
        x, y = self.wide.world_to_pixel(5.0, 10.0)

        self.assertTrue(later[int(y), int(x)].any())

    def test_falls_back_to_the_stored_pixel_without_a_calibration(self):
        frame = _blank(1200, 1200)
        self.drawer.draw(frame, self.bounce, None)

        self.assertTrue(frame[999, 999].any())

    def test_falls_back_to_the_stored_pixel_when_the_bounce_has_no_world_position(self):
        undated = BounceEvent(frame_idx=10, x=999.0, y=999.0, world_x=None, world_y=None)
        frame = _blank(1200, 1200)
        BounceMarkerDrawer(fps=30.0).draw(frame, undated, self.narrow)

        self.assertTrue(frame[999, 999].any())

    def test_a_gap_beyond_the_rally_threshold_clears_earlier_markers(self):
        early = BounceEvent(frame_idx=10, x=100.0, y=100.0, world_x=5.0, world_y=10.0)
        # 30fps, gap_seconds default 4.0 -> anything more than 120 frames later
        later = BounceEvent(frame_idx=500, x=200.0, y=200.0, world_x=80.0, world_y=90.0)

        self.drawer.draw(_blank(600, 1200), early, self.narrow)
        frame = _blank(600, 1200)
        self.drawer.draw(frame, later, self.narrow)

        self.assertEqual(self.drawer.markers, [later])
        x, y = self.narrow.world_to_pixel(5.0, 10.0)
        self.assertFalse(frame[int(y) - 5:int(y) + 5, int(x) - 5:int(x) + 5].any())

    def test_a_gap_within_the_rally_threshold_keeps_earlier_markers(self):
        early = BounceEvent(frame_idx=10, x=100.0, y=100.0, world_x=5.0, world_y=10.0)
        soon_after = BounceEvent(frame_idx=40, x=200.0, y=200.0, world_x=6.0, world_y=11.0)  # 1s later

        self.drawer.draw(_blank(600, 1200), early, self.narrow)
        self.drawer.draw(_blank(600, 1200), soon_after, self.narrow)

        self.assertEqual(self.drawer.markers, [early, soon_after])


class ImpactMarkerDrawerBounceTrackingTest(unittest.TestCase):
    """Same fix, same reasoning, applied to the --contacts path: a bounce
    marker here must also track the court rather than stay fixed on screen.
    Contacts need none of this - see the drawer's own docstring - so they
    are not covered here."""

    def _bounce(self, frame_idx=10, x=999.0, y=999.0, kind="bounce"):
        return BounceCandidate(
            frame_idx=frame_idx, t=float(frame_idx), x=x, y=y,
            restitution=0.5, horizontal_ratio=0.8, speed_ratio=0.7, rmse=1.0,
            is_bounce=kind == "bounce", kind=kind,
        )

    def setUp(self):
        self.narrow = scaled_calibration(scale=1.0)
        self.wide = scaled_calibration(scale=2.0)
        self.impact = self._bounce()
        self.drawer = ImpactMarkerDrawer(fps=30.0)

    def test_computes_world_position_from_the_calibration_at_its_own_frame(self):
        frame = _blank(600, 1200)
        # frame_idx 10 == the impact's own frame, so this IS the impact's
        # own calibration, per ImpactMarkerDrawer's contract
        self.drawer.draw(frame, 10, {10: self.impact}, self.narrow)
        world_x, world_y, *_ = self.drawer.bounces[0]

        self.assertEqual((world_x, world_y), self.narrow.pixel_to_world(999.0, 999.0))

    def test_the_marker_tracks_a_later_frames_calibration(self):
        frame = _blank(2000, 2000)
        self.drawer.draw(frame, 10, {10: self.impact}, self.narrow)
        later = _blank(2000, 2000)
        self.drawer.draw(later, 40, {10: self.impact}, self.wide)
        world = self.narrow.pixel_to_world(999.0, 999.0)
        x, y = self.wide.world_to_pixel(*world)

        self.assertTrue(later[int(y), int(x)].any())

    def test_falls_back_to_the_stored_pixel_without_any_calibration(self):
        frame = _blank(1200, 1200)
        self.drawer.draw(frame, 10, {10: self.impact}, None)

        self.assertTrue(frame[999, 999].any())

    def test_an_unknown_impact_draws_no_bounce_marker(self):
        frame = _blank(600, 1200)
        self.drawer.draw(frame, 10, {10: self._bounce(kind="unknown")}, self.narrow)

        self.assertEqual(self.drawer.bounces, [])

    def test_a_rally_gap_clears_earlier_bounce_markers(self):
        # 30fps, default rally_gap_seconds=4.0 -> anything past 120 frames
        first = self._bounce(frame_idx=10)
        later = self._bounce(frame_idx=500, x=50.0, y=50.0)
        self.drawer.draw(_blank(600, 1200), 10, {10: first}, self.narrow)

        self.drawer.draw(_blank(600, 1200), 500, {500: later}, self.narrow)

        self.assertEqual(len(self.drawer.bounces), 1)
        self.assertEqual(self.drawer.bounces[0][2:4], (50.0, 50.0))

    def test_a_contact_still_resets_the_gap_even_though_it_draws_no_marker(self):
        # the rally-gap clock has to track ANY impact, not just bounces -
        # otherwise a rally with contacts spaced under the threshold, but
        # with no bounce in between for a long stretch, would wrongly clear
        first = self._bounce(frame_idx=10)
        contact_mid_rally = self._bounce(frame_idx=100, kind="contact")
        second_bounce = self._bounce(frame_idx=150, x=50.0, y=50.0)
        self.drawer.draw(_blank(600, 1200), 10, {10: first}, self.narrow)
        self.drawer.draw(_blank(600, 1200), 100, {100: contact_mid_rally}, self.narrow)

        self.drawer.draw(_blank(600, 1200), 150, {150: second_bounce}, self.narrow)

        # gap from the contact (frame 100) to this bounce (frame 150) is
        # well under the threshold, so the frame-10 bounce should survive
        self.assertEqual(len(self.drawer.bounces), 2)

    def test_within_the_rally_gap_earlier_bounce_markers_survive(self):
        first = self._bounce(frame_idx=10)
        soon_after = self._bounce(frame_idx=40, x=50.0, y=50.0)  # 1s later at 30fps
        self.drawer.draw(_blank(600, 1200), 10, {10: first}, self.narrow)

        self.drawer.draw(_blank(600, 1200), 40, {40: soon_after}, self.narrow)

        self.assertEqual(len(self.drawer.bounces), 2)


class StatsPanelDrawerTest(unittest.TestCase):
    def setUp(self):
        self.narrow = scaled_calibration(scale=1.0)
        self.drawer = StatsPanelDrawer(fps=30.0, width=100)

    def test_widens_the_frame_by_its_own_width(self):
        frame = _blank(400, 640)
        result = self.drawer.draw(frame, 0)

        self.assertEqual(result.shape, (400, 740, 3))

    def test_no_rally_yet_before_any_impact(self):
        self.drawer.draw(_blank(), 0)
        self.assertEqual(self.drawer.rally_count, 0)

    def test_first_impact_starts_rally_one(self):
        self.drawer.draw(_blank(), 5, impact=_impact(5, "contact"))
        self.assertEqual(self.drawer.rally_count, 1)
        self.assertEqual(self.drawer.total_contacts, 1)
        self.assertEqual(self.drawer.current_rally_shots, 1)

    def test_a_close_impact_stays_in_the_same_rally(self):
        self.drawer.draw(_blank(), 0, impact=_impact(0, "contact"))
        self.drawer.draw(_blank(), 30, impact=_impact(30, "bounce"))  # 1s later

        self.assertEqual(self.drawer.rally_count, 1)
        self.assertEqual(self.drawer.total_bounces, 1)
        self.assertEqual(self.drawer.current_rally_shots, 1)
        self.assertEqual(self.drawer.current_rally_bounces, 1)

    def test_a_long_gap_starts_a_new_rally_and_resets_the_current_tally(self):
        self.drawer.draw(_blank(), 0, impact=_impact(0, "contact"))
        self.drawer.draw(_blank(), 30, impact=_impact(30, "bounce"))
        far_frame = int(30 + self.drawer.rally_gap_seconds * 30 * 2)  # well past the gap
        self.drawer.draw(_blank(), far_frame, impact=_impact(far_frame, "contact"))

        self.assertEqual(self.drawer.rally_count, 2)
        # totals still remember the first rally's bounce...
        self.assertEqual(self.drawer.total_bounces, 1)
        # ...but the CURRENT rally's tally starts over
        self.assertEqual(self.drawer.current_rally_shots, 1)
        self.assertEqual(self.drawer.current_rally_bounces, 0)

    def test_unknown_impacts_count_toward_the_total_but_not_shots_or_bounces(self):
        self.drawer.draw(_blank(), 0, impact=_impact(0, "unknown"))

        self.assertEqual(self.drawer.total_unattributed, 1)
        self.assertEqual(self.drawer.total_contacts, 0)
        self.assertEqual(self.drawer.total_bounces, 0)

    def test_contact_side_is_attributed_from_the_calibration_at_its_own_frame(self):
        # world (5, 2) is on the far half - see classify_court_half
        far_contact = _impact(0, "contact", x=5.0, y=2.0)
        self.drawer.draw(_blank(), 0, impact=far_contact, calibration=self.narrow)

        self.assertEqual(self.drawer.far_contacts, 1)
        self.assertEqual(self.drawer.near_contacts, 0)

    def test_no_calibration_leaves_the_side_split_at_zero(self):
        self.drawer.draw(_blank(), 0, impact=_impact(0, "contact"), calibration=None)

        self.assertEqual(self.drawer.near_contacts, 0)
        self.assertEqual(self.drawer.far_contacts, 0)

    def test_peak_speed_tracks_the_fastest_completed_shot_so_far(self):
        slower = ShotSpeed(start_frame=0, end_frame=10, peak_frame=5, peak_speed=80.0, unit="km/h")
        faster = ShotSpeed(start_frame=11, end_frame=20, peak_frame=15, peak_speed=120.0, unit="km/h")

        self.drawer.draw(_blank(), 10, completed_shot=slower)
        self.assertEqual(self.drawer.peak_speed, (80.0, "km/h"))

        self.drawer.draw(_blank(), 20, completed_shot=faster)
        self.assertEqual(self.drawer.peak_speed, (120.0, "km/h"))

    def test_a_slower_completed_shot_does_not_lower_the_running_peak(self):
        faster = ShotSpeed(start_frame=0, end_frame=10, peak_frame=5, peak_speed=120.0, unit="km/h")
        slower = ShotSpeed(start_frame=11, end_frame=20, peak_frame=15, peak_speed=80.0, unit="km/h")

        self.drawer.draw(_blank(), 10, completed_shot=faster)
        self.drawer.draw(_blank(), 20, completed_shot=slower)

        self.assertEqual(self.drawer.peak_speed, (120.0, "km/h"))

    def test_a_frame_with_nothing_new_still_draws_the_current_totals(self):
        self.drawer.draw(_blank(), 0, impact=_impact(0, "contact"))
        result = self.drawer.draw(_blank(400, 640), 1)  # no impact, no completed shot

        self.assertTrue(_painted(result) > 0)  # the panel text is still drawn
        self.assertEqual(self.drawer.total_contacts, 1)  # state carried over


if __name__ == "__main__":
    unittest.main()
