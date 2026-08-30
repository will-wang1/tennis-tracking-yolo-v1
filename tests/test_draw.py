import unittest

import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.analysis.flight_segmenter import find_flight_segments
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.tracking.ball_tracker import TrackedPosition
from src.visualize.draw import ARC_COLOR, CourtOverlayDrawer, ShotArcDrawer


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


def _impact(frame_idx, kind):
    return BounceCandidate(
        frame_idx=frame_idx, t=float(frame_idx), x=0.0, y=0.0,
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


if __name__ == "__main__":
    unittest.main()
