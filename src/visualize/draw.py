"""Overlay tracked ball/pose/stroke/speed/bounce data on a frame.

Each drawer follows the same stateful "call once per frame" pattern:
construct once, then call `.draw(frame, ...)` every frame in a loop (see
main.py). Most drawers mutate and return the same `frame` array; the
exception is `SidebarDrawer`, which returns a new, wider array since it
changes the frame's width - call it last in the per-frame chain.
"""

from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration, FULL_COURT_REFERENCE_POINTS
from src.analysis.flight_segmenter import FlightSegment, find_segment_impacts
from src.analysis.stroke_classifier import StrokePrediction
from src.detection.pose_detector import PersonPose
from src.tracking.ball_tracker import TrackedPosition

TRAIL_COLOR_DETECTED = (0, 255, 255)  # yellow
TRAIL_COLOR_INTERPOLATED = (0, 165, 255)  # orange
ARC_COLOR = (0, 255, 255)  # yellow, same as the trail it replaces
ARC_BALL_COLOR = (255, 255, 255)  # white, so the ball reads against its own arc

POSE_KEYPOINT_COLOR = (0, 0, 255)  # red
POSE_SKELETON_COLOR = (0, 255, 0)  # green
POSE_SKELETON_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 6), (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]

BOUNCE_MARKER_COLOR = (255, 0, 255)  # magenta
SHOT_LABEL_COLOR = (128, 255, 255)  # pale yellow
CONTACT_MARKER_COLOR = (0, 165, 255)  # orange, clearly not the bounce magenta

SIDEBAR_BACKGROUND = (30, 30, 30)
SIDEBAR_TEXT_COLOR = (255, 255, 255)

COURT_LINE_COLOR = (0, 200, 255)  # orange
COURT_CORNER_COLOR = (0, 255, 255)  # yellow
COURT_LINE_EDGES = [
    ("baseline_far_left", "baseline_far_right"),
    ("baseline_near_left", "baseline_near_right"),
    ("baseline_far_left", "baseline_near_left"),
    ("baseline_far_right", "baseline_near_right"),
    ("singles_far_left", "singles_near_left"),
    ("singles_far_right", "singles_near_right"),
    ("service_far_left", "service_far_right"),
    ("service_near_left", "service_near_right"),
    ("center_service_far", "center_service_near"),
]
# The 4 doubles-court corners, as opposed to the 14 full keypoints - what a
# viewer would point to as "the corners of the court".
COURT_CORNER_NAMES = ("baseline_far_left", "baseline_far_right", "baseline_near_left", "baseline_near_right")


class TrailDrawer:
    def __init__(self, trail_length: int = 15):
        self.trail: deque[TrackedPosition] = deque(maxlen=trail_length)

    def draw(self, frame: np.ndarray, position: Optional[TrackedPosition]) -> np.ndarray:
        if position is not None:
            self.trail.append(position)

        for i, point in enumerate(self.trail):
            color = TRAIL_COLOR_INTERPOLATED if point.interpolated else TRAIL_COLOR_DETECTED
            fade = (i + 1) / max(len(self.trail), 1)
            radius = max(2, int(6 * fade))
            cv2.circle(frame, (int(point.x), int(point.y)), radius, color, -1)

        return frame



class ShotArcDrawer:
    """The whole current SHOT as fitted curves, in place of a trail of past
    positions.

    A shot is racket to racket, so it is not one parabola: the ball flies,
    lands, and flies again, and the two arcs meet at the bounce. This joins
    the flights of one shot and leaves the previous shot behind, so what is
    on screen is the path the ball has taken since somebody last hit it.

    A trail says where the ball has been, one dot per frame, and its shape
    is whatever the detector happened to output - jitter and all. The arc
    says the same thing as a curve taken from the flight the segmenter
    already fitted (`flight_segmenter.find_flight_segments`), so it is the
    smooth path the ball actually flew rather than a join-the-dots of noisy
    samples.

    NOTHING here is drawn from a raw detection, including the ball marker.
    That is the point rather than a detail: a ball in free flight follows a
    parabola, so a blob that does not sit on one was never the ball, and an
    overlay built only from fitted flights cannot show a false positive at
    all. RANSAC is what enforces it - a stray detection disagrees with the
    curve the other samples agree on, so it is excluded from the segment and
    has no say in its shape. The old trail had no such defence: every dot it
    was handed went on the screen, which is why a bad frame showed up as the
    ball teleporting into the crowd and back.

    The flip side is that the overlay is silent where the fit is: between
    two flights - the moment of a bounce or a strike - and through any
    stretch too short or too sparse to fit a flight to, there is a held arc
    but no ball marker. That is honest. Guessing a position from a curve
    that has ended would be inventing one.

    Which flights belong together is decided by what happened between them,
    and the rule is deliberately one-sided: join across an impact only when
    it was classified a BOUNCE, since that is the one event we can say left
    the ball in play on the same shot. A contact ends the shot by
    definition. So does an "unknown" - an impact nobody could attribute is
    not evidence of continuity, and drawing through one would assert
    something the classifier explicitly declined to. A boundary with no
    impact at all is a flight the segmenter split in two, so those join,
    but only across a gap of a few frames; a long unexplained hole is not
    something to draw a continuous shot through.

    The flights of one shot are drawn as ONE polyline rather than as separate
    curves, and that is what makes a shot look like a path instead of two
    arcs. A segment's samples stop wherever the detector last saw the ball,
    typically a frame or three short of the bounce, so its curve ends in
    mid-air; measured on the zverev clip consecutive flights of one shot end
    and begin 2 to 58 pixels apart. Joined into a single line, that gap is
    closed by the line itself and reads as the corner a bounce actually is.

    Extending each curve to the intersection the segmenter computes
    (`find_segment_impacts`) was the obvious fix and is worse, which is worth
    recording. That intersection is solved in y alone, so at the meeting
    instant the two parabolas can still sit tens of pixels apart in x - and
    they can sit apart in the wrong DIRECTION, the later flight extrapolating
    back to a point ahead of where the earlier one ends. The drawn path then
    doubles back on itself. Pinning both ends to the shared meeting position
    leaves a hook where the line jumps sideways; spreading that correction
    along the tail turns the hook into a zigzag and makes the two arcs cross.
    Each of those looked worse on the clip than simply connecting the flights
    where their own samples end. The curves are measurements: bending them to
    agree hides the disagreement rather than resolving it.

    Flights already completed are drawn whole. The one in progress grows to
    the current frame and no further, so the picture never shows where the
    ball is about to go. When the shot ends it stays up until the next one
    starts, which is what makes it readable in a still frame.
    """

    def __init__(
        self,
        segments: list[FlightSegment],
        impacts=(),
        samples_per_frame: float = 2.0,
        join_window: int = 6,
        max_join_gap: int = 4,
    ):
        self.segments = sorted(segments, key=lambda s: s.start_frame)
        self.samples_per_frame = samples_per_frame
        self.join_window = join_window
        self.max_join_gap = max_join_gap
        self._impact_kinds = [(impact.frame_idx, impact.kind) for impact in impacts]
        # when consecutive flights meet - used to decide what separated them,
        # not to extend either curve; see the note on extension above
        index_of = {id(segment): i for i, segment in enumerate(self.segments)}
        self._meeting = {
            index_of[id(meeting.after)]: meeting
            for meeting in find_segment_impacts(self.segments)
            if id(meeting.after) in index_of
        }
        self._shot_start = self._group_into_shots()

    def _joins_previous(self, index: int) -> bool:
        before, after = self.segments[index - 1], self.segments[index]
        meeting = self._meeting.get(index)
        at = meeting.t if meeting is not None else (before.end_frame + after.start_frame) / 2
        near = sorted(
            (abs(frame - at), kind)
            for frame, kind in self._impact_kinds
            if abs(frame - at) <= self.join_window
        )
        if near:
            return near[0][1] == "bounce"
        return after.start_frame - before.end_frame <= self.max_join_gap

    def _group_into_shots(self) -> list[int]:
        """For each flight, the index of the flight its shot began with."""
        starts = []
        for i, segment in enumerate(self.segments):
            if i and self._joins_previous(i):
                starts.append(starts[i - 1])
            else:
                starts.append(i)
        return starts

    def _arc_points(self, segment: FlightSegment, first: float, last: float) -> list:
        steps = max(int((last - first) * self.samples_per_frame), 1)
        points = []
        for i in range(steps + 1):
            t = first + (last - first) * i / steps
            x, y = segment.position(t)
            points.append((int(round(x)), int(round(y))))
        return points

    def segment_for(self, frame_idx: int) -> Optional[FlightSegment]:
        """The flight this frame is in, or the most recent one to have
        ended. Nothing before the first flight begins - there is no shot to
        show yet, and extrapolating a curve backwards out of a flight that
        has not started would draw a path the ball never took."""
        current = None
        for segment in self.segments:
            if segment.start_frame > frame_idx:
                break
            current = segment
        return current

    def draw(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        current = self.segment_for(frame_idx)
        if current is None:
            return frame

        index = self.segments.index(current)
        path = []
        for i in range(self._shot_start[index], index + 1):
            segment = self.segments[i]
            last = min(frame_idx, segment.end_frame) if i == index else segment.end_frame
            path.extend(self._arc_points(segment, float(segment.start_frame), float(last)))

        if len(path) > 1:
            cv2.polylines(
                frame, [np.array(path, dtype=np.int32)], False, ARC_COLOR, 2, cv2.LINE_AA
            )
        head = path[-1] if path else None

        if current.start_frame <= frame_idx <= current.end_frame and head is not None:
            # the ball's place on its own fitted curve, not wherever the
            # detector last pointed - so a bad frame moves nothing
            cv2.circle(frame, head, 5, ARC_BALL_COLOR, -1)
            cv2.circle(frame, head, 5, ARC_COLOR, 1, cv2.LINE_AA)
        return frame


class PoseDrawer:
    """Skeleton lines/joints for every detected player, plus a stroke label
    above the striker's head when a classifier prediction is available."""

    def draw(
        self,
        frame: np.ndarray,
        poses: list[PersonPose],
        striker: Optional[PersonPose] = None,
        stroke: Optional[StrokePrediction] = None,
    ) -> np.ndarray:
        for pose in poses:
            for a, b in POSE_SKELETON_EDGES:
                xa, ya = pose.keypoints[a]
                xb, yb = pose.keypoints[b]
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), POSE_SKELETON_COLOR, 2)
            for x, y in pose.keypoints:
                cv2.circle(frame, (int(x), int(y)), 4, POSE_KEYPOINT_COLOR, -1)

        if stroke is not None and striker is not None:
            head_x, head_y = striker.keypoints[0]  # nose
            label = f"{stroke.label.upper()} ({stroke.confidence:.0%})"
            cv2.putText(
                frame,
                label,
                (int(head_x) - 40, int(head_y) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                POSE_SKELETON_COLOR,
                2,
            )

        return frame


class BounceMarkerDrawer:
    """Every bounce detected so far stays drawn for the rest of the video -
    a persistent landing map, not a transient flash.

    A bounce is a fixed spot on the COURT, so each marker is reprojected
    fresh from its `world_x`/`world_y` through the CURRENT frame's
    `calibration` - not drawn at the pixel position it happened to occupy
    the moment it was detected. A panning/zooming broadcast camera changes
    the pixel<->world mapping frame to frame (see `CourtOverlayDrawer`,
    which reprojects the court lines the same way), and a marker held fixed
    in screen space drifts away from those lines as the camera moves -
    measured on the zverev clip, up to 94px over 744 frames for a single
    fixed court point. Falls back to the stored pixel position when no
    world coordinate or no current calibration is available.
    """

    def __init__(self):
        self.markers: list[BounceEvent] = []

    def draw(
        self, frame: np.ndarray, bounce: Optional[BounceEvent], calibration: Optional[CourtCalibration] = None
    ) -> np.ndarray:
        if bounce is not None:
            self.markers.append(bounce)

        for marker in self.markers:
            if calibration is not None and marker.world_x is not None:
                x, y = calibration.world_to_pixel(marker.world_x, marker.world_y)
            else:
                x, y = marker.x, marker.y
            cv2.drawMarker(frame, (int(x), int(y)), BOUNCE_MARKER_COLOR, cv2.MARKER_TILTED_CROSS, 16, 2)

        return frame


class ImpactMarkerDrawer:
    """Marks every impact the ball takes, bounces and racket contacts alike,
    so a run can be checked by eye rather than taken on trust.

    The two are drawn differently on purpose. A bounce is a magenta cross
    that STAYS for the rest of the video, building up the landing map. A
    contact is a transient orange circle, shown only for `hold_frames`
    around the moment it happens - there are many more of them, and leaving
    them all on screen would bury the landing map they are meant to give
    context to. Both carry their timestamp, so what's on screen can be
    matched against the impact list the run prints.

    Impacts of kind "unknown" are drawn as NEITHER. They are real kinks in
    the trajectory that the classifier could not attribute, and marking them
    as contacts by default put an orange circle on the server's ball toss
    and on a stray blob over the net. A marker is a claim; no evidence, no
    marker.

    A bounce marks a fixed spot on the COURT and stays on screen for the
    rest of the video, so it is reprojected fresh from its world position
    through each frame's own `calibration` rather than held at a fixed
    pixel - the same reasoning as `BounceMarkerDrawer`, which see. A
    contact needs none of this: it is only ever drawn within `hold_frames`
    of its own moment, looked up fresh from `impacts_by_frame` each time,
    so it is never on screen long enough for camera drift to matter.
    """

    def __init__(self, fps: float, hold_frames: int = 30):
        self.fps = fps
        self.hold_frames = hold_frames
        # world_x, world_y (None if no calibration was available at the
        # impact's own frame), x, y (pixel fallback), seconds
        self.bounces: list[tuple[Optional[float], Optional[float], float, float, float]] = []

    def draw(
        self,
        frame,
        frame_idx: int,
        impacts_by_frame: dict,
        calibration: Optional[CourtCalibration] = None,
    ) -> "np.ndarray":
        impact = impacts_by_frame.get(frame_idx)
        if impact is not None and impact.kind == "bounce":
            # `calibration` here is this call's, i.e. frame_idx's - the
            # impact's own frame, since this branch only runs the one time
            # frame_idx == impact.frame_idx
            world_x, world_y = (
                calibration.pixel_to_world(impact.x, impact.y) if calibration is not None else (None, None)
            )
            self.bounces.append((world_x, world_y, impact.x, impact.y, impact.t / self.fps))

        for world_x, world_y, x, y, seconds in self.bounces:
            if calibration is not None and world_x is not None:
                x, y = calibration.world_to_pixel(world_x, world_y)
            cv2.drawMarker(
                frame, (int(x), int(y)), BOUNCE_MARKER_COLOR, cv2.MARKER_TILTED_CROSS, 16, 2
            )
            cv2.putText(
                frame, f"BOUNCE {seconds:.2f}s", (int(x) + 12, int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOUNCE_MARKER_COLOR, 1,
            )

        # Contacts are looked up by scanning the recent window rather than
        # kept in a list, so seeking or re-running a frame can't double-count.
        for offset in range(self.hold_frames):
            recent = impacts_by_frame.get(frame_idx - offset)
            if recent is None or recent.kind != "contact":
                continue
            x, y = int(recent.x), int(recent.y)
            cv2.circle(frame, (x, y), 14, CONTACT_MARKER_COLOR, 2)
            cv2.putText(
                frame, f"CONTACT {recent.t / self.fps:.2f}s", (x + 16, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONTACT_MARKER_COLOR, 1,
            )
        return frame


class ShotLabelDrawer:
    """Draws a player's current shot label (from
    `src.analysis.shot_classifier.ShotEventTracker`, already windowed to
    stay visible a beat after the frame it was actually detected on)
    directly above their detection box - a sidebar reading is easy to miss
    while watching the rally itself, this puts it right where the eye
    already is."""

    def draw(
        self,
        frame: np.ndarray,
        bbox: Optional[tuple[float, float, float, float]],
        label: Optional[str],
    ) -> np.ndarray:
        if bbox is None or label is None:
            return frame
        x1, y1, _, _ = bbox
        cv2.putText(
            frame,
            label.upper(),
            (int(x1), max(0, int(y1) - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            SHOT_LABEL_COLOR,
            2,
        )
        return frame


class CourtOverlayDrawer:
    """Court-line wireframe + corner markers, reprojected fresh from
    whichever `CourtCalibration` is passed to `draw()` each call - a panning/
    zooming broadcast camera changes the pixel<->world mapping frame to
    frame, so the reprojection can't be cached once like a fixed-camera
    assumption would allow. `calibration=None` (the court detector lost the
    court that frame) just skips drawing for that frame."""

    def draw(self, frame: np.ndarray, calibration: Optional[CourtCalibration]) -> np.ndarray:
        if calibration is None:
            return frame

        points_px: dict[str, tuple[float, float]] = {
            name: calibration.world_to_pixel(world_x, world_y)
            for name, (world_x, world_y) in FULL_COURT_REFERENCE_POINTS.items()
        }

        for a, b in COURT_LINE_EDGES:
            xa, ya = points_px[a]
            xb, yb = points_px[b]
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), COURT_LINE_COLOR, 2)

        for name in COURT_CORNER_NAMES:
            x, y = points_px[name]
            cv2.circle(frame, (int(x), int(y)), 8, COURT_CORNER_COLOR, -1)

        return frame


class SidebarDrawer:
    """Composites a fixed-width text panel to the right of the frame.
    Returns a NEW, wider array - unlike the other drawers here, it doesn't
    mutate in place. Call this LAST in the per-frame chain."""

    def __init__(self, width: int = 250):
        self.width = width

    def draw(
        self,
        frame: np.ndarray,
        stroke_label: Optional[str],
        speed: Optional[tuple[float, str]],
    ) -> np.ndarray:
        """`speed` is (value, unit) - e.g. (42.3, "km/h") - the current
        LIVE instantaneous ball speed this frame, not a shot summary, so it
        updates continuously rather than only within bounce-segmented shots.
        `stroke_label` is the display label from a
        `src.analysis.shot_classifier.ShotEventTracker` (already windowed
        to stay visible a beat after the frame it was actually detected
        on), or None."""
        height = frame.shape[0]
        sidebar = np.full((height, self.width, 3), SIDEBAR_BACKGROUND, dtype=np.uint8)

        lines = ["Stroke:", stroke_label.upper() if stroke_label else "-", "", "Speed:"]
        if speed is not None:
            value, unit = speed
            lines.append(f"{value:.0f} {unit}")
        else:
            lines.append("-")

        for i, line in enumerate(lines):
            cv2.putText(
                sidebar,
                line,
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                SIDEBAR_TEXT_COLOR,
                1,
            )

        return np.hstack([frame, sidebar])
