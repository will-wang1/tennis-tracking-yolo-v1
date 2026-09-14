"""Fold one clip's already-computed impacts, shot speeds, bounce locations
and stroke counts into a single match-level summary.

Nothing here detects anything new - every number is a fold over data
`main.py`'s pipeline (BallTracker, touchdown_detector, speed_estimator,
shot_classifier) already produced. That matters for the one genuinely new
piece of logic this module adds, RALLY SEGMENTATION: there is no direct
"ball in/out of play" signal anywhere upstream, only the impacts themselves,
so a rally boundary here is inferred purely from the GAP between consecutive
impacts (bounce, contact, or unknown - all three are real events, even an
unattributed one) being unusually large.

`DEFAULT_RALLY_GAP_SECONDS` is picked from the longest in-rally gap actually
measured across every clip this project has looked at closely: 2.12s, on
the zverev clip's net-volley stretch around 17.97s-20.09s (three impacts in
quick succession with real spacing between them). Set comfortably above
that and comfortably below a real between-point pause (players resetting,
a new serve - several seconds in practice), but HONESTLY UNTESTED against
any footage that actually contains more than one rally, since none of this
project's clips do. Treat the resulting rally count as a reasonable guess
on new footage, not a validated measurement, the same way the rest of this
project treats an unlabelled clip's bounce/contact verdicts.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import json

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration
from src.analysis.court_zones import classify_court_half, classify_landing_zone
from src.analysis.match_log import MatchLog
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.speed_estimator import ShotSpeed

DEFAULT_RALLY_GAP_SECONDS = 4.0

# A sprinting elite tennis player tops out around 8-9 m/s over short bursts.
# Set well above that (rather than at it) because this guards against
# DETECTOR jumps, not real player speed - a missed/wrong player box for one
# frame (a line judge, a ball kid, a stray Faster R-CNN false positive) can
# move the "player" position several metres in a single frame with nothing
# physical behind it, and a threshold too close to real sprint speed would
# also reject a genuine hard sprint. Frame pairs that imply a faster jump
# than this are excluded from the distance/speed fold entirely, the same
# outlier-rejection reasoning `BallTracker` applies to the ball.
DEFAULT_MAX_PLAYER_SPEED_MPS = 12.0


@dataclass(frozen=True)
class RallyStats:
    start_frame: int
    end_frame: int
    duration_s: float
    shot_count: int  # contacts - times a player struck the ball, serve included
    bounce_count: int
    unattributed_count: int
    peak_speed: Optional[float] = None
    peak_speed_unit: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_s": round(self.duration_s, 2),
            "shot_count": self.shot_count,
            "bounce_count": self.bounce_count,
            "unattributed_count": self.unattributed_count,
            "peak_speed": round(self.peak_speed, 1) if self.peak_speed is not None else None,
            "peak_speed_unit": self.peak_speed_unit,
        }


@dataclass(frozen=True)
class PlayerMovementStats:
    distance_m: float
    average_speed_mps: float
    max_speed_mps: float
    tracked_frames: int  # frame-to-frame steps actually used (after outlier rejection)

    def to_dict(self) -> dict:
        return {
            "distance_m": round(self.distance_m, 1),
            "average_speed_mps": round(self.average_speed_mps, 2),
            "max_speed_mps": round(self.max_speed_mps, 2),
            "tracked_frames": self.tracked_frames,
        }


@dataclass(frozen=True)
class ServeSpeedReading:
    frame_idx: int
    t_s: float
    peak_speed: Optional[float]
    unit: Optional[str]

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "t_s": round(self.t_s, 2),
            "peak_speed": round(self.peak_speed, 1) if self.peak_speed is not None else None,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class MatchStats:
    fps: float
    rallies: list[RallyStats]
    total_bounces: int
    total_contacts: int
    total_unattributed: int
    shot_speeds: list[ShotSpeed] = field(default_factory=list)
    # world (x, y) metres, court-relative - one per bounce with a calibration
    bounce_locations: list[tuple[float, float]] = field(default_factory=list)
    # one court_zones.LandingZone.label() per bounce, parallel to bounce_locations
    bounce_zone_labels: list[str] = field(default_factory=list)
    near_shot_counts: Optional[dict[str, int]] = None
    far_shot_counts: Optional[dict[str, int]] = None
    near_player_movement: Optional[PlayerMovementStats] = None
    far_player_movement: Optional[PlayerMovementStats] = None
    near_serve_speeds: list[ServeSpeedReading] = field(default_factory=list)
    far_serve_speeds: list[ServeSpeedReading] = field(default_factory=list)
    # Frame indices where the camera itself changed (see scene_cuts.py) -
    # video-level metadata, not folded from impacts like everything else
    # here, but a downstream reader (an AI agent especially) needs it to
    # avoid treating a mid-point closeup cutaway as evidence of anything
    # about the RALLY, and to not trust tracking/calibration continuity
    # across the seam.
    scene_cuts: list[int] = field(default_factory=list)
    # Serve/point structure (see serve_sequences.py, match_log.py) - who
    # served each point, faults, double faults, first-serve-in rate, rally
    # length. Deliberately carries NO point winner and no score - see
    # match_log.py's module docstring for why that's a scope decision, not
    # a gap. None when the caller didn't have what it takes to build one
    # (needs a court calibration and player identity tracking, at minimum).
    match_log: Optional[MatchLog] = None
    # world (x, y) metres and a "near"/"far" half label, one pair per racket
    # CONTACT with a calibration - see attribute_contacts_to_court_half.
    # Unlike near_shot_counts/far_shot_counts (pose-classified, near player
    # only - see shot_classifier.py's module docstring), this covers BOTH
    # players and needs no pose model at all, at the cost of shot TYPE:
    # it can say a shot happened and which player hit it, not forehand vs.
    # backhand vs. serve.
    contact_locations: list[tuple[float, float]] = field(default_factory=list)
    contact_sides: list[str] = field(default_factory=list)

    @property
    def bounce_zone_counts(self) -> dict[str, int]:
        return dict(Counter(self.bounce_zone_labels))

    @property
    def contact_side_counts(self) -> dict[str, int]:
        return dict(Counter(self.contact_sides))

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "rally_count": len(self.rallies),
            "rallies": [r.to_dict() for r in self.rallies],
            "total_bounces": self.total_bounces,
            "total_contacts": self.total_contacts,
            "total_unattributed": self.total_unattributed,
            "shot_speeds": [
                {
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "peak_speed": round(s.peak_speed, 1),
                    "unit": s.unit,
                    "method": s.method,
                }
                for s in self.shot_speeds
            ],
            "bounce_locations": [[round(x, 2), round(y, 2)] for x, y in self.bounce_locations],
            "bounce_zone_labels": self.bounce_zone_labels,
            "bounce_zone_counts": self.bounce_zone_counts,
            "near_shot_counts": self.near_shot_counts,
            "far_shot_counts": self.far_shot_counts,
            "near_player_movement": self.near_player_movement.to_dict()
            if self.near_player_movement is not None
            else None,
            "far_player_movement": self.far_player_movement.to_dict()
            if self.far_player_movement is not None
            else None,
            "near_serve_speeds": [s.to_dict() for s in self.near_serve_speeds],
            "far_serve_speeds": [s.to_dict() for s in self.far_serve_speeds],
            "contact_locations": [[round(x, 2), round(y, 2)] for x, y in self.contact_locations],
            "contact_sides": self.contact_sides,
            "contact_side_counts": self.contact_side_counts,
            "scene_cuts": self.scene_cuts,
            "match_log": self.match_log.to_dict() if self.match_log is not None else None,
        }

    def write_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def compute_player_movement(
    world_positions_by_frame: dict[int, tuple[float, float]],
    fps: float,
    max_plausible_speed_mps: float = DEFAULT_MAX_PLAYER_SPEED_MPS,
) -> Optional[PlayerMovementStats]:
    """Distance covered and speed from a player's per-frame court position.

    Only consecutive tracked frame PAIRS are folded in, and only when the
    implied speed is physically plausible - see `DEFAULT_MAX_PLAYER_SPEED_MPS`
    for why. A gap (an occluded or undetected frame) contributes nothing on
    either side of it rather than being bridged, the same call `BallTracker`
    makes for a gap too long to interpolate: a straight line across a real
    gap is a guess about where the player was, not a measurement.

    Returns None if there's nothing to measure (fewer than two tracked
    frames), so a caller can tell "no player data" apart from "stood still".
    """
    frames = sorted(world_positions_by_frame)
    if len(frames) < 2:
        return None

    total_distance = 0.0
    max_speed = 0.0
    used_steps = 0
    for a, b in zip(frames, frames[1:]):
        elapsed = (b - a) / fps
        (ax, ay), (bx, by) = world_positions_by_frame[a], world_positions_by_frame[b]
        distance = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        speed = distance / elapsed
        if speed > max_plausible_speed_mps:
            continue
        total_distance += distance
        max_speed = max(max_speed, speed)
        used_steps += 1

    if used_steps == 0:
        return None

    total_time_s = (frames[-1] - frames[0]) / fps
    average_speed = total_distance / total_time_s if total_time_s > 0 else 0.0
    return PlayerMovementStats(
        distance_m=total_distance,
        average_speed_mps=average_speed,
        max_speed_mps=max_speed,
        tracked_frames=used_steps,
    )


def compute_serve_speed_trend(
    serve_events: Sequence[tuple[int, str]],
    shots: Sequence[ShotSpeed],
    fps: float,
) -> list[ServeSpeedReading]:
    """One reading per serve event (from `ShotEventTracker.events`, already
    filtered to label == "serve" by the caller - or pass the raw list and
    this filters it itself), in chronological order, pairing each serve's
    frame with whichever tracked shot speed overlaps it. `peak_speed` is
    None when no shot speed was tracked at that instant (e.g. --speed was
    off, or the ball wasn't tracked right at the serve) - a missing
    reading, not a zero.
    """
    readings = []
    for frame_idx, label in serve_events:
        if label != "serve":
            continue
        overlapping = [s for s in shots if s.start_frame <= frame_idx <= s.end_frame]
        speed = max(overlapping, key=lambda s: s.peak_speed) if overlapping else None
        readings.append(
            ServeSpeedReading(
                frame_idx=frame_idx,
                t_s=frame_idx / fps,
                peak_speed=speed.peak_speed if speed is not None else None,
                unit=speed.unit if speed is not None else None,
            )
        )
    return readings


def attribute_contacts_to_court_half(
    contacts: Sequence[BounceCandidate],
    calibrations_by_frame: dict[int, CourtCalibration],
) -> list[tuple[tuple[float, float], str]]:
    """Which player hit each racket contact, from where the racket met the
    ball rather than from a pose model - so unlike shot_classifier.py this
    works identically for the near AND far player. A contact's own (x, y)
    IS where a player was standing: nobody can legally hit a ball sitting
    on the far side of the net, so the projected court position at the
    instant of contact already answers the question directly, the same way
    `player_reach_ratio` already leans on player-proximity-to-impact
    elsewhere in this pipeline (touchdown_detector.py).

    One (world_xy, half) pair per contact that HAS a calibration for its
    frame, in the same order as `contacts` - contacts with no calibration
    are silently skipped rather than guessed at.

    HONEST CAVEAT: a contact is, by definition, airborne (racket height,
    not ground level), and the ground-plane homography used to project it
    is only exact for a point ON the court - see
    parabolic_bounce_detector.py's module docstring for the mechanism. The
    error pushes an elevated point's apparent depth AWAY from the camera,
    which is why a far-court contact's projected world_y regularly comes
    out negative (beyond the true far baseline) rather than landing exactly
    where the player's feet were. That inflation makes a contact look
    FARTHER from the net than it really is, never closer, so it cannot by
    itself manufacture a wrong-side attribution - the one case this could
    still misattribute is a contact struck very close to the net itself
    (e.g. a volley), where a real position a few tens of centimetres on one
    side could in principle project across the net line. Sanity-checked
    against both labelled clips' cached impacts: on zverev_rally all 17
    contacts alternate near/far/near/far... perfectly, matching a rally
    where players return each other's shots in turn; on video_input2 most
    do too, and every place they don't (1.96s, 12.34s) is a contact the
    touchdown_detector module docstring already documents as a genuinely
    misclassified bounce, not a new failure this introduces.
    """
    results = []
    for contact in contacts:
        calibration = calibrations_by_frame.get(contact.frame_idx)
        if calibration is None:
            continue
        world_xy = calibration.pixel_to_world(contact.x, contact.y)
        results.append((world_xy, classify_court_half(world_xy[1])))
    return results


def _shot_speed_overlapping(
    shots: Sequence[ShotSpeed], start_frame: int, end_frame: int
) -> Optional[ShotSpeed]:
    overlapping = [s for s in shots if s.start_frame <= end_frame and s.end_frame >= start_frame]
    if not overlapping:
        return None
    return max(overlapping, key=lambda s: s.peak_speed)


def compute_match_stats(
    impacts: Sequence[BounceCandidate],
    shots: Sequence[ShotSpeed],
    bounces: Sequence[BounceEvent],
    fps: float,
    near_shot_counts: Optional[dict[str, int]] = None,
    far_shot_counts: Optional[dict[str, int]] = None,
    near_player_positions_by_frame: Optional[dict[int, tuple[float, float]]] = None,
    far_player_positions_by_frame: Optional[dict[int, tuple[float, float]]] = None,
    near_shot_events: Optional[Sequence[tuple[int, str]]] = None,
    far_shot_events: Optional[Sequence[tuple[int, str]]] = None,
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
    scene_cuts: Optional[Sequence[int]] = None,
    match_log: Optional[MatchLog] = None,
    rally_gap_seconds: float = DEFAULT_RALLY_GAP_SECONDS,
) -> MatchStats:
    """`impacts` is `ImpactAnalysis.impacts` (bounce/contact/unknown already
    attributed), `shots` is speed_estimator's per-shot peak readings, and
    `bounces` is the same list `main.py` draws markers from - see this
    module's docstring for why a rally boundary is a gap between impacts
    and nothing more.

    `near_player_positions_by_frame`/`far_player_positions_by_frame` are one
    world (x, y) point per frame - `main.py` already collects a LIST per
    frame (`PlayerDetector.split_top_bottom` can return more than one box,
    e.g. a stray line-judge detection); pass the primary player's point per
    frame, chosen however the caller trusts most (this module doesn't
    re-derive which detection is the real player).

    `near_shot_events`/`far_shot_events` are `ShotEventTracker.events`
    (`(frame_idx, label)` for every counted shot) - only used to find serve
    events, so passing the whole list is fine.

    `calibrations_by_frame` (the same map `main.py` builds under
    --show-court) drives `contact_locations`/`contact_sides` - see
    `attribute_contacts_to_court_half`. Without it those come back empty,
    same as any other calibration-dependent field here.

    `match_log` is a pre-built `match_log.MatchLog` (from
    `serve_sequences.classify_serve_sequences` + `match_log.build_match_log`,
    which need a court calibration and `player_identity.PlayerIdentityTracker`
    output that this function doesn't have inputs to build itself) - passed
    straight through, not computed here.
    """
    ordered = sorted(impacts, key=lambda impact: impact.t)

    groups: list[list[BounceCandidate]] = []
    for impact in ordered:
        if groups and (impact.t - groups[-1][-1].t) / fps > rally_gap_seconds:
            groups.append([])
        elif not groups:
            groups.append([])
        groups[-1].append(impact)

    rallies = []
    for group in groups:
        start_frame = group[0].frame_idx
        end_frame = group[-1].frame_idx
        duration_s = (group[-1].t - group[0].t) / fps
        shot_count = sum(1 for i in group if i.kind == "contact")
        bounce_count = sum(1 for i in group if i.kind == "bounce")
        unattributed_count = sum(1 for i in group if i.kind == "unknown")
        peak = _shot_speed_overlapping(shots, start_frame, end_frame)
        rallies.append(
            RallyStats(
                start_frame=start_frame,
                end_frame=end_frame,
                duration_s=duration_s,
                shot_count=shot_count,
                bounce_count=bounce_count,
                unattributed_count=unattributed_count,
                peak_speed=peak.peak_speed if peak is not None else None,
                peak_speed_unit=peak.unit if peak is not None else None,
            )
        )

    bounce_locations = [
        (b.world_x, b.world_y) for b in bounces if b.world_x is not None and b.world_y is not None
    ]

    contact_locations: list[tuple[float, float]] = []
    contact_sides: list[str] = []
    if calibrations_by_frame:
        contacts = [impact for impact in impacts if impact.kind == "contact"]
        for world_xy, half in attribute_contacts_to_court_half(contacts, calibrations_by_frame):
            contact_locations.append(world_xy)
            contact_sides.append(half)

    return MatchStats(
        fps=fps,
        rallies=rallies,
        total_bounces=sum(1 for i in impacts if i.kind == "bounce"),
        total_contacts=sum(1 for i in impacts if i.kind == "contact"),
        total_unattributed=sum(1 for i in impacts if i.kind == "unknown"),
        shot_speeds=list(shots),
        bounce_locations=bounce_locations,
        bounce_zone_labels=[classify_landing_zone(x, y).label() for x, y in bounce_locations],
        near_shot_counts=dict(near_shot_counts) if near_shot_counts is not None else None,
        far_shot_counts=dict(far_shot_counts) if far_shot_counts is not None else None,
        near_player_movement=compute_player_movement(near_player_positions_by_frame, fps)
        if near_player_positions_by_frame
        else None,
        far_player_movement=compute_player_movement(far_player_positions_by_frame, fps)
        if far_player_positions_by_frame
        else None,
        near_serve_speeds=compute_serve_speed_trend(near_shot_events, shots, fps)
        if near_shot_events
        else [],
        far_serve_speeds=compute_serve_speed_trend(far_shot_events, shots, fps)
        if far_shot_events
        else [],
        contact_locations=contact_locations,
        contact_sides=contact_sides,
        scene_cuts=list(scene_cuts) if scene_cuts else [],
        match_log=match_log,
    )
