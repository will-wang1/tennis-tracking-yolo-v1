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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import json

from src.analysis.bounce_detector import BounceEvent
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.analysis.speed_estimator import ShotSpeed

DEFAULT_RALLY_GAP_SECONDS = 4.0


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
class MatchStats:
    rallies: list[RallyStats]
    total_bounces: int
    total_contacts: int
    total_unattributed: int
    shot_speeds: list[ShotSpeed] = field(default_factory=list)
    # world (x, y) metres, court-relative - one per bounce with a calibration
    bounce_locations: list[tuple[float, float]] = field(default_factory=list)
    near_shot_counts: Optional[dict[str, int]] = None
    far_shot_counts: Optional[dict[str, int]] = None

    def to_dict(self) -> dict:
        return {
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
            "near_shot_counts": self.near_shot_counts,
            "far_shot_counts": self.far_shot_counts,
        }

    def write_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


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
    rally_gap_seconds: float = DEFAULT_RALLY_GAP_SECONDS,
) -> MatchStats:
    """`impacts` is `ImpactAnalysis.impacts` (bounce/contact/unknown already
    attributed), `shots` is speed_estimator's per-shot peak readings, and
    `bounces` is the same list `main.py` draws markers from - see this
    module's docstring for why a rally boundary is a gap between impacts
    and nothing more.
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

    return MatchStats(
        rallies=rallies,
        total_bounces=sum(1 for i in impacts if i.kind == "bounce"),
        total_contacts=sum(1 for i in impacts if i.kind == "contact"),
        total_unattributed=sum(1 for i in impacts if i.kind == "unknown"),
        shot_speeds=list(shots),
        bounce_locations=[
            (b.world_x, b.world_y) for b in bounces if b.world_x is not None and b.world_y is not None
        ],
        near_shot_counts=dict(near_shot_counts) if near_shot_counts is not None else None,
        far_shot_counts=dict(far_shot_counts) if far_shot_counts is not None else None,
    )
