"""Session structure: drills and breaks, and per-player queue and idle time.

DRILLS AND BREAKS. A drill is a stretch of play: ball-in-play runs joined
across gaps shorter than BREAK_S (a rotation, a ball fetch, a reset). A gap
of BREAK_S or more with no ball in play is a break. Camera cutaways are
not analysed but do not split a drill - play on both sides of one is the
same drill. BREAK_S is a CONVENTION: the only coaching footage so far is a
single continuous drill whose longest in-play gap is 6.6s, so it cannot
validate where "pause" ends and "break" begins. Every segment says so.

For a break, the one thing measurable without audio is whether the players
GATHERED - stood close together, not moving - which is what an
instruction huddle looks like. It is reported as "players gathered", never
as "instruction": the tool cannot hear, and a gathering can be anything.

QUEUE AND IDLE, per player, counted only while a drill is running (standing
about during a break is not waiting for a turn):
  idle   standing still for at least MIN_STILL_S in a row, anywhere
  queue  the part of idle spent OUTSIDE the court lines - behind a
         baseline or beside the court - which is where players wait to
         rotate in (seen on Dingles at 37s: one player waiting behind the
         near baseline, four lined up along the far one)
Idle includes ready position on court, which is why queue is reported
separately and is the number that means "waiting for a turn".

STANDING STILL is judged over STILL_WINDOW_S rather than frame to frame:
far-court positions jitter by tens of centimetres (one pixel is ~13cm at
the far baseline on a fence-height camera), and a one-second test reads
that jitter as walking. Under STILL_MAX_MOVE_M in STILL_WINDOW_S is still.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from src.analysis.person_tracks import COURT_LENGTH_M, COURT_WIDTH_M

BREAK_S = 15.0
MIN_STILL_S = 3.0
STILL_WINDOW_S = 2.0
STILL_MAX_MOVE_M = 0.6
OUTSIDE_MARGIN_M = 0.5
GATHERED_SPREAD_M = 4.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


@dataclass
class Segment:
    kind: str  # "drill" | "break"
    start: int
    end: int  # exclusive frame
    analysed_frames: int
    in_play_frames: int
    gathered: Optional[bool] = None  # breaks only

    def to_dict(self, fps: float) -> dict:
        out = {
            "kind": self.kind,
            "start_s": round(self.start / fps, 1),
            "end_s": round(self.end / fps, 1),
            "duration_s": round((self.end - self.start) / fps, 1),
            "analysed_s": round(self.analysed_frames / fps, 1),
            "ball_in_play_share": round(self.in_play_frames / self.analysed_frames, 3) if self.analysed_frames else None,
        }
        if self.kind == "break":
            out["players_gathered"] = self.gathered
        return out


def drill_segments(in_play: np.ndarray, analysed: np.ndarray, fps: float, break_s: float = BREAK_S) -> list[Segment]:
    """Split the timeline into drills and breaks. Gaps are measured in
    ANALYSED time, so a cutaway inside a drill neither splits it nor counts
    as a break. Time before the first and after the last play is not a
    segment: nothing says whether a drill had started."""
    n = len(in_play)
    live = in_play & analysed
    play = runs(live)
    if not play:
        return []
    segments: list[Segment] = []
    start, end = play[0]
    for a, b in play[1:]:
        gap_analysed = int(analysed[end:a].sum())
        if gap_analysed >= break_s * fps:
            segments.append(Segment("drill", start, end, int(analysed[start:end].sum()), int(live[start:end].sum())))
            segments.append(Segment("break", end, a, gap_analysed, 0))
            start = a
        end = b
    segments.append(Segment("drill", start, end, int(analysed[start:end].sum()), int(live[start:end].sum())))
    return segments


def still_mask(positions: dict[int, tuple[float, float]], fps: float, n: int) -> np.ndarray:
    """Frame-wise: was this person standing still - moved under
    STILL_MAX_MOVE_M over the STILL_WINDOW_S centred on the frame? Frames
    without a position are False (unknown is not still)."""
    out = np.zeros(n, dtype=bool)
    half = int(round(STILL_WINDOW_S * fps / 2))
    for f, (x, y) in positions.items():
        a, b = positions.get(f - half), positions.get(f + half)
        if a is None or b is None or not (0 <= f < n):
            continue
        if max(np.hypot(a[0] - x, a[1] - y), np.hypot(b[0] - x, b[1] - y)) < STILL_MAX_MOVE_M:
            out[f] = True
    return out


def outside_court(position: tuple[float, float], margin: float = OUTSIDE_MARGIN_M) -> bool:
    x, y = position
    return x < -margin or x > COURT_WIDTH_M + margin or y < -margin or y > COURT_LENGTH_M + margin


def queue_and_idle(
    positions: dict[int, tuple[float, float]],
    fps: float,
    drill_mask: np.ndarray,
    min_still_s: float = MIN_STILL_S,
) -> dict:
    """Seconds idle and queueing during drills, and the longest queue."""
    n = len(drill_mask)
    still = still_mask(positions, fps, n) & drill_mask
    def long_runs(mask):
        out = np.zeros(n, dtype=bool)
        for a, b in runs(mask):
            if b - a >= min_still_s * fps:
                out[a:b] = True
        return out

    idle = long_runs(still)
    # Queue is its own run of standing still OFF court, at least as long as
    # min_still_s - not the off-court part of an idle run, which let a 3s
    # still spell that was only partly off court count as a sub-3s "queue".
    off = np.array([f in positions and outside_court(positions[f]) for f in range(n)])
    queue = long_runs(still & off)
    queue_runs = runs(queue)
    on_camera_in_drills = sum(1 for f in positions if 0 <= f < n and drill_mask[f])
    idle_n, queue_n = int(idle.sum()), int(queue.sum())
    return {
        "idle_s": round(idle_n / fps, 1),
        "queue_s": round(queue_n / fps, 1),
        "longest_queue_s": round(max((b - a for a, b in queue_runs), default=0) / fps, 1),
        "on_camera_in_drills_s": round(on_camera_in_drills / fps, 1),
        "idle_share": round(idle_n / on_camera_in_drills, 3) if on_camera_in_drills else None,
        "queue_share": round(queue_n / on_camera_in_drills, 3) if on_camera_in_drills else None,
    }


def players_gathered(
    positions_by_player: dict[int, dict[int, tuple[float, float]]], frames: Iterable[int], fps: float
) -> Optional[bool]:
    """During a break: were the players standing close together, still?
    Judged on the median over the break of the players' median distance
    from their centroid. None when fewer than two players were seen."""
    spreads = []
    frames = list(frames)
    for f in frames[:: max(1, int(fps / 2))]:
        pts = [pos[f] for pos in positions_by_player.values() if f in pos]
        if len(pts) < 2:
            continue
        arr = np.array(pts)
        spreads.append(float(np.median(np.linalg.norm(arr - arr.mean(axis=0), axis=1))))
    if not spreads:
        return None
    return bool(np.median(spreads) < GATHERED_SPREAD_M)


# ---- court zones, in the player's OWN half -----------------------------

NET_Y_M = COURT_LENGTH_M / 2
SERVICE_LINE_FROM_NET_M = 6.4
DEEP_BEHIND_BASELINE_M = 1.5
ZONES = ("net", "baseline", "deep", "off_court")


def zone_of(position: tuple[float, float]) -> str:
    """net       inside the service line of the half they are in
    baseline  service line to DEEP_BEHIND_BASELINE_M behind the baseline -
              where rallying players normally stand
    deep      further back than that
    off_court beside the court, outside the doubles sidelines by 0.5m+"""
    x, y = position
    if x < -OUTSIDE_MARGIN_M or x > COURT_WIDTH_M + OUTSIDE_MARGIN_M:
        return "off_court"
    from_net = abs(y - NET_Y_M)
    if from_net <= SERVICE_LINE_FROM_NET_M:
        return "net"
    if from_net <= NET_Y_M + DEEP_BEHIND_BASELINE_M:
        return "baseline"
    return "deep"


def zone_shares(positions: dict[int, tuple[float, float]], drill_mask: np.ndarray) -> dict[str, float]:
    """Share of a player's drill time in each zone."""
    counts = dict.fromkeys(ZONES, 0)
    total = 0
    for f, pos in positions.items():
        if 0 <= f < len(drill_mask) and drill_mask[f]:
            counts[zone_of(pos)] += 1
            total += 1
    return {z: round(counts[z] / total, 3) if total else 0.0 for z in ZONES}


# ---- intensity over the session ----------------------------------------


def intensity_windows(
    positions_by_player: dict[int, dict[int, tuple[float, float]]],
    in_play: np.ndarray,
    drill_mask: np.ndarray,
    fps: float,
    window_s: Optional[float] = None,
) -> list[dict]:
    """Drill time cut into equal windows - thirds by default, or
    `window_s` long for a long session - each with its ball-in-play share
    and every player's metres per minute. Only drill frames count, so a
    break between drills does not read as a slump."""
    drill_frames = np.flatnonzero(drill_mask)
    if len(drill_frames) < fps * 6:
        return []
    if window_s is None:
        chunks = np.array_split(drill_frames, 3)
    else:
        size = max(1, int(window_s * fps))
        chunks = [drill_frames[i:i + size] for i in range(0, len(drill_frames), size)]
        if len(chunks) > 1 and len(chunks[-1]) < size / 2:
            chunks[-2] = np.concatenate([chunks[-2], chunks.pop()])
    windows = []
    for chunk in chunks:
        frames = set(int(f) for f in chunk)
        per_player = {}
        for player, pos in positions_by_player.items():
            seen = sorted(f for f in frames if f in pos)
            if len(seen) < fps * 2:
                continue
            dist = sum(
                float(np.hypot(pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]))
                for a, b in zip(seen, seen[1:])
                if b - a == 1
            )
            per_player[str(player)] = round(dist / (len(seen) / fps / 60.0), 1)
        windows.append({
            "start_s": round(int(chunk[0]) / fps, 1),
            "end_s": round((int(chunk[-1]) + 1) / fps, 1),
            "ball_in_play_share": round(float(in_play[chunk].mean()), 3),
            "metres_per_minute": per_player,
        })
    return windows
