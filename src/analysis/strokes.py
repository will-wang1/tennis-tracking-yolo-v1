"""Find every person's swings and name each stroke: serve, overhead,
forehand, backhand, forehand volley, backhand volley - and, for
groundstrokes hit high to low, slice.

RULES, NOT A TRAINED MODEL. No labelled coaching strokes exist yet, and no
pretrained model in this project covers these classes (the pretrained GRU
knows forehand/backhand/serve on near players only). So each call below is
a stated geometric rule over pose, racket and court position - readable,
testable, and wrong in knowable ways - until labels exist to measure it or
train something better. Every stroke record carries the measurements its
call was made from, so a disputed call can be checked rather than argued.

WHAT A SWING IS. A peak in wrist speed, measured in torso-lengths per
second relative to the hips, so a far player (~90px tall) and a near one
(~280px) are on the same scale and running does not count. Measured on the
two near players of dingles_serve_volley: typical wrist speed ~2 torso/s,
peaks at shots 8-24 torso/s. A swing is CONFIRMED as a hit when the ball
is within reach (0.6 box heights, the classifier's own reach test) within
0.32s of the peak; otherwise it is kept and flagged - shadow swings and
feeds happen constantly in coaching and are not errors to throw away.

THE CALLS, in order:
  serve / overhead   racket wrist above the head (above the nose by half a
                     torso) around the peak; serve at or behind a
                     baseline, overhead anywhere else
  volley             player within VOLLEY_NET_DISTANCE_M of the net - a
                     position proxy for "hit before it bounced"
  forehand/backhand  groundstrokes: the DIRECTION the racket wrist sweeps
                     through contact (a forehand crosses away from the
                     racket side, a backhand toward it); volleys: which side
                     of the body contact is on, relative to where that
                     player's racket hand rests. Both in the PLAYER'S own
                     frame: a near-court player faces away from the camera
                     (their right is image right), a far-court one faces it
  slice              a groundstroke whose racket wrist travels high to low
                     through contact. The weakest call here: slice is
                     mostly about the racket face, which pose cannot see.

KNOWN FALSE STROKES: a player bending to pick up a ball, or bouncing it
before a serve, moves the wrist fast next to a ball, so both can pass as a
confirmed hit. Nothing here separates them yet.

HANDEDNESS comes from the racket, not the wrists: two-handed backhands and
body rotation make both wrists peak together, so "the faster wrist" is not
the racket hand. Where a racket was detected at a person's swings, the
wrist nearest it is their racket hand; a person with no racket seen falls
back to the faster wrist, and says so.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

# COCO keypoints
NOSE, L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 5, 6, 9, 10, 11, 12

SWING_PEAK_TORSO_PER_S = 8.0
SWING_MIN_SPACING_S = 0.6
HIT_REACH_BOX_HEIGHTS = 0.6
HIT_WINDOW_S = 0.32
ABOVE_HEAD_TORSO = 0.5
# Volley zone, from the data: on dingles_serve_volley the far players'
# confirmed hits cluster 4-8m from the net (27) and 10-15m (34), with one
# hit in 8-10m - the volleyers stand around the service line (6.4m), not at
# the net, and a first guess of 4.6m cut that cluster in half (3 volleys
# in a serve-and-volley drill). 8m sits in the gap. POSITION IS A PROXY:
# the real test is "hit before it bounced", which needs bounce detection;
# a mid-court shot taken after a bounce inside 8m will read as a volley.
VOLLEY_NET_DISTANCE_M = 8.0
BASELINE_ZONE_M = 1.5
SLICE_DROP_TORSO = 0.25
MIN_KEYPOINT_CONF = 0.3

NET_Y_M = 11.885
COURT_LENGTH_M = 23.77


@dataclass
class BodyFrame:
    """One person's pose in one frame, normalised: origin at mid-hip,
    unit = torso length, image axes (y down)."""

    frame: int
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    nose: np.ndarray
    shoulder_mid: np.ndarray
    torso_px: float
    hip_px: np.ndarray
    wrist_conf: tuple[float, float]


def body_frames(poses_by_frame: list[dict], track_id: int) -> dict[int, BodyFrame]:
    out = {}
    for f, frame_poses in enumerate(poses_by_frame):
        if track_id not in frame_poses:
            continue
        xy, conf = frame_poses[track_id]
        if min(conf[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]]) < MIN_KEYPOINT_CONF:
            continue
        shoulder = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2.0
        hip = (xy[L_HIP] + xy[R_HIP]) / 2.0
        torso = float(np.linalg.norm(shoulder - hip))
        if torso < 5.0:
            continue
        norm = lambda p: (p - hip) / torso  # noqa: E731
        out[f] = BodyFrame(
            frame=f,
            left_wrist=norm(xy[L_WRIST]),
            right_wrist=norm(xy[R_WRIST]),
            nose=norm(xy[NOSE]),
            shoulder_mid=norm(shoulder),
            torso_px=torso,
            hip_px=hip,
            wrist_conf=(float(conf[L_WRIST]), float(conf[R_WRIST])),
        )
    return out


def swing_peaks(frames: dict[int, BodyFrame], fps: float) -> list[tuple[int, float]]:
    """(frame, peak speed) of every swing: peaks of the faster wrist's
    speed, lightly smoothed, above SWING_PEAK_TORSO_PER_S and at least
    SWING_MIN_SPACING_S apart. Frame pairs across a gap are not measured."""
    fs = sorted(frames)
    times, speeds = [], []
    for a, b in zip(fs, fs[1:]):
        if b - a != 1:
            continue
        fa, fb = frames[a], frames[b]
        v = max(
            float(np.linalg.norm(fb.left_wrist - fa.left_wrist)),
            float(np.linalg.norm(fb.right_wrist - fa.right_wrist)),
        ) * fps
        times.append(b)
        speeds.append(v)
    if len(speeds) < 5:
        return []
    smooth = np.convolve(np.array(speeds), np.ones(3) / 3.0, mode="same")
    peaks, _ = find_peaks(smooth, height=SWING_PEAK_TORSO_PER_S, distance=max(1, int(SWING_MIN_SPACING_S * fps)))
    return [(int(times[i]), float(smooth[i])) for i in peaks]


def _window(frames: dict[int, BodyFrame], start: int, end: int) -> list[BodyFrame]:
    return [frames[f] for f in range(start, end + 1) if f in frames]


@dataclass
class Stroke:
    track_id: int
    frame: int
    t_s: float
    stroke: str  # serve | overhead | forehand | backhand | forehand_volley | backhand_volley | slice
    side: Optional[str]  # forehand | backhand | None (serve/overhead)
    hit_confirmed: bool
    peak_speed: float
    court_position_m: Optional[tuple[float, float]]
    racket_hand: str  # left | right
    racket_hand_source: str  # racket | wrist_speed
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "t_s": round(self.t_s, 2),
            "frame": self.frame,
            "stroke": self.stroke,
            "side": self.side,
            "hit_confirmed": self.hit_confirmed,
            "peak_speed_torso_per_s": round(self.peak_speed, 1),
            "court_position_m": None if self.court_position_m is None else [round(v, 2) for v in self.court_position_m],
            "racket_hand": self.racket_hand,
            "racket_hand_source": self.racket_hand_source,
            "evidence": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.evidence.items()},
        }


def resting_lateral(frames: dict[int, BodyFrame], racket_hand: str) -> float:
    """Where this person's racket wrist sits across their body at rest, in
    image-x torso units relative to the shoulder midpoint: the median over
    every frame they were posed in, which is mostly ready position.

    Needed because a player holding a racket keeps the racket hand on its
    own side of the body even at rest. A groundstroke backhand swings the
    wrist far across and reads correctly against the midline; a backhand
    VOLLEY moves it only a little across - still the "forehand" side of the
    midline. Measured on Dingles before this correction: 26 forehand
    volleys to 3 backhand, with the volleys' side measure a median +0.44
    torso onto the forehand side. So the side is judged by which way the
    wrist moves away from this resting offset."""
    wrist = "right_wrist" if racket_hand == "right" else "left_wrist"
    values = [getattr(b, wrist)[0] - b.shoulder_mid[0] for b in frames.values()]
    return float(np.median(values)) if values else 0.0


def classify_swing(
    frames: dict[int, BodyFrame],
    peak_frame: int,
    fps: float,
    racket_hand: str,
    court_position_m: Optional[tuple[float, float]],
    rest_lateral: float = 0.0,
) -> tuple[str, Optional[str], dict]:
    """(stroke, side, evidence) for one swing - see the module docstring
    for each rule."""
    wrist = "right_wrist" if racket_hand == "right" else "left_wrist"
    pre = _window(frames, peak_frame - int(0.4 * fps), peak_frame - int(0.08 * fps))
    # Serve evidence is taken BEFORE contact only: a topspin groundstroke
    # finishes with the racket over the head, and a window running past the
    # peak called a clear groundstroke (P1, 9.72s on Dingles) a serve.
    around = _window(frames, peak_frame - int(0.3 * fps), peak_frame)
    through_a = frames.get(peak_frame - int(round(0.12 * fps)))
    through_b = frames.get(peak_frame + int(round(0.12 * fps)))
    after = _window(frames, peak_frame + int(0.04 * fps), peak_frame + int(0.16 * fps))
    before = _window(frames, peak_frame - int(0.24 * fps), peak_frame - int(0.12 * fps))
    evidence: dict = {}

    # (A "pickup" guard - racket hand below the knees just after the peak -
    # was tried for P1 at 9.72s on Dingles, a player bending to collect a
    # ball. It missed that case and relabelled a genuine low volley (P123,
    # 64.56s) as a pickup, so it was removed: low shots and pickups are not
    # separable by wrist height alone. Pickups remain a known false
    # "stroke"; see the module docstring.)

    # serve / overhead: racket wrist above the head around the peak
    height_above_nose = max((b.nose[1] - getattr(b, wrist)[1] for b in around), default=0.0)
    evidence["wrist_above_nose_torso"] = float(height_above_nose)
    if height_above_nose > ABOVE_HEAD_TORSO:
        at_baseline = court_position_m is not None and (
            court_position_m[1] < BASELINE_ZONE_M or court_position_m[1] > COURT_LENGTH_M - BASELINE_ZONE_M
        )
        evidence["at_baseline"] = bool(at_baseline)
        return ("serve" if at_baseline else "overhead"), None, evidence

    # Player frame: a near-court player faces away from the camera (their
    # right is image right), a far-court one faces it.
    facing_away = court_position_m is None or court_position_m[1] >= NET_Y_M
    player_right = 1.0 if facing_away else -1.0  # sign of image-x that is the player's right
    hand_sign = 1.0 if racket_hand == "right" else -1.0
    evidence["facing"] = "away" if facing_away else "camera"
    # +: toward the racket-hand side of the body, in torso units
    to_racket_side = lambda dx: float(dx) * player_right * hand_sign  # noqa: E731

    # Contact side relative to where this player's racket hand rests - the
    # cue for VOLLEYS, which barely move sideways (see resting_lateral).
    contact_side = (
        to_racket_side(np.mean([getattr(b, wrist)[0] - b.shoulder_mid[0] for b in pre]) - rest_lateral) if pre else 0.0
    )
    evidence["contact_side_vs_rest_torso"] = contact_side

    # volley: near the net
    if court_position_m is not None:
        distance_to_net = abs(court_position_m[1] - NET_Y_M)
        evidence["distance_to_net_m"] = float(distance_to_net)
        if distance_to_net < VOLLEY_NET_DISTANCE_M:
            side = "forehand" if contact_side >= 0 else "backhand"
            return f"{side}_volley", side, evidence

    # Groundstroke side from the DIRECTION of the swing through contact: a
    # forehand carries the racket wrist from its own side across the body,
    # a backhand from the far side toward its own. Direction, not position:
    # judging by where the wrist sits called a left-hander's forehand (P1,
    # 8.84s) a backhand, because his resting hand sits far out to his side.
    if through_a is not None and through_b is not None:
        sweep = to_racket_side(getattr(through_b, wrist)[0] - getattr(through_a, wrist)[0])
    else:
        sweep = -contact_side  # no frames either side of the peak: fall back to position
    evidence["sweep_toward_racket_side_torso"] = sweep
    side = "backhand" if sweep > 0 else "forehand"

    # slice: groundstroke wrist travelling high to low through contact
    if before and after:
        drop = float(np.mean([getattr(b, wrist)[1] for b in after]) - np.mean([getattr(b, wrist)[1] for b in before]))
        evidence["wrist_drop_torso"] = drop  # image y down: positive = wrist moved DOWN
        if drop > SLICE_DROP_TORSO:
            return "slice", side, evidence
    return side, side, evidence


def racket_hand_from_wrist_speed(frames: dict[int, BodyFrame], peaks: list[tuple[int, float]], fps: float) -> str:
    """Fallback when no racket was seen: the wrist that moves faster at the
    person's swing peaks."""
    left = right = 0.0
    for f, _ in peaks:
        a, b = frames.get(f - 1), frames.get(f)
        if a is None or b is None:
            continue
        left += float(np.linalg.norm(b.left_wrist - a.left_wrist))
        right += float(np.linalg.norm(b.right_wrist - a.right_wrist))
    return "left" if left > right else "right"


def ball_within_reach(ball_by_frame, box_by_frame: dict[int, tuple], frame: int, fps: float) -> bool:
    """Was a tracked ball within HIT_REACH_BOX_HEIGHTS of this person's box
    within HIT_WINDOW_S of `frame`?"""
    window = int(round(HIT_WINDOW_S * fps))
    for g in range(frame - window, frame + window + 1):
        ball = ball_by_frame[g] if 0 <= g < len(ball_by_frame) else None
        box = box_by_frame.get(g)
        if ball is None or box is None:
            continue
        x1, y1, x2, y2 = box
        dx = max(x1 - ball.x, 0.0, ball.x - x2)
        dy = max(y1 - ball.y, 0.0, ball.y - y2)
        if float(np.hypot(dx, dy)) / max(y2 - y1, 1.0) <= HIT_REACH_BOX_HEIGHTS:
            return True
    return False
