import importlib.util
import sys
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "build_stroke_data_e2e", Path(__file__).resolve().parent.parent / "scripts" / "build_stroke_data_e2e.py"
)
build = importlib.util.module_from_spec(_SPEC)
sys.modules["build_stroke_data_e2e"] = build
_SPEC.loader.exec_module(build)

SHAPE = (1080, 1920, 3)
PLAYER, UMPIRE, SPECTATOR = (900, 700, 980, 900), (100, 500, 160, 600), (1000, 980, 1200, 1080)


def frames(n, player_at, player_missing=()):
    """n frames of the near end: a player moving along the baseline, the
    umpire beside the net on every frame, and a spectator's head cut off by
    the bottom of the picture on every frame."""
    out = []
    for f in range(n):
        boxes, feet = [UMPIRE, SPECTATOR], [(-1.5, 13.0), (5.0, 26.0)]
        if f not in player_missing:
            boxes.append(PLAYER)
            feet.append(player_at(f))
        out.append((f, np.array(boxes, float), feet))
    return out


def test_the_player_not_the_umpire_or_a_cut_off_spectator():
    track = build.follow_player(frames(50, lambda f: (3.0 + 0.1 * f, 23.0)), far=False, frame_shape=SHAPE)
    assert len(track) == 50
    assert all(tuple(box) == PLAYER for box in track.values())


def test_a_missed_detection_leaves_a_gap_rather_than_jumping_to_someone_else():
    missing = set(range(20, 26))
    track = build.follow_player(frames(50, lambda f: (3.0 + 0.1 * f, 23.0), missing), far=False, frame_shape=SHAPE)
    assert set(track) == set(range(50)) - missing
    assert all(tuple(box) == PLAYER for box in track.values())


def test_nobody_at_the_far_end():
    assert build.follow_player(frames(10, lambda f: (5.0, 23.0)), far=True, frame_shape=SHAPE) == {}


def test_not_a_ball_kid_at_the_back_who_is_seen_more_than_the_player():
    # The ball kid stands still between the sidelines 6m behind the baseline
    # on every frame; the player is missed now and then.
    out = []
    for f in range(60):
        boxes, feet = [(1400, 300, 1440, 400)], [(5.0, 29.0)]
        if f % 10:
            boxes.append(PLAYER)
            feet.append((4.0 + 0.05 * f, 24.0))
        out.append((f, np.array(boxes, float), feet))
    track = build.follow_player(out, far=False, frame_shape=SHAPE)
    assert all(tuple(box) == PLAYER for box in track.values())


def test_a_player_lost_for_a_while_is_joined_up_again():
    # Lost for 2 seconds (longer than LOST_FRAMES), found 4m away: a new
    # person to the follower, but the same player.
    missing = set(range(20, 70))
    track = build.follow_player(frames(100, lambda f: (2.0 if f < 20 else 6.0, 23.0), missing),
                                far=False, frame_shape=SHAPE)
    assert set(track) == set(range(100)) - missing
    assert all(tuple(box) == PLAYER for box in track.values())


def test_a_close_up_is_not_joined_on():
    # The picture cuts to a close-up while the player is lost: someone huge
    # whose feet happen to map into the court.
    out = frames(100, lambda f: (2.0, 23.0), missing := set(range(20, 70)))
    for f in missing:
        out[f] = (f, np.array([UMPIRE, (600, 100, 1300, 1000)], float), [(-1.5, 13.0), (5.0, 24.0)])
    track = build.follow_player(out, far=False, frame_shape=SHAPE)
    assert set(track) == set(range(100)) - missing


def test_only_the_shot_the_strokes_are_in():
    # A close-up (bright), then the wide shot with the strokes (dark), then a replay (bright).
    rng = np.random.default_rng(0)
    def frame(level):
        return np.clip(rng.normal(level, 3, (36, 64, 3)), 0, 255).astype(np.uint8)
    images = {f: frame(200 if f < 140 or f >= 260 else 60) for f in range(100, 300)}
    assert build.stroke_shot(images, [150, 200, 250]) == (140, 259)
