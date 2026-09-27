"""Every swing of every player, from wrist movement (src/analysis/swings.py).

Needs a pose on EVERY frame: run scripts/track_poses.py with --every 1.

    python scripts/detect_swings.py \\
        --ball-cache outputs/dingles_serve_volley/replay_cache.pkl \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --poses outputs/dingles_serve_volley/poses.pkl \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --players outputs/dingles_serve_volley/players.json \\
        --out outputs/dingles_serve_volley/swings.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.analysis.swings import THRESHOLD, detect_swings  # noqa: E402
from src.session.structure import NET_Y_M, SERVICE_LINE_FROM_NET_M as NET_ZONE_M  # noqa: E402
from src.tracking.candidate_tracker import track_ball_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ball-cache", required=True)
    parser.add_argument("--people", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--report", help="session_report.json - to skip its cutaways")
    parser.add_argument("--players", help="players.json - to name players and drop the ball cart")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cache = pickle.load(open(args.ball_cache, "rb"))
    people = pickle.load(open(args.people, "rb"))
    poses = pickle.load(open(args.poses, "rb"))
    if poses.get("every", 1) != 1:
        raise SystemExit(f"{args.poses} has a pose every {poses['every']} frames; swings need every frame "
                         "(scripts/track_poses.py --every 1)")
    fps = cache["fps"]
    ball = {}
    for track in track_ball_paths(cache["candidates"]):
        for f, d in track.detections.items():
            ball.setdefault(f, []).append((d.x, d.y))
    skip_frames = set()
    court = None
    if args.report:
        report = json.loads(Path(args.report).read_text())
        for a, b in report["clip"]["cutaways_s"]:
            skip_frames.update(range(int(round(a * fps)), int(round(b * fps))))
        court = CourtCalibration(homography=np.array(report["court"]["homography"]))
    player_of, not_people = {}, set()
    if args.players:
        doc = json.loads(Path(args.players).read_text())
        player_of = {int(t): p for t, p in doc["player_of_tracklet"].items()}
        not_people = set(doc.get("not_people", []))

    swings = detect_swings(people["people"], poses["poses"], ball, skip_tracklets=not_people,
                           skip_frames=skip_frames, threshold=args.threshold)
    boxes = {(f, int(t)): (x1, y1, x2, y2) for f, row in enumerate(people["people"])
             for t, x1, y1, x2, y2, _c in row if t is not None}
    out = []
    for s in swings:
        item = {"t_s": round(s.frame / fps, 2), "frame": s.frame, "tracklet": s.tracklet,
                "player": player_of.get(s.tracklet), "strength": round(s.strength, 3), "ball_near": s.ball_near}
        box = boxes.get((s.frame, s.tracklet))
        if court is not None and box is not None:
            # Where the hitter stood: which end, and whether inside the
            # service line (a shot from the net) - for rallies and themes.
            x, y = court.pixel_to_world((box[0] + box[2]) / 2.0, box[3])
            item["court_m"] = [round(x, 2), round(y, 2)]
            item["end"] = "far" if y < NET_Y_M else "near"
            item["at_net"] = abs(y - NET_Y_M) <= NET_ZONE_M
        out.append(item)
    per_player = Counter(s["player"] for s in out)
    Path(args.out).write_text(json.dumps({
        "method": "wrist travel relative to the shoulders, in player heights (src/analysis/swings.py)",
        "threshold": args.threshold,
        "per_player": {str(k): v for k, v in sorted(per_player.items(), key=lambda kv: str(kv[0]))},
        "swings": out,
    }, indent=1))
    print(f"{len(out)} swings: " + ", ".join(f"Player {k}: {v}" for k, v in sorted(per_player.items(), key=lambda kv: str(kv[0]))))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
