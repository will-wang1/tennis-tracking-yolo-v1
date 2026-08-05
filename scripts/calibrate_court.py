"""Turn four manually-read court-corner pixel coordinates into a calibration file.

Run scripts/extract_calibration_frame.py first to get a still frame, read off
the pixel (x, y) of the baseline-left, baseline-right, service-left, and
service-right corners nearest the camera in an image viewer, then:

    python scripts/calibrate_court.py --frame outputs/calibration_frame.jpg \
        --baseline-left 340,980 --baseline-right 1580,980 \
        --service-left 520,650 --service-right 1400,650 \
        --court-type singles --out configs/court_calibration.json

This maps those four pixel points to their known real-world positions (a
standard court is 8.23m wide for singles, 10.97m for doubles, with the
baseline-to-service-line distance fixed at 5.485m) via a homography, and
saves it for src/analysis/court_calibration.py to load at inference time.

The printed meters-per-pixel sanity check is worth a glance before trusting
the result - if it's wildly off (e.g. under 1cm/px or over 10cm/px for a
typical broadcast shot), a coordinate was probably misread.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import (  # noqa: E402
    CORNER_ORDER,
    DOUBLES_COURT_REFERENCE_POINTS,
    SINGLES_COURT_REFERENCE_POINTS,
    CourtCalibration,
)


def parse_point(value: str) -> tuple[float, float]:
    x_str, y_str = value.split(",")
    return float(x_str), float(y_str)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", required=True, help="Reference frame image (record-keeping only)")
    parser.add_argument("--baseline-left", required=True, type=parse_point, metavar="X,Y")
    parser.add_argument("--baseline-right", required=True, type=parse_point, metavar="X,Y")
    parser.add_argument("--service-left", required=True, type=parse_point, metavar="X,Y")
    parser.add_argument("--service-right", required=True, type=parse_point, metavar="X,Y")
    parser.add_argument("--court-type", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--out", default="configs/court_calibration.json")
    args = parser.parse_args()

    world_reference = (
        SINGLES_COURT_REFERENCE_POINTS if args.court_type == "singles" else DOUBLES_COURT_REFERENCE_POINTS
    )
    pixel_by_corner = {
        "baseline_left": args.baseline_left,
        "baseline_right": args.baseline_right,
        "service_left": args.service_left,
        "service_right": args.service_right,
    }
    pixel_points = [pixel_by_corner[corner] for corner in CORNER_ORDER]
    world_points = [world_reference[corner] for corner in CORNER_ORDER]

    calibration = CourtCalibration.from_points(pixel_points, world_points)
    calibration.save(args.out)

    # sanity check: meters-per-pixel near the center of the 4 clicked points
    center_px = sum(p[0] for p in pixel_points) / 4, sum(p[1] for p in pixel_points) / 4
    nearby_px = (center_px[0] + 10, center_px[1])
    meters_per_10px = calibration.pixel_distance_to_meters(*center_px, *nearby_px)

    print(f"Wrote calibration to {args.out}")
    print(f"Sanity check: ~{meters_per_10px / 10:.4f} m/px near the court center")
    print(
        "If that's under 0.01 m/px or over 0.10 m/px for a typical broadcast "
        "shot, double check the four coordinates you entered."
    )


if __name__ == "__main__":
    main()
