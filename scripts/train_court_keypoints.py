"""Train a YOLO-pose model to detect the 14 standard tennis court keypoints.

Unlike the ball detector, the court is a large, mostly-rigid structure that
fills most of the frame, so this deliberately does NOT copy that script's
tiny-object defaults (mosaic disabled, low box weight, etc.) - normal YOLO
pose-training defaults are appropriate here.

Run scripts/download_court_dataset.py then scripts/prepare_court_keypoint_dataset.py
first to produce configs/court_keypoints_dataset.yaml.

    python scripts/train_court_keypoints.py --epochs 60
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "configs" / "court_keypoints_dataset.yaml"),
        help="Path to YOLO pose dataset yaml (from scripts/prepare_court_keypoint_dataset.py)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Base checkpoint to fine-tune (COCO-pose-pretrained .pt, or a previous run's "
        "best.pt to resume tuning)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="The source dataset is large (~8.8k images) relative to the ball dataset, so "
        "this converges in far fewer epochs than scripts/train.py's default",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=15, help="Early-stop patience")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; auto-detected if omitted")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--name", default="court_keypoint_detector")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader worker processes - see scripts/train.py's docstring for why this "
        "needs lowering from ultralytics' default of 8 on Windows",
    )
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        exist_ok=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest checkpoint: {best}")
    print("Use it directly via --pose-weights, e.g.:")
    print(f"  python scripts/calibrate_court_auto.py --input match.mp4 --keypoint-weights {best}")


if __name__ == "__main__":
    main()
