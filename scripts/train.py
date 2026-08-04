"""Fine-tune a YOLO model to detect the tennis ball.

The ball is a handful of pixels and moves fast enough to blur, so the
defaults below deliberately diverge from generic YOLO fine-tuning:

- imgsz=1280: broadcast frames downscaled to the usual 640 shrink the ball
  to a couple of pixels, which is often smaller than the smallest anchor
  the detector can localize.
- mosaic=0.0: mosaic augmentation stitches four images into quadrants,
  which regularly crops out or badly distorts an object this small; plain
  scale/translate augmentation preserves it instead.
- close_mosaic=0: nothing to disable if mosaic was never on.
- lower box loss weight increase (box=7.5) and higher default epoch count:
  small-object gradients are noisier, so it needs longer to converge.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(REPO_ROOT / "configs" / "ball_dataset.yaml"),
        help="Path to YOLO dataset yaml",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base checkpoint to fine-tune (COCO-pretrained .pt, or a previous run's best.pt to resume tuning)",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=30, help="Early-stop patience")
    parser.add_argument("--device", default=None, help="e.g. 0 or cpu; auto-detected if omitted")
    parser.add_argument("--project", default=str(REPO_ROOT / "runs"))
    parser.add_argument("--name", default="ball_detector")
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
        mosaic=0.0,
        scale=0.5,
        translate=0.2,
        box=7.5,
        cls=0.5,
        exist_ok=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nBest checkpoint: {best}")
    print(f"Copy it into weights/ball_detector.pt to use it for inference:")
    print(f"  cp {best} {REPO_ROOT / 'weights' / 'ball_detector.pt'}")


if __name__ == "__main__":
    main()
