"""Download the court-keypoint reference dataset into data/raw_court_source/.

Source: yastrebksv/TennisCourtDetector (https://github.com/yastrebksv/TennisCourtDetector)
- 8,841 broadcast frames across hard, clay, and grass courts, each hand-annotated
  with the 14 standard court keypoints (see src.detection.court_keypoint_detector).
  Chosen over the smaller Roboflow tennis-court-keypoint datasets (500-2.5k images,
  usually a single court surface) specifically because it already spans all three
  surfaces - this repo's own test clips are hard court and grass.
- Free direct download, no API key needed.

    python scripts/download_court_dataset.py

Then run scripts/prepare_court_keypoint_dataset.py to convert it into the
YOLO-pose format scripts/train_court_keypoints.py expects.
"""

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Google Drive file IDs linked from the dataset's README.
DATASET_FILE_ID = "1lhAaeQCmk2y440PmagA0KmIVBIysVMwu"
PRETRAINED_WEIGHTS_FILE_ID = "1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "raw_court_source"))
    parser.add_argument(
        "--skip-pretrained",
        action="store_true",
        help="Skip downloading the reference project's own pretrained weights "
        "(useful for comparison/sanity-checking, not required to train your own)",
    )
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown (see requirements.txt)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_path = out_dir / "dataset.zip"
    print(f"Downloading dataset to {zip_path} (this is a multi-GB file, will take a while)...")
    gdown.download(id=DATASET_FILE_ID, output=str(zip_path), quiet=False)

    print(f"Extracting to {out_dir}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    zip_path.unlink()

    if not args.skip_pretrained:
        weights_dir = REPO_ROOT / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "court_keypoint_reference.pt"
        print(f"Downloading reference project's pretrained weights to {weights_path}...")
        gdown.download(id=PRETRAINED_WEIGHTS_FILE_ID, output=str(weights_path), quiet=False)
        print(
            "Note: those weights are in the reference project's own PyTorch/TrackNet "
            "format, not directly loadable by this repo's ultralytics-based "
            "CourtKeypointDetector - they're useful as a reference/sanity-check, not "
            "a drop-in checkpoint. Train your own with scripts/train_court_keypoints.py."
        )

    print(f"Done. Raw dataset at {out_dir} - now run scripts/prepare_court_keypoint_dataset.py")


if __name__ == "__main__":
    main()
