"""Download the project's own pretrained checkpoints into weights/.

Fetches exactly the files main.py/src/pipeline.py already default to (see
PipelineOptions in src/pipeline.py) by Google Drive file ID, so this is a
one-command way to populate weights/ on a new machine (a laptop running the
worker, say) instead of downloading each file by hand through the Drive UI.

    python scripts/download_pretrained_weights.py

Skips a file if it already exists locally - safe to re-run. Downloads
everything by default; use --only to fetch a subset (see --help for names).

Note: gdown needs the files to be link-shareable. If a download fails with
a permission/HTML-page error instead of the real file, the Drive folder
isn't public - open it directly instead:
https://drive.google.com/drive/folders/1bO2j3vnALQbPGg8d1inkqUImXPHy_tDH
"""

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# name -> (Drive file ID, output filename under weights/)
FILES = {
    "ball_detector": ("1BlE2hjEW4HJnpc4uUD_1RbPyMCN3bid2", "ball_detector.pt"),
    "wasb": ("1V5Kn_5DIJT7_iPrZMuitYP38PSiqsmym", "wasb_tennis_pretrained.pth.tar"),
    "court_net": ("18hP5vmbhU20CPS6annGAbeaA35ZHLHts", "court_net_pretrained.pt"),
    "tracknet": ("1yODWl3BdgyGSWYXjXc7YCeyR5gGOgZe-", "tracknet_pretrained.pt"),
    "bounce_catboost": ("1hYZwxnv_1JZEnv4DfAnjIp2DGtWrP2d0", "bounce_catboost_pretrained.cbm"),
    "movenet": ("1ODn4KQKYa6xpr55rFja1ishSqtqqzBIE", "movenet_singlepose_lightning_int8.tflite"),
    "shot_classifier": ("1oyutra59zenPvbnvsH_5OTlktbL0oLaF", "shot_classifier_rnn_pretrained.h5"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FILES),
        help=f"Only fetch these (default: all). Choices: {', '.join(FILES)}",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "weights"))
    args = parser.parse_args()

    try:
        import gdown
    except ImportError:
        raise SystemExit("gdown is required: pip install gdown (see requirements.txt)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.only or list(FILES)
    for name in names:
        file_id, filename = FILES[name]
        dest = out_dir / filename
        if dest.exists():
            print(f"Skipping {filename} (already exists)")
            continue
        print(f"Downloading {filename}...")
        gdown.download(id=file_id, output=str(dest), quiet=False)

    print(f"Done. Weights are in {out_dir}")


if __name__ == "__main__":
    main()
