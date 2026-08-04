"""Download the tennis-ball detection dataset from Roboflow into data/raw/.

Requires ROBOFLOW_API_KEY to be set (see data/README.md for how to get one).
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="viren-dhanwani")
    parser.add_argument("--project", default="tennis-ball-detection")
    parser.add_argument("--version", type=int, default=6)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "raw"))
    args = parser.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit(
            "ROBOFLOW_API_KEY is not set. Create a free account at "
            "https://roboflow.com, copy your API key, and run:\n"
            "  export ROBOFLOW_API_KEY=...\n"
            "then re-run this script. See data/README.md for details."
        )

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8", location=args.out)
    print(f"Dataset downloaded to {dataset.location}")


if __name__ == "__main__":
    main()
