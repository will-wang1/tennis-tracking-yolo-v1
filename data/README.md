# Dataset

Training data is not committed to this repo. Fetch it into `data/raw/` before
running `scripts/train.py`, using one of the two paths below.

## Option A - Roboflow (recommended)

A ready-made, pre-labeled tennis ball dataset (broadcast footage, bounding
boxes around the ball) is available on Roboflow Universe:
https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection

1. Create a free Roboflow account and copy your API key.
2. `export ROBOFLOW_API_KEY=...`
3. `python scripts/download_dataset.py`

This downloads the dataset in YOLOv8 format directly into `data/raw/`,
matching the layout `configs/ball_dataset.yaml` expects
(`train/images`, `valid/images`, `test/images`, each with a sibling
`labels/` folder).

## Option B - Your own footage

If you'd rather label your own clips (e.g. to match a specific camera angle
or court):

1. `python scripts/extract_frames.py --video path/to/match.mp4 --out data/raw/unlabeled --every 5`
   to pull frames out of your own match footage.
2. Label the ball in each frame with a bounding box using any YOLO-format
   annotation tool (e.g. [CVAT](https://www.cvat.ai/),
   [LabelImg](https://github.com/HumanSignal/labelImg), or the Roboflow
   annotator). Export as YOLO format, single class `ball`.
3. Arrange the export into `data/raw/{train,valid,test}/{images,labels}`
   to match `configs/ball_dataset.yaml`.

## Why the ball needs its own dataset

The ball is a handful of pixels, moving fast enough to motion-blur, and is
frequently occluded by a player or the net. General-purpose COCO-pretrained
YOLO weights don't have a "tennis ball" class and won't reliably pick it out
at broadcast resolution - that's the entire reason this repo fine-tunes on
ball-specific footage instead of using an off-the-shelf detector.
