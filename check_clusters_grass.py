import pandas as pd, cv2, os

video_path = r"C:\Users\willi\Downloads\test_video.mp4"
df = pd.read_csv("outputs/detections_grasscourt.csv")

regions = {
    "bottom_crowd": (600, 660, 1040, 1080),
    "upper_boxes": (860, 900, 200, 240),
    "top_stands": (860, 900, 80, 120),
}

cap = cv2.VideoCapture(video_path)
print("video opened:", cap.isOpened())
for name, (xmin, xmax, ymin, ymax) in regions.items():
    hits = df[df.x.between(xmin, xmax) & df.y.between(ymin, ymax)]
    out_dir = f"outputs/check_frames_grass/{name}"
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    sample = hits.iloc[:: max(1, len(hits) // 6)].head(6)
    for i, row in sample.iterrows():
        cap.set(cv2.CAP_PROP_POS_FRAMES, row.frame)
        ok, frame = cap.read()
        if ok:
            x, y = int(row.x), int(row.y)
            crop = frame[max(0, y - 80):y + 80, max(0, x - 80):x + 80]
            cv2.imwrite(f"{out_dir}/frame_{int(row.frame)}_full.jpg", frame)
            cv2.imwrite(f"{out_dir}/frame_{int(row.frame)}_crop.jpg", crop)
            saved += 1
    print(name, "-> matches:", len(hits), "saved:", saved)
