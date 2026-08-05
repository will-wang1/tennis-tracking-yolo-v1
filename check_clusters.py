import pandas as pd, cv2, os

video_path = r"C:\Users\willi\Downloads\tennis point.mp4"
df = pd.read_csv("outputs/detections_hardcourt.csv")

regions = {
    "sidebar_graphic": (1530, 1570, 455, 505),
    "net_center_a": (830, 900, 410, 470),
    "net_center_b": (950, 1010, 390, 430),
}

cap = cv2.VideoCapture(video_path)
print("video opened:", cap.isOpened())
for name, (xmin, xmax, ymin, ymax) in regions.items():
    hits = df[df.x.between(xmin, xmax) & df.y.between(ymin, ymax)]
    out_dir = f"outputs/check_frames/{name}"
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    # spread samples across the frame range instead of just the first few
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
