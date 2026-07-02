"""Localize the OpenAI flower watermark in catalog sora frames via multi-scale
gray NCC with the master flower template; write tight openai_logo (class 6) labels."""
import glob, os
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import cv2

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRATCH, "yolo_ds")
IMAGES_ROOT = "/truenas-big/images"
REPO = "/home/peter/repos/detesia/woahamark_detection"

# purge previous noisy sora pseudo-labels
for f in glob.glob(os.path.join(OUT, "images", "*", "sorareal*.jpg")) + \
         glob.glob(os.path.join(OUT, "labels", "*", "sorareal*.txt")):
    os.remove(f)

bench = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")
EXCLUDE = set(os.path.basename(p).split(".")[0] for p in bench.path)

t = cv2.imread(os.path.join(REPO, "images/watermarks/openai_watermark.png"), cv2.IMREAD_UNCHANGED)
if t.shape[2] == 4:
    alpha = t[:, :, 3].astype(np.float32) / 255.0
    tg_full = (cv2.cvtColor(t[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) * alpha).astype(np.uint8)
else:
    tg_full = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

files = sorted(glob.glob("/truenas-big/catalog/images/source_dataset_name=sora/**/*.parquet", recursive=True))
paths = []
for f in files:
    df = pq.read_table(f, columns=["sha256", "file_type"]).to_pandas()
    df = df.iloc[len(df)//3:]
    for _, r in df.iterrows():
        if r.sha256 in EXCLUDE: continue
        p = f"{IMAGES_ROOT}/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.{r.file_type or 'JPEG'}"
        if os.path.exists(p):
            paths.append(p)
        if len(paths) >= 800: break
    if len(paths) >= 800: break
print("pool:", len(paths))

n = 0
for i, p in enumerate(paths):
    img = cv2.imread(p)
    if img is None: continue
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best = (0.0, None, None)
    for ts in (24, 32, 40, 48, 60):
        tt = cv2.resize(tg_full, (ts, ts))
        if ts >= h or ts >= w: continue
        res = cv2.matchTemplate(gray, tt, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (mx, loc, ts)
    score, loc, ts = best
    if score < 0.55 or loc is None:
        continue
    x1, y1 = loc; x2, y2 = x1 + ts, y1 + ts
    split = "val" if n % 20 == 0 else "train"
    name = f"oaireal{n:05d}"
    cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
        f.write(f"6 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {ts/w:.6f} {ts/h:.6f}\n")
    n += 1
    if n % 50 == 0: print(n, flush=True)
print("openai_logo real labels:", n)
