"""Assemble yolo_ds_v3: v1-era positives + real sora pseudo-labels + realistic
tick/flower composites; lighter negative set (v2's hard negatives killed recall)."""
import glob, os, shutil, sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import cv2

SCRATCH = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SCRATCH, "yolo_ds")
DST = os.path.join(SCRATCH, "yolo_ds_v3")
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")
import detect

CLASSES = ["dalle", "gemini", "grok", "minimax_hailuo", "text_tpdne", "sora", "openai_logo"]
rng = np.random.default_rng(77)

for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(os.path.join(DST, sub), exist_ok=True)

# 1) carry over from yolo_ds: all real*/realneg*, bg* positives EXCEPT class-5
#    (wrong-style sora template), bg* negatives subsampled to ~700
neg_kept = 0
for split in ("train", "val"):
    for lp in sorted(glob.glob(os.path.join(SRC, "labels", split, "*.txt"))):
        base = os.path.basename(lp)[:-4]
        ip = os.path.join(SRC, "images", split, base + ".jpg")
        if not os.path.exists(ip): continue
        txt = open(lp).read().strip()
        if base.startswith(("sorareal", "oaireal")):
            continue  # rebuilt below
        if txt:
            ci = int(txt.split()[0])
            if base.startswith("bg") and ci == 5:
                continue  # drop wrong-appearance sora synthetics
        else:
            if base.startswith("bg"):
                if neg_kept >= 700: continue
                neg_kept += 1
        os.link(ip, os.path.join(DST, "images", split, base + ".jpg"))
        shutil.copy(lp, os.path.join(DST, "labels", split, base + ".txt"))

print("carried over; wild negatives kept:", neg_kept, flush=True)

# 2) real sora pseudo-labels: v1 yolo conf>0.3 on fresh catalog sora frames
bench = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")
EXCLUDE = set(os.path.basename(p).split(".")[0] for p in bench.path)
files = sorted(glob.glob("/truenas-big/catalog/images/source_dataset_name=sora/**/*.parquet", recursive=True))
paths = []
for f in files:
    d = pq.read_table(f, columns=["sha256", "file_type"]).to_pandas()
    d = d.iloc[len(d)//3:]
    for _, r in d.iterrows():
        if r.sha256 in EXCLUDE: continue
        p = f"/truenas-big/images/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.{r.file_type or 'JPEG'}"
        if os.path.exists(p): paths.append(p)
        if len(paths) >= 1200: break
    if len(paths) >= 1200: break
print("sora pool:", len(paths), flush=True)

detect.load_yolo()
n_sora = 0
for p in paths:
    try:
        y = detect.yolo_detect(p)
    except Exception:
        continue
    if not y or y["confidence"] < 0.3: continue
    img = cv2.imread(p)
    if img is None: continue
    h, w = img.shape[:2]
    x1, y1, x2, y2 = y["bbox"]
    if x2 - x1 < 15 or y2 - y1 < 8: continue
    split = "val" if n_sora % 15 == 0 else "train"
    name = f"sorareal{n_sora:05d}"
    cv2.imwrite(os.path.join(DST, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(DST, "labels", split, name + ".txt"), "w") as f:
        f.write(f"5 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n")
    n_sora += 1
print("real sora pseudo-labels:", n_sora, flush=True)

# 3) composites: tick rows (class 5) + flower (class 6) onto wild bg negatives
TICKS = [cv2.imread(os.path.join(SCRATCH, "tick_candidates", f"c{i:03d}.png"), cv2.IMREAD_GRAYSCALE)
         for i in (0, 6, 14, 18)]
TICKS = [t for t in TICKS if t is not None]
flower = cv2.imread("images/watermarks/openai_watermark.png", cv2.IMREAD_UNCHANGED)
fl_alpha = flower[:, :, 3].astype(np.float32) / 255.0 if flower.shape[2] == 4 else None

bg_pool = sorted(glob.glob(os.path.join(DST, "images", "train", "bg*.jpg")))
bg_neg = [p for p in bg_pool if open(p.replace("images", "labels").replace(".jpg", ".txt")).read().strip() == ""]
rng.shuffle(bg_neg)
n_comp = 0
for i, bp in enumerate(bg_neg[:360]):
    img = cv2.imread(bp)
    if img is None: continue
    h, w = img.shape[:2]
    kind = "tick" if i % 2 == 0 else "flower"
    res = img.copy()
    if kind == "tick":
        t = TICKS[int(rng.integers(0, len(TICKS)))]
        tw = max(int(w * rng.uniform(0.14, 0.24)), 40)
        th = max(int(t.shape[0] * tw / t.shape[1]), 8)
        if tw >= w or th >= h: continue
        tr = cv2.resize(t, (tw, th)).astype(np.float32)
        lo, hi = np.percentile(tr, 40), max(tr.max(), 1)
        a = np.clip((tr - lo) / (hi - lo + 1e-6), 0, 1) * rng.uniform(0.5, 0.9)
        x = int(rng.integers(0, w - tw)); y = int(rng.integers(0, h - th))
        roi0 = img[y:y+th, x:x+tw]
        if roi0.mean() > 185: continue
        roi = roi0.astype(np.float32)
        res[y:y+th, x:x+tw] = (roi * (1 - a[..., None]) + 255 * a[..., None]).astype(np.uint8)
        ci, bw_, bh_ = 5, tw, th
    else:
        ts = max(int(w * rng.uniform(0.025, 0.05)), 16)
        if ts >= h or ts >= w: continue
        fr = cv2.resize(flower, (ts, ts), interpolation=cv2.INTER_AREA)
        a = (fr[:, :, 3].astype(np.float32) / 255.0 if fr.shape[2] == 4 else
             cv2.cvtColor(fr[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
        a = a * rng.uniform(0.35, 0.75)
        x = int(rng.integers(0, w - ts)); y = int(rng.integers(0, h - ts))
        roi0 = img[y:y+ts, x:x+ts]
        if roi0.mean() > 185: continue
        roi = roi0.astype(np.float32)
        res[y:y+ts, x:x+ts] = (roi * (1 - a[..., None]) + 255 * a[..., None]).astype(np.uint8)
        ci, bw_, bh_ = 6, ts, ts
        th = ts
    # degrade: mild blur + jpeg round-trip
    if rng.random() < 0.5:
        res = cv2.GaussianBlur(res, (3, 3), 0.6)
    q = int(rng.integers(55, 90))
    res = cv2.imdecode(cv2.imencode(".jpg", res, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    split = "val" if n_comp % 20 == 0 else "train"
    name = f"comp{n_comp:05d}"
    cv2.imwrite(os.path.join(DST, "images", split, name + ".jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(DST, "labels", split, name + ".txt"), "w") as f:
        f.write(f"{ci} {(x+bw_/2)/w:.6f} {(y+th/2)/h:.6f} {bw_/w:.6f} {th/h:.6f}\n")
    n_comp += 1
print("composites:", n_comp, flush=True)

with open(os.path.join(DST, "data.yaml"), "w") as f:
    f.write(f"path: {DST}\ntrain: images/train\nval: images/val\nnames:\n")
    for i, c in enumerate(CLASSES):
        f.write(f"  {i}: {c}\n")
print("done")
