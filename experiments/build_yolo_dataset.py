"""Build YOLO training dataset for exp13:
- negatives: wild catalog images (disjoint from wild benchmark)
- positives: synthetic watermark overlays (7 classes) on wild backgrounds w/ exact boxes
- positives: real train-split images with YOLO-v1 pseudo-boxes (added separately)
"""
import glob, os, sys, random
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import cv2

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRATCH, "yolo_ds")
IMAGES_ROOT = "/truenas-big/images"
CATALOG = "/truenas-big/catalog/images"
REPO = "/home/peter/repos/detesia/woahamark_detection"

CLASSES = ["dalle", "gemini", "grok", "minimax_hailuo", "text_tpdne", "sora", "openai_logo"]

# benchmark shas to exclude
bench = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")
EXCLUDE = set(os.path.basename(p).split(".")[0] for p in bench.path)

PARTS = {"coco_2017": 400, "openimages": 400, "laion400m": 300, "hunyuan": 250, "wan": 250,
         "shareveo3": 250, "gpt_4o": 250, "nano_150k": 300, "civitai": 250, "ideogram_scrape": 250,
         "journeydb": 300, "dalle3": 200}

def sample_backgrounds():
    rows = []
    for part, n in PARTS.items():
        files = sorted(glob.glob(f"{CATALOG}/source_dataset_name={part}/**/*.parquet", recursive=True))
        got = 0
        for f in files:
            if got >= n: break
            try:
                df = pq.read_table(f, columns=["sha256", "file_type"]).to_pandas()
            except Exception:
                continue
            # skip the first rows used by the benchmark sampler; sample from the back half
            df = df.iloc[len(df)//2:]
            step = max(1, len(df) // max(1, n - got))
            for _, r in df.iloc[::step].iterrows():
                if got >= n: break
                if r.sha256 in EXCLUDE: continue
                p = f"{IMAGES_ROOT}/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.{r.file_type or 'JPEG'}"
                if not os.path.exists(p):
                    cands = glob.glob(f"{IMAGES_ROOT}/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.*")
                    if not cands: continue
                    p = cands[0]
                rows.append(p)
                got += 1
        print(part, got, flush=True)
    return rows

# ── watermark overlay (extends watermark_aug to sora/openai) ──
sys.path.insert(0, REPO)
os.chdir(REPO)

def load_tmpl(name):
    t = cv2.imread(f"images/watermarks/{name}", cv2.IMREAD_UNCHANGED)
    return t

TMPL = {
    "dalle": load_tmpl("dalle_watermark.png"),
    "gemini": load_tmpl("gemini_watermark.png"),
    "grok": load_tmpl("grok_watermark.png"),
    "minimax_hailuo": load_tmpl("hailuoaixminimax_watermark.png"),
    "text_tpdne": load_tmpl("this-person-does-not-exist_watermark.png"),
    "sora": load_tmpl("sora_watermark.png"),
    "openai_logo": load_tmpl("openai_watermark.png"),
}
# rendered size on a 1024-wide image: (w, h) approx
REND = {"dalle": (80, 16), "gemini": (22, 22), "grok": (150, 50), "minimax_hailuo": (200, 25),
        "text_tpdne": (330, 25), "sora": (90, 36), "openai_logo": (48, 48)}
# position mode: realistic corner vs anywhere
RANDOM_POS = {"sora", "openai_logo"}  # sora watermark moves; openai logo varies

def overlay(img, cls, rng):
    h, w = img.shape[:2]
    t = TMPL[cls]
    if t is None: return None
    scale = w / 1024.0
    tw = max(int(REND[cls][0] * scale * rng.uniform(0.8, 1.3)), 12)
    th = max(int(REND[cls][1] * scale * rng.uniform(0.8, 1.3)), 6)
    if tw >= w or th >= h: return None
    tr = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
    for _attempt in range(6):
        if cls in RANDOM_POS or rng.random() < 0.25:
            x = rng.integers(0, w - tw); y = rng.integers(0, h - th)
        elif cls == "text_tpdne":
            x = (w - tw) // 2 + int(rng.integers(-20, 20)); y = int(rng.integers(3, 15))
        else:
            x = w - tw - int(rng.integers(5, 30) * scale); y = h - th - int(rng.integers(5, 30) * scale)
        x, y = max(0, int(x)), max(0, int(y))
        roi0 = img[y:y+th, x:x+tw]
        # white semi-transparent marks are invisible on bright backgrounds — retry
        if cls != "dalle" and roi0.mean() > 190:
            continue
        break
    else:
        return None
    res = img.copy()
    if cls == "dalle":
        res[y:y+th, x:x+tw] = tr[:, :, :3]
    else:
        if tr.ndim == 3 and tr.shape[2] == 4:
            a = tr[:, :, 3].astype(np.float32) / 255.0
        else:
            g = cv2.cvtColor(tr[:, :, :3] if tr.ndim == 3 else tr, cv2.COLOR_BGR2GRAY)
            a = (g.astype(np.float32) / 255.0)
        op = 0.45 + rng.uniform(-0.12, 0.15)
        a = a * op
        fg = np.full((th, tw, 3), 255, np.float32)
        roi = res[y:y+th, x:x+tw].astype(np.float32)
        res[y:y+th, x:x+tw] = (roi * (1 - a[..., None]) + fg * a[..., None]).astype(np.uint8)
        # final visibility check: composite must perceptibly differ from original
        if np.abs(res[y:y+th, x:x+tw].astype(np.float32) - roi).mean() < 4.0:
            return None
    return res, (x, y, tw, th)

def main():
    rng = np.random.default_rng(1234)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    bgs = sample_backgrounds()
    rng.shuffle(bgs)
    print("backgrounds:", len(bgs), flush=True)
    n_pos = n_neg = 0
    idx = 0
    for i, p in enumerate(bgs):
        img = cv2.imread(p)
        if img is None: continue
        h, w = img.shape[:2]
        if max(h, w) > 1600:
            s = 1600 / max(h, w); img = cv2.resize(img, (int(w*s), int(h*s))); h, w = img.shape[:2]
        split = "val" if idx % 20 == 0 else "train"
        name = f"bg{idx:05d}"
        if rng.random() < 0.5:
            ci = int(rng.integers(0, len(CLASSES)))
            out = overlay(img, CLASSES[ci], rng)
            if out is None: continue
            res, (x, y, tw, th) = out
            cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cx, cy = (x + tw/2) / w, (y + th/2) / h
            with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
                f.write(f"{ci} {cx:.6f} {cy:.6f} {tw/w:.6f} {th/h:.6f}\n")
            n_pos += 1
        else:
            cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(os.path.join(OUT, "labels", split, name + ".txt"), "w").close()
            n_neg += 1
        idx += 1
        if idx % 500 == 0: print(idx, flush=True)
    print(f"synthetic: pos={n_pos} neg={n_neg}")

if __name__ == "__main__":
    main()
