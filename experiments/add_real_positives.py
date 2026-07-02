"""Add real train-split positives with YOLO-v1 pseudo-boxes to the exp13 dataset."""
import sys, os
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")
import cv2
import prepare, detect

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRATCH, "yolo_ds")
CLASSES = ["dalle", "gemini", "grok", "minimax_hailuo", "text_tpdne", "sora", "openai_logo"]

dataset = prepare.discover_images()
train_set, _ = prepare.split_dataset(dataset)
detect.load_yolo()

n_ok = n_skip = 0
for i, s in enumerate(train_set):
    lab = s["label"]
    if lab == "clean":
        # real dataset cleans as extra negatives (cap 300)
        if n_skip >= 300: continue
        img = cv2.imread(s["path"])
        if img is None: continue
        split = "val" if i % 20 == 0 else "train"
        name = f"realneg{i:05d}"
        cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(os.path.join(OUT, "labels", split, name + ".txt"), "w").close()
        n_skip += 1
        continue
    try:
        r = detect.yolo_detect(s["path"])
    except Exception:
        continue
    if not r or r["label"] != lab or r["confidence"] < 0.55:
        continue
    img = cv2.imread(s["path"])
    if img is None: continue
    h, w = img.shape[:2]
    x1, y1, x2, y2 = r["bbox"]
    ci = CLASSES.index(lab)
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    split = "val" if i % 20 == 0 else "train"
    name = f"real{i:05d}"
    cv2.imwrite(os.path.join(OUT, "images", split, name + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(OUT, "labels", split, name + ".txt"), "w") as f:
        f.write(f"{ci} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    n_ok += 1
    if n_ok % 200 == 0: print(n_ok, flush=True)

print(f"real positives: {n_ok}, real negatives: {n_skip}")

# dataset yaml
with open(os.path.join(OUT, "data.yaml"), "w") as f:
    f.write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n")
    for i, c in enumerate(CLASSES):
        f.write(f"  {i}: {c}\n")
print("wrote data.yaml")
