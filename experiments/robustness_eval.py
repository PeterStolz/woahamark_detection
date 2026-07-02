"""Robustness benchmark: how does detection survive re-encoding/downscaling?"""
import sys, os, time
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")
import cv2
import pandas as pd
import prepare, detect

SCRATCH = os.path.dirname(os.path.abspath(__file__))
VAR_DIR = os.path.join(SCRATCH, "robust_variants")
os.makedirs(VAR_DIR, exist_ok=True)

dataset = prepare.discover_images()
train_set, val_set = prepare.split_dataset(dataset)
templates = prepare.get_templates()
positives = [s for s in val_set if s["label"] != "clean"]
cleans = [s for s in val_set if s["label"] == "clean"]

def variant(path, kind, out):
    img = cv2.imread(path)
    if img is None: return None
    if kind == "jpeg50":
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    elif kind == "jpeg30":
        cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 30])
    elif kind == "half":
        h, w = img.shape[:2]
        cv2.imwrite(out, cv2.resize(img, (w//2, h//2)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out

print("setup...", flush=True)
detect.setup(train_set=train_set, templates=templates)

rows = []
t0 = time.time()
for kind in ("orig", "jpeg50", "jpeg30", "half"):
    for group, samples in (("pos", positives), ("clean", cleans[:40])):
        for i, s in enumerate(samples):
            if kind == "orig":
                p = s["path"]
            else:
                p = variant(s["path"], kind, os.path.join(VAR_DIR, f"{kind}_{group}_{i}.jpg"))
                if p is None: continue
            r = detect.detect(p)
            rows.append({"kind": kind, "group": group, "true": s["label"],
                         "pred": r.get("label"), "binary": r.get("binary")})
    print(kind, f"({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(SCRATCH, "robustness_results.tsv"), sep="\t", index=False)
print("\n=== binary recall on val positives / FP on val clean ===")
for kind in ("orig", "jpeg50", "jpeg30", "half"):
    g = df[(df.kind == kind) & (df.group == "pos")]
    c = df[(df.kind == kind) & (df.group == "clean")]
    cls_ok = (g.pred == g.true).mean()
    print(f"{kind:8s} recall={(g.binary=='watermarked').mean():.3f} class-acc={cls_ok:.3f} cleanFP={(c.binary=='watermarked').mean():.3f}")
