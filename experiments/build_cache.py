"""Cache raw model outputs (YOLO, GBT binary/multi, CNN) for val + wild + sora frames.
Lets us sweep fusion rules offline without re-running setup."""
import sys, os, time, json, glob
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")

import numpy as np
import pandas as pd
import prepare, detect

SCRATCH = os.path.dirname(os.path.abspath(__file__))

dataset = prepare.discover_images()
train_set, val_set = prepare.split_dataset(dataset)
templates = prepare.get_templates()
print("setup...", flush=True)
detect.setup(train_set=train_set, templates=templates)

sets = {"val": [(s["path"], s["label"]) for s in val_set]}
wild = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")
sets["wild"] = [(r.path, f"{r.partition}|{r.weak_label}") for r in wild.itertuples()]
sets["sora_vid"] = [(p, "sora_video|watermarked") for p in sorted(glob.glob(os.path.join(SCRATCH, "sora_frames/*.jpg")))]

rows = []
t0 = time.time()
n_done = 0
for set_name, items in sets.items():
    for path, meta in items:
        try:
            yolo = detect.yolo_detect(path)
            gray, edges, hsv, h, w = detect.load_image(path)
            ts = detect.get_template_scores(edges, gray, h, w)
            feats = detect.extract_features(gray, edges, hsv, h, w, ts)
            cnn_p = detect.get_wm_cnn_prob(path)
            feats.append(cnn_p)
            bp = detect.BINARY_MODEL.predict_proba([feats])[0]
            wm_prob = float(bp[list(detect.BINARY_MODEL.classes_).index("watermarked")])
            proba = detect.MODEL.predict_proba([feats])[0]
            classes = list(detect.MODEL.classes_)
            row = {"set": set_name, "path": path, "meta": meta,
                   "yolo_label": yolo["label"] if yolo else "",
                   "yolo_conf": yolo["confidence"] if yolo else 0.0,
                   "yolo_bbox": json.dumps(yolo["bbox"]) if yolo else "",
                   "wm_prob": wm_prob, "cnn_prob": cnn_p}
            for c, p in zip(classes, proba):
                row[f"p_{c}"] = float(p)
            # keep top template scores for potential verification rules
            for k, v in ts.items():
                row[f"ts_{k}"] = v
            rows.append(row)
        except Exception as e:
            rows.append({"set": set_name, "path": path, "meta": meta, "error": str(e)[:80]})
        n_done += 1
        if n_done % 100 == 0:
            print(f"{n_done} ({time.time()-t0:.0f}s)", flush=True)

pd.DataFrame(rows).to_parquet(os.path.join(SCRATCH, "pred_cache.parquet"))
print(f"done: {len(rows)} rows in {time.time()-t0:.0f}s")
