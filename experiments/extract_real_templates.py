"""Extract real watermark crops from train positives using YOLO boxes.
Produces 3 grayscale crops per class for NCC verification."""
import sys, os
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")
import cv2
import numpy as np
import prepare, detect

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRATCH, "real_templates")
os.makedirs(OUT, exist_ok=True)

dataset = prepare.discover_images()
train_set, _ = prepare.split_dataset(dataset)
detect.load_yolo()

want = {"minimax_hailuo": 40, "text_tpdne": 40, "grok": 40, "gemini": 40, "dalle": 11}
per_class = {}
for s in train_set:
    lab = s["label"]
    if lab not in want:
        continue
    per_class.setdefault(lab, [])
    if len(per_class[lab]) >= want[lab]:
        continue
    per_class[lab].append(s["path"])

for lab, paths in per_class.items():
    dets = []
    for p in paths:
        r = detect.yolo_detect(p)
        if r and r["label"] == lab and r["confidence"] > 0.5:
            img = cv2.imread(p)
            if img is None:
                continue
            x1, y1, x2, y2 = r["bbox"]
            crop = img[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0:
                continue
            dets.append((r["confidence"], crop, img.shape[1]))
    dets.sort(key=lambda d: -d[0])
    for i, (conf, crop, imw) in enumerate(dets[:3]):
        # normalize to a 1024-wide-image scale so scale ladder is consistent
        scale = 1024.0 / imw
        crop = cv2.resize(crop, (max(8, int(crop.shape[1]*scale)), max(8, int(crop.shape[0]*scale))))
        cv2.imwrite(os.path.join(OUT, f"{lab}_{i}.png"), crop)
        print(lab, i, f"conf={conf:.2f}", crop.shape)
