"""Crop bottom-right corners of wild samples into per-partition montages for visual QA."""
import sys, os
import cv2
import numpy as np
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")

CROP = 220  # px from bottom-right
PER_PART = 12
COLS = 4

for part in df.partition.unique():
    paths = df[df.partition == part].path.head(PER_PART).tolist()
    tiles = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        c = img[max(0, h - CROP):, max(0, w - CROP):]
        c = cv2.resize(c, (CROP, CROP))
        cv2.rectangle(c, (0, 0), (CROP - 1, CROP - 1), (0, 255, 0), 1)
        tiles.append(c)
    if not tiles:
        continue
    rows = []
    for i in range(0, len(tiles), COLS):
        row = tiles[i:i + COLS]
        while len(row) < COLS:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(os.path.join(SCRATCH, f"corners_{part}.jpg"), np.vstack(rows))
    print(part, len(tiles))
