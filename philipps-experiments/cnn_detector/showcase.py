"""Best/worst showcase montage per class. Shows the 192x192 BR-anchored crop the
model actually sees plus the per-class score."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from .config import CLASSES, DEBUG_DIR, INPUT_SIZE
from .data import random_crop_and_resize


TILE_W, TILE_H = 200, 200
GAP = 8
LABEL_H = 28
PANEL_HEADER_H = 36
PANEL_GAP = 14


def get_crop(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((TILE_H, TILE_W, 3), 50, dtype=np.uint8)
    crop = random_crop_and_resize(img, INPUT_SIZE, augment=False, rng=random.Random(0))
    return cv2.resize(crop, (TILE_W, TILE_H))


def panel(title, rows, threshold, cls, n=6):
    items = rows[:n]
    cell_w = TILE_W
    cell_h = TILE_H + LABEL_H
    grid_w = n * cell_w + (n - 1) * GAP
    canvas = np.full((PANEL_HEADER_H + cell_h, grid_w, 3), 28, dtype=np.uint8)
    cv2.putText(canvas, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"threshold={threshold:.2f}", (grid_w - 220, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    for i, r in enumerate(items):
        x0 = i * (cell_w + GAP); y0 = PANEL_HEADER_H
        canvas[y0:y0 + TILE_H, x0:x0 + cell_w] = get_crop(r["path"])
        sc = float(r[f"score_{cls}"]); lab = int(r[f"label_{cls}"]); pred = int(r[f"pred_{cls}"])
        kind = "POS" if lab == 1 else "NEG"
        verdict = "[OK]" if pred == lab else "[MISS]"
        cv2.putText(canvas, f"{kind} pred={'POS' if pred else 'NEG'} {verdict}  s={sc:.3f}",
                    (x0 + 4, y0 + TILE_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, Path(r["path"]).name[:28],
                    (x0 + 4, y0 + TILE_H + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
    return canvas


def stack(panels):
    width = max(p.shape[1] for p in panels)
    out = []
    for p in panels:
        if p.shape[1] < width:
            p = np.concatenate([p, np.full((p.shape[0], width - p.shape[1], 3), 28, np.uint8)], axis=1)
        out.append(p)
        out.append(np.full((PANEL_GAP, width, 3), 18, np.uint8))
    return np.concatenate(out[:-1], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", default=str(DEBUG_DIR / "full_eval" / "all_scores.csv"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out-dir", default=str(DEBUG_DIR))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.scores_csv)))
    out_dir = Path(args.out_dir)

    for cls in CLASSES:
        thr_col = f"score_{cls}"
        # Threshold info isn't in CSV; pick a default 0.5 just for display ribbon.
        # Actual prediction is from pred_{cls} column.
        pos = [r for r in rows if int(r[f"label_{cls}"]) == 1]
        neg = [r for r in rows if int(r[f"label_{cls}"]) == 0]
        best_pos = sorted([r for r in pos if int(r[f"pred_{cls}"]) == 1],
                          key=lambda r: -float(r[thr_col]))
        worst_pos = sorted(pos, key=lambda r: float(r[thr_col]))
        best_neg = sorted([r for r in neg if int(r[f"pred_{cls}"]) == 0],
                          key=lambda r: float(r[thr_col]))
        worst_neg = sorted(neg, key=lambda r: -float(r[thr_col]))

        # Try to read the threshold from the trained checkpoint for the title ribbon.
        try:
            import torch
            from .config import CHECKPOINTS_DIR
            ck = torch.load(CHECKPOINTS_DIR / "cnn_v3.pt", map_location="cpu", weights_only=False)
            thr = float(ck["thresholds"][cls])
        except Exception:
            thr = 0.5

        panels = [
            panel(f"BEST {cls} - highest-confidence true positives",
                  best_pos, thr, cls, args.n),
            panel(f"WORST {cls} - lowest scores on label=1 (FN if pred=NEG)",
                  worst_pos, thr, cls, args.n),
            panel(f"BEST not-{cls} - most confidently negative",
                  best_neg, thr, cls, args.n),
            panel(f"WORST not-{cls} - highest scores on label=0 (closest to triggering)",
                  worst_neg, thr, cls, args.n),
        ]
        montage = stack(panels)
        out_path = out_dir / f"showcase_{cls}.jpg"
        cv2.imwrite(str(out_path), montage)
        print(f"wrote {out_path} ({montage.shape[1]}x{montage.shape[0]})")


if __name__ == "__main__":
    main()
