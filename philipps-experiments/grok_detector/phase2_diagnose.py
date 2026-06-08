"""Phase 2: diagnose. Build ROI+Canny montages for worst-FN and top-scoring-negatives."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .config import CANNY_HI, CANNY_LO, DEBUG_DIR, ROI_FRAC


def load_test_scores(csv_path="debug/phase1/test_scores.csv"):
    rows = list(csv.DictReader(open(csv_path)))
    pos = sorted([r for r in rows if int(r["label"]) == 1], key=lambda r: float(r["score"]))
    neg = sorted([r for r in rows if int(r["label"]) == 0], key=lambda r: -float(r["score"]))
    return pos, neg


def roi_canny(img):
    h, w = img.shape[:2]
    rh, rw = max(1, int(h * ROI_FRAC)), max(1, int(w * ROI_FRAC))
    roi = img[h - rh:, w - rw:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    return roi, edges


def build_panel(rows, max_n=24, tile_w=180, tile_h=120):
    items = rows[:max_n]
    cols = 6
    n_rows = (len(items) + cols - 1) // cols
    cell_w = tile_w * 2 + 8  # ROI side-by-side with edges
    cell_h = tile_h + 24
    canvas = np.full((n_rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for i, r in enumerate(items):
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            continue
        roi, edges = roi_canny(img)
        roi_r = cv2.resize(roi, (tile_w, tile_h))
        edges_r = cv2.cvtColor(cv2.resize(edges, (tile_w, tile_h)), cv2.COLOR_GRAY2BGR)
        cell = np.concatenate([roi_r, np.full((tile_h, 8, 3), 30, np.uint8), edges_r], axis=1)
        rr, cc = divmod(i, cols)
        y0 = rr * cell_h
        x0 = cc * cell_w
        canvas[y0:y0 + tile_h, x0:x0 + cell_w] = cell
        label = f"{Path(r['path']).name[:30]} s={float(r['score']):.3f}"
        cv2.putText(canvas, label, (x0 + 2, y0 + tile_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    out = DEBUG_DIR / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    pos, neg = load_test_scores()

    cv2.imwrite(str(out / "worst_fn_pos_lowest_scores.jpg"), build_panel(pos, max_n=24))
    cv2.imwrite(str(out / "top_neg_highest_scores.jpg"), build_panel(neg, max_n=24))
    print("wrote panels to", out)

    print("\nWorst-scoring positives (potential FN drivers):")
    for r in pos[:12]:
        print(f"  {float(r['score']):.3f}  {Path(r['path']).name}")
    print("\nHighest-scoring negatives (would-be FPs at lower threshold):")
    for r in neg[:12]:
        print(f"  {float(r['score']):.3f}  {Path(r['path']).name}")


if __name__ == "__main__":
    main()
