"""Best/worst showcase montage. ROI-only (no Canny — this detector matches color, not edges)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .config import DEBUG_DIR, ROI_FRAC


TILE_W, TILE_H = 240, 160
GAP = 8
LABEL_H = 28
PANEL_HEADER_H = 36
PANEL_GAP = 14


def get_roi(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return np.full((TILE_H, TILE_W, 3), 50, dtype=np.uint8)
    h, w = img.shape[:2]
    rh, rw = max(1, int(h * ROI_FRAC)), max(1, int(w * ROI_FRAC))
    roi = img[h - rh:, w - rw:]
    return cv2.resize(roi, (TILE_W, TILE_H))


def panel(title, rows, threshold, n=6):
    items = rows[:n]
    cell_w, cell_h = TILE_W, TILE_H + LABEL_H
    grid_w = n * cell_w + (n - 1) * GAP
    canvas = np.full((PANEL_HEADER_H + cell_h, grid_w, 3), 28, dtype=np.uint8)
    cv2.putText(canvas, title, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"threshold={threshold:.2f}", (grid_w - 220, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    for i, r in enumerate(items):
        roi_img = get_roi(r["path"])
        x0 = i * (cell_w + GAP); y0 = PANEL_HEADER_H
        canvas[y0:y0 + TILE_H, x0:x0 + cell_w] = roi_img
        sc = float(r["score"]); lab = int(r["label"]); pred = int(r["prediction"])
        cls = "POS" if lab == 1 else "NEG"
        verdict = "[OK]" if pred == lab else "[MISS]"
        cv2.putText(canvas, f"{cls} pred={'POS' if pred else 'NEG'} {verdict}  s={sc:.3f}",
                    (x0 + 4, y0 + TILE_H + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, Path(r["path"]).name[:36],
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
    ap.add_argument("--threshold", type=float, default=0.59)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default=str(DEBUG_DIR / "showcase.jpg"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.scores_csv)))
    pos = [r for r in rows if int(r["label"]) == 1]
    neg = [r for r in rows if int(r["label"]) == 0]
    best_pos = sorted([r for r in pos if int(r["prediction"]) == 1], key=lambda r: -float(r["score"]))
    worst_pos = sorted(pos, key=lambda r: float(r["score"]))
    best_neg = sorted([r for r in neg if int(r["prediction"]) == 0], key=lambda r: float(r["score"]))
    worst_neg = sorted(neg, key=lambda r: -float(r["score"]))

    panels = [
        panel("BEST dalle - highest-confidence true positives", best_pos, args.threshold, args.n),
        panel("WORST dalle - lowest scores on label=1 (false negatives if pred=NEG)",
              worst_pos, args.threshold, args.n),
        panel("BEST real - most confidently 'no watermark'", best_neg, args.threshold, args.n),
        panel("WORST real - highest scores on label=0 (closest to triggering)",
              worst_neg, args.threshold, args.n),
    ]
    out = stack(panels)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"wrote {args.out} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    main()
