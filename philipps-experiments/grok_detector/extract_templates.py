"""Phase 3: extract real-sample watermark templates from train positives.

Strategy:
- Use the master template; on each train positive, run multi-scale Canny matching
  with a wide scale range (incl. small scales not in spec ladder) to find the best
  (scale, location). Keep top N by NCC score.
- Crop the matched ROI region (slightly padded), composite to grayscale.
- Save as PNGs under grok_detector/real_templates/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .config import CANNY_HI, CANNY_LO, POS_DIR, ROI_FRAC
from .data import list_images
from .detect import load_template_gray


WIDE_SCALES = tuple(np.round(np.linspace(0.08, 1.0, 24), 3).tolist())


def best_match(roi_edges, tpl_edges):
    rh, rw = roi_edges.shape[:2]
    th0, tw0 = tpl_edges.shape[:2]
    best = (-1.0, None, None)  # score, (x,y,w,h), scale
    for s in WIDE_SCALES:
        th, tw = max(1, int(round(th0 * s))), max(1, int(round(tw0 * s)))
        if th >= rh or tw >= rw:
            continue
        tpl_s = cv2.resize(tpl_edges, (tw, th), interpolation=cv2.INTER_AREA)
        if tpl_s.sum() == 0:
            continue
        res = cv2.matchTemplate(roi_edges, tpl_s, cv2.TM_CCOEFF_NORMED)
        _, mx, _, ml = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), (ml[0], ml[1], tw, th), float(s))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="number of real templates to keep")
    ap.add_argument("--pad", type=int, default=2, help="pixels of padding around the matched region")
    ap.add_argument("--out", type=str, default="grok_detector/real_templates")
    ap.add_argument("--min-score", type=float, default=0.40,
                    help="only consider matches with NCC >= this on the master template")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = load_template_gray()
    tpl_edges = cv2.Canny(template, CANNY_LO, CANNY_HI)

    # restrict to train split positives
    from .data import load_split
    split = load_split()
    train_pos = [p for p, l in split["train"] if l == 1]

    cands = []  # (score, scale, src_path, crop_gray)
    for path in train_pos:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        rh, rw = int(h * ROI_FRAC), int(w * ROI_FRAC)
        roi = img[h - rh:, w - rw:]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, CANNY_LO, CANNY_HI)
        sc, box, scale = best_match(roi_edges, tpl_edges)
        if box is None:
            continue
        x, y, bw, bh = box
        # slight pad and clamp
        x0 = max(0, x - args.pad); y0 = max(0, y - args.pad)
        x1 = min(rw, x + bw + args.pad); y1 = min(rh, y + bh + args.pad)
        crop = roi_gray[y0:y1, x0:x1]
        cands.append((sc, scale, path, crop))

    cands = [c for c in cands if c[0] >= args.min_score]
    cands.sort(key=lambda t: -t[0])
    print(f"{len(cands)} candidates above min-score={args.min_score}")

    # Real Grok watermarks render at a near-constant scale (~0.20 of master).
    # Pick top-N by NCC, requiring distinct source images so multi-template
    # captures different background/edge artifacts rather than the same image
    # at multiple offsets.
    chosen = []
    seen_paths = set()
    for sc, scale, p, crop in cands:
        if p in seen_paths:
            continue
        seen_paths.add(p)
        chosen.append((sc, scale, p, crop))
        if len(chosen) >= args.n:
            break

    print(f"selected {len(chosen)} real templates:")
    for i, (sc, scale, p, crop) in enumerate(chosen):
        out_path = out_dir / f"real_template_{i:02d}.png"
        cv2.imwrite(str(out_path), crop)
        print(f"  [{i}] score={sc:.3f} scale={scale:.2f} size={crop.shape} src={Path(p).name} -> {out_path}")


if __name__ == "__main__":
    main()
