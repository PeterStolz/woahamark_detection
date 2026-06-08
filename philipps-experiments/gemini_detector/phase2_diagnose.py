"""Phase 2 diagnose: ROI/Canny montages of all positives + a wide-scale probe."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import CANNY_HI, CANNY_LO, DEBUG_DIR, ROI_FRAC
from .data import load_split
from .detect import load_template_gray


def roi_canny(img):
    h, w = img.shape[:2]
    rh, rw = max(1, int(h * ROI_FRAC)), max(1, int(w * ROI_FRAC))
    roi = img[h - rh:, w - rw:]
    edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), CANNY_LO, CANNY_HI)
    return roi, edges


def panel(rows, max_n=24, tile_w=180, tile_h=180, scores=None, label_prefix=""):
    cols = 6
    n_rows = (min(len(rows), max_n) + cols - 1) // cols
    cell_w, cell_h = tile_w * 2 + 8, tile_h + 24
    canvas = np.full((n_rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    for i, p in enumerate(rows[:max_n]):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        roi, edges = roi_canny(img)
        roi_r = cv2.resize(roi, (tile_w, tile_h))
        edges_r = cv2.cvtColor(cv2.resize(edges, (tile_w, tile_h)), cv2.COLOR_GRAY2BGR)
        cell = np.concatenate([roi_r, np.full((tile_h, 8, 3), 60, np.uint8), edges_r], axis=1)
        rr, cc = divmod(i, cols)
        canvas[rr * cell_h:rr * cell_h + tile_h, cc * cell_w:cc * cell_w + cell_w] = cell
        text = Path(p).name[:24]
        if scores is not None:
            text = f"{text} s={scores[i]:.2f}"
        cv2.putText(canvas, label_prefix + text,
                    (cc * cell_w + 2, rr * cell_h + tile_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def wide_scale_probe(roi_edges, tpl_edges, scales):
    """Return (best_score, best_scale, best_size_wh)."""
    rh, rw = roi_edges.shape[:2]
    th0, tw0 = tpl_edges.shape[:2]
    best = (-1.0, None, None)
    for s in scales:
        th = max(1, int(round(th0 * s)))
        tw = max(1, int(round(tw0 * s)))
        if th >= rh or tw >= rw:
            continue
        tpl_s = cv2.resize(tpl_edges, (tw, th), interpolation=cv2.INTER_AREA)
        if tpl_s.sum() == 0:
            continue
        res = cv2.matchTemplate(roi_edges, tpl_s, cv2.TM_CCOEFF_NORMED)
        m = float(res.max())
        if m > best[0]:
            best = (m, s, (tw, th))
    return best


def main():
    out = DEBUG_DIR / "phase2"
    out.mkdir(parents=True, exist_ok=True)
    split = load_split()

    test_pos = [p for p, l in split["test"] if l == 1]
    test_neg = [p for p, l in split["test"] if l == 0]

    cv2.imwrite(str(out / "test_positives.jpg"),
                panel(test_pos, max_n=24))
    cv2.imwrite(str(out / "test_negatives_random.jpg"),
                panel(test_neg[:24], max_n=24))

    # Wide-scale probe — what scale would actually pick up the sparkle?
    template = load_template_gray()
    tpl_edges = cv2.Canny(template, CANNY_LO, CANNY_HI)
    wide = tuple(np.round(np.linspace(0.01, 0.5, 25), 3).tolist())

    print("\nWide-scale probe across train positives (scales 0.01..0.5):")
    print("score  scale  template_w  template_h  src")
    pos_results = []
    train_pos = [p for p, l in split["train"] if l == 1]
    for path in train_pos:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        roi, edges = roi_canny(img)
        sc, scale, size = wide_scale_probe(edges, tpl_edges, wide)
        if size is not None:
            pos_results.append((sc, scale, size, path))
    pos_results.sort(key=lambda r: -r[0])
    for sc, scale, size, p in pos_results[:10]:
        print(f"  {sc:.3f}  {scale:.3f}  {size[0]}x{size[1]}  {Path(p).name}")
    if pos_results:
        scales_only = [r[1] for r in pos_results if r[0] >= 0.30]
        print(f"\n{len(scales_only)} positives matched ≥0.30; "
              f"scales: min={min(scales_only) if scales_only else 'n/a'}, "
              f"max={max(scales_only) if scales_only else 'n/a'}, "
              f"median={np.median(scales_only) if scales_only else 'n/a'}")


if __name__ == "__main__":
    main()
