"""For each FN in debug/grok/full_eval/false_negatives, build a readable
diagnostic. Per FN row:
    [full image with ROI 18% drawn in green]
    [ROI 18%]      [Canny of 18% ROI]
    [ROI 35%]      [Canny of 35% ROI]
plus three test scores in the label:
    locked      - the shipped detector score (0.85..1.15 scales, real templates)
    real-wide   - locked detector but with scales 0.3..1.5 on the real templates
    real-roi35  - real-wide AND ROI 35% (tests ROI-too-tight hypothesis)
Output: 5 batches of ~12 FNs each into debug/grok/full_eval/fn_diagnostic_NN.jpg"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from .config import CANNY_HI, CANNY_LO, ROI_FRAC
from .detect import (
    DEFAULT_NARROW_SCALES,
    ScoreConfig,
    default_config_and_template,
    load_template_gray,
    load_templates,
    real_template_paths,
    score,
)


WIDE_SCALES = (0.3, 0.45, 0.6, 0.75, 0.85, 0.92, 1.0, 1.08, 1.15, 1.3, 1.5)


def br_roi(img_bgr, frac):
    h, w = img_bgr.shape[:2]
    rh, rw = max(1, int(h * frac)), max(1, int(w * frac))
    return img_bgr[h - rh:, w - rw:], (h - rh, w - rw, h, w)


def fit_thumb(img, max_w, max_h):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    if s < 1.0:
        return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def pad_to(img, target_w, target_h, color=(28, 28, 28)):
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), color, np.uint8)
    canvas[:h, :w] = img
    return canvas


def label_strip(width, lines, height_per_line=18, fg=(220, 220, 230)):
    h = height_per_line * len(lines) + 6
    s = np.full((h, width, 3), 18, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(s, t, (4, height_per_line * (i + 1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, fg, 1, cv2.LINE_AA)
    return s


def main():
    rows = list(csv.DictReader(open("debug/grok/full_eval/false_negatives/_index.csv")))
    print(f"{len(rows)} false negatives")

    cfg_locked, primary = default_config_and_template()
    real_paths = real_template_paths()
    real_tpls = load_templates(real_paths)
    print(f"locked: ROI={cfg_locked.roi_frac}  scales={cfg_locked.scales}  "
          f"templates={len(real_tpls)} real")

    cfg_widescales = ScoreConfig(roi_frac=ROI_FRAC, scales=WIDE_SCALES, extra_templates=real_tpls[1:])
    cfg_widescales_roi35 = ScoreConfig(roi_frac=0.35, scales=WIDE_SCALES, extra_templates=real_tpls[1:])
    cfg_locked_roi35 = ScoreConfig(roi_frac=0.35, scales=cfg_locked.scales, extra_templates=real_tpls[1:])

    # Layout
    FULL_W = 480
    ROI_W = 220
    EDGE_W = 220
    ROW_H = 220
    GAP = 6
    LABEL_LINES = 3
    LABEL_H = 18 * LABEL_LINES + 6

    cell_w = FULL_W + GAP + (ROI_W + GAP + EDGE_W) * 2
    cell_h = ROW_H + LABEL_H

    # Per-row builder
    def build_row(r):
        path = r["path"]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]

        roi18, (y0a, x0a, y1a, x1a) = br_roi(img, 0.18)
        roi35, (y0b, x0b, y1b, x1b) = br_roi(img, 0.35)

        edges18 = cv2.Canny(cv2.cvtColor(roi18, cv2.COLOR_BGR2GRAY), CANNY_LO, CANNY_HI)
        edges35 = cv2.Canny(cv2.cvtColor(roi35, cv2.COLOR_BGR2GRAY), CANNY_LO, CANNY_HI)

        s_locked = float(score(img, primary, cfg_locked))
        s_wide = float(score(img, primary, cfg_widescales))
        s_wide35 = float(score(img, primary, cfg_widescales_roi35))
        s_lock35 = float(score(img, primary, cfg_locked_roi35))

        # Annotated full image: green rectangle around 18% ROI, blue around 35% ROI
        ann = img.copy()
        thick = max(2, int(min(W, H) / 250))
        cv2.rectangle(ann, (x0b, y0b), (x1b - 1, y1b - 1), (255, 120, 0), thick)   # blue: 35%
        cv2.rectangle(ann, (x0a, y0a), (x1a - 1, y1a - 1), (0, 255, 0), thick)     # green: 18%

        full_thumb = pad_to(fit_thumb(ann, FULL_W, ROW_H), FULL_W, ROW_H)
        roi18_thumb = pad_to(fit_thumb(roi18, ROI_W, ROW_H), ROI_W, ROW_H)
        edges18_thumb = pad_to(fit_thumb(cv2.cvtColor(edges18, cv2.COLOR_GRAY2BGR), EDGE_W, ROW_H), EDGE_W, ROW_H)
        roi35_thumb = pad_to(fit_thumb(roi35, ROI_W, ROW_H), ROI_W, ROW_H)
        edges35_thumb = pad_to(fit_thumb(cv2.cvtColor(edges35, cv2.COLOR_GRAY2BGR), EDGE_W, ROW_H), EDGE_W, ROW_H)

        gap_col = np.full((ROW_H, GAP, 3), 18, np.uint8)
        roi_block_18 = np.concatenate([roi18_thumb, gap_col, edges18_thumb], axis=1)
        roi_block_35 = np.concatenate([roi35_thumb, gap_col, edges35_thumb], axis=1)

        row_top = np.concatenate([full_thumb, gap_col, roi_block_18, gap_col, roi_block_35], axis=1)

        l1 = (f"#{int(r['rank']):>2}  s_locked={s_locked:.3f}  "
              f"split={r['split']}  {Path(path).name[:60]}")
        l2 = f"     img={W}x{H}   ROI18%={x1a-x0a}x{y1a-y0a}   ROI35%={x1b-x0b}x{y1b-y0b}"
        # Highlight any score that exceeds the locked threshold (0.27).
        def fmt(s, *, hl=False):
            return ("**" + f"{s:.3f}" + "**") if hl else f"{s:.3f}"
        thr = 0.27
        l3 = (f"     locked={fmt(s_locked, hl=s_locked>=thr)}   "
              f"wide-scales={fmt(s_wide, hl=s_wide>=thr)}   "
              f"locked+ROI35={fmt(s_lock35, hl=s_lock35>=thr)}   "
              f"wide+ROI35={fmt(s_wide35, hl=s_wide35>=thr)}   thr={thr}")

        label = label_strip(row_top.shape[1], [l1, l2, l3])
        full_row = np.concatenate([row_top, label], axis=0)
        return full_row, {
            "rank": int(r["rank"]),
            "name": Path(path).name,
            "W": W, "H": H,
            "s_locked": s_locked, "s_wide": s_wide,
            "s_lock35": s_lock35, "s_wide35": s_wide35,
        }

    diagnostics = []
    out_dir = Path("debug/grok/full_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    BATCH = 12
    batch_idx = 0
    current_rows = []

    def flush():
        nonlocal batch_idx, current_rows
        if not current_rows:
            return
        montage = np.concatenate(current_rows, axis=0)
        path = out_dir / f"fn_diagnostic_{batch_idx:02d}.jpg"
        cv2.imwrite(str(path), montage)
        print(f"  wrote {path}  ({montage.shape[1]}x{montage.shape[0]})")
        batch_idx += 1
        current_rows = []

    for r in rows:
        result = build_row(r)
        if result is None:
            continue
        row_img, diag = result
        current_rows.append(row_img)
        diagnostics.append(diag)
        if len(current_rows) >= BATCH:
            flush()
    flush()

    # Summary
    thr = 0.27
    print("\nWould a different config have caught these FNs? (locked threshold = 0.27)")
    s_locked_pass = sum(1 for d in diagnostics if d["s_locked"] >= thr)
    s_wide_pass = sum(1 for d in diagnostics if d["s_wide"] >= thr)
    s_lock35_pass = sum(1 for d in diagnostics if d["s_lock35"] >= thr)
    s_wide35_pass = sum(1 for d in diagnostics if d["s_wide35"] >= thr)
    print(f"  locked (current ship): {s_locked_pass}/{len(diagnostics)} above thr (sanity = 0)")
    print(f"  wide-scales,  ROI 18%: {s_wide_pass}/{len(diagnostics)}")
    print(f"  locked,       ROI 35%: {s_lock35_pass}/{len(diagnostics)}")
    print(f"  wide-scales,  ROI 35%: {s_wide35_pass}/{len(diagnostics)}")

    # Per-image categorisation
    cats = {"locked_only": [], "needs_wide": [], "needs_roi": [], "needs_both": [], "still_misses": []}
    for d in diagnostics:
        if d["s_locked"] >= thr:
            cats["locked_only"].append(d)
        elif d["s_wide"] >= thr and d["s_lock35"] < thr:
            cats["needs_wide"].append(d)
        elif d["s_lock35"] >= thr and d["s_wide"] < thr:
            cats["needs_roi"].append(d)
        elif d["s_wide35"] >= thr:
            cats["needs_both"].append(d)
        else:
            cats["still_misses"].append(d)

    print(f"\nCategorisation:")
    for k, items in cats.items():
        print(f"  {k}: {len(items)}")
        for d in items[:8]:
            print(f"    #{d['rank']:>2} img={d['W']}x{d['H']:<5} "
                  f"locked={d['s_locked']:.3f} wide={d['s_wide']:.3f} "
                  f"lock35={d['s_lock35']:.3f} wide35={d['s_wide35']:.3f}  {d['name'][:55]}")
        if len(items) > 8:
            print(f"    ... and {len(items)-8} more")


if __name__ == "__main__":
    main()
