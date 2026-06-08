"""Phase 0: dataset inventory + sample debug thumbnails + write deterministic split."""
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .config import DEBUG_DIR, NEG_DIR, POS_DIR, ROI_FRAC, SEED
from .data import list_images, make_split, save_split


def inventory(d: Path):
    files = list_images(d)
    exts = Counter(p.suffix.lower() for p in files)
    return files, exts


def thumb_with_roi(img_bgr, roi_frac=ROI_FRAC, max_side=400):
    h, w = img_bgr.shape[:2]
    rh = int(h * roi_frac)
    rw = int(w * roi_frac)
    out = img_bgr.copy()
    cv2.rectangle(out, (w - rw, h - rh), (w - 1, h - 1), (0, 255, 0), 2)
    s = max_side / max(h, w)
    if s < 1.0:
        out = cv2.resize(out, (int(w * s), int(h * s)))
    return out


def main():
    out_dir = DEBUG_DIR / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    summary_lines = []

    for label, d in [("pos_grok", POS_DIR), ("neg_real", NEG_DIR)]:
        files, exts = inventory(d)
        print(f"[{label}] {d}: {len(files)} files, exts={dict(exts)}")
        summary_lines.append(f"[{label}] {d}: {len(files)} files, exts={dict(exts)}")
        sample = rng.sample(files, k=min(5, len(files)))
        for i, p in enumerate(sample):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  could not read {p}")
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_lum = float(gray.mean())
            rh, rw = int(h * ROI_FRAC), int(w * ROI_FRAC)
            roi = gray[h - rh:, w - rw:]
            roi_lum = float(roi.mean())
            line = f"  {label} {i}: {p.name}  {w}x{h}  mean_lum={mean_lum:.1f} roi_lum={roi_lum:.1f}"
            print(line)
            summary_lines.append(line)
            cv2.imwrite(str(out_dir / f"{label}_{i}_{p.stem}.jpg"), thumb_with_roi(img))

    split = make_split(SEED)
    save_split(split)
    print(f"\nsplit saved to splits/split.json")

    for name in ("train", "val", "test"):
        items = split[name]
        n_pos = sum(1 for _, l in items if l == 1)
        n_neg = sum(1 for _, l in items if l == 0)
        line = f"  {name}: total={len(items)} pos={n_pos} neg={n_neg}"
        print(line)
        summary_lines.append(line)

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
