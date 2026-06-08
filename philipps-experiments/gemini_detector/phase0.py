"""Phase 0: inventory + sanity thumbnails + deterministic split."""
import random
from collections import Counter
from pathlib import Path

import cv2

from .config import DEBUG_DIR, NEG_DIR, POS_DIR, ROI_FRAC, SEED
from .data import list_images, make_split, save_split


def thumb_with_roi(img_bgr, roi_frac=ROI_FRAC, max_side=400):
    h, w = img_bgr.shape[:2]
    rh, rw = int(h * roi_frac), int(w * roi_frac)
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
    summary = []

    for label, d in [("pos_gemini", POS_DIR), ("neg_real", NEG_DIR)]:
        files = list_images(d)
        exts = Counter(p.suffix.lower() for p in files)
        line = f"[{label}] {d}: {len(files)} files, exts={dict(exts)}"
        print(line); summary.append(line)
        for i, p in enumerate(rng.sample(files, k=min(5, len(files)))):
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rh, rw = int(h * ROI_FRAC), int(w * ROI_FRAC)
            roi_lum = float(gray[h - rh:, w - rw:].mean())
            line = f"  {label} {i}: {p.name}  {w}x{h}  mean_lum={float(gray.mean()):.1f} roi_lum={roi_lum:.1f}"
            print(line); summary.append(line)
            cv2.imwrite(str(out_dir / f"{label}_{i}_{p.stem}.jpg"), thumb_with_roi(img))

    split = make_split(SEED)
    save_split(split)
    print("\nsplit saved to splits/gemini_split.json")
    for name in ("train", "val", "test"):
        items = split[name]
        n_pos = sum(1 for _, l in items if l == 1)
        n_neg = sum(1 for _, l in items if l == 0)
        line = f"  {name}: total={len(items)} pos={n_pos} neg={n_neg}"
        print(line); summary.append(line)

    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
