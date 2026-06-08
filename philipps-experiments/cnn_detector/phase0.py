"""Phase 0: build unified split, sanity-check counts + an augmented batch montage."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from .config import CLASSES, DEBUG_DIR, INPUT_SIZE, SEED
from .data import WatermarkDataset, load_split, make_split, save_split


def main():
    out = DEBUG_DIR / "phase0"
    out.mkdir(parents=True, exist_ok=True)

    split = make_split(SEED)
    save_split(split)
    print(f"split saved to splits/cnn_split.json")

    for name in ("train", "val", "test"):
        items = split[name]
        sources = Counter(e["source"] for e in items)
        n_grok = sum(1 for e in items if e["labels"]["grok"] == 1)
        n_gem = sum(1 for e in items if e["labels"]["gemini"] == 1)
        n_neg = sum(1 for e in items if all(v == 0 for v in e["labels"].values()))
        print(f"\n{name}: total={len(items)} grok={n_grok} gemini={n_gem} all-neg={n_neg}")
        for s, c in sorted(sources.items()):
            print(f"  {s}: {c}")

    # One augmented batch montage so we can eyeball the augmentation pipeline.
    ds = WatermarkDataset(split["train"], augment=True, input_size=INPUT_SIZE, seed=SEED)
    loader = DataLoader(ds, batch_size=24, shuffle=True, num_workers=0)
    batch_x, batch_y, batch_paths = next(iter(loader))
    montage_imgs = []
    for i in range(min(24, batch_x.shape[0])):
        x = batch_x[i].numpy()
        x = (x * 0.5 + 0.5) * 255.0
        x = np.clip(np.transpose(x, (1, 2, 0)), 0, 255).astype(np.uint8)
        # annotate label
        label_str = " / ".join(f"{c}={int(batch_y[i, j])}" for j, c in enumerate(CLASSES))
        annotated = x.copy()
        cv2.putText(annotated, label_str, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)
        montage_imgs.append(annotated)

    cols = 6
    rows = (len(montage_imgs) + cols - 1) // cols
    canvas = np.full((rows * INPUT_SIZE, cols * INPUT_SIZE, 3), 30, dtype=np.uint8)
    for i, img in enumerate(montage_imgs):
        r, c = divmod(i, cols)
        canvas[r * INPUT_SIZE:(r + 1) * INPUT_SIZE, c * INPUT_SIZE:(c + 1) * INPUT_SIZE] = img
    cv2.imwrite(str(out / "augmented_batch.jpg"), canvas)
    print(f"\nwrote {out / 'augmented_batch.jpg'}")


if __name__ == "__main__":
    main()
