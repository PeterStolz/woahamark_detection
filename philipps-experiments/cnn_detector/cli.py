"""CLI: python -m cnn_detector.cli <image_path> [--ckpt path]"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import torch

from .config import CHECKPOINTS_DIR, CLASSES, INPUT_SIZE
from .data import random_crop_and_resize
from .model import WatermarkCNN
from .train import pick_device, sigmoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    ap.add_argument("--ckpt", default=str(CHECKPOINTS_DIR / "cnn_v3.pt"))
    args = ap.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read {args.image_path}", file=sys.stderr)
        sys.exit(1)

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = WatermarkCNN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    thresholds = ckpt["thresholds"]

    import random
    from .model import NORM_MEAN, NORM_STD
    crop = random_crop_and_resize(img, INPUT_SIZE, augment=False, rng=random.Random(0))
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))[None]
    x_t = torch.from_numpy(x).to(device)
    with torch.no_grad():
        logits = model(x_t).cpu().numpy()[0]
    scores = sigmoid(logits)

    print(f"image: {args.image_path}")
    for k, c in enumerate(CLASSES):
        thr = float(thresholds[c])
        pred = "POS" if scores[k] >= thr else "NEG"
        print(f"  {c}: score={scores[k]:.4f} threshold={thr:.4f} -> {pred}")


if __name__ == "__main__":
    main()
