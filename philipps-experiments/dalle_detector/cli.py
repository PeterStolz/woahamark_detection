"""CLI: python -m dalle_detector.cli <image_path> [--threshold X]"""
from __future__ import annotations

import argparse
import sys

import cv2

from .detect import DEFAULT_THRESHOLD, classify, default_config_and_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read {args.image_path}", file=sys.stderr)
        sys.exit(1)
    cfg, template = default_config_and_template()
    s, pred = classify(img, template, args.threshold, cfg)
    label = "dalle_watermark" if pred == 1 else "no_watermark"
    print(f"score={s:.4f} threshold={args.threshold:.4f} label={label}")


if __name__ == "__main__":
    main()
