"""Unified split + multi-label dataset for the grok/gemini CNN."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    CLASSES,
    HARD_NEG_DIRS,
    IMAGE_EXTS,
    INPUT_SIZE,
    NEG_DIR,
    POS_DIRS,
    ROI_FRAC,
    SEED,
    SPLITS_PATH,
)


def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


# ---------- split ----------

def make_split(seed: int = SEED) -> Dict[str, List[dict]]:
    """Each entry: {path, labels: {grok: 0/1, gemini: 0/1}, source: 'pos_grok'/'pos_gemini'/'neg'/'hard_neg'}."""
    rng = random.Random(seed)

    def _split_70_15_15(items):
        items = list(items); rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    train, val, test = [], [], []

    # Positives per class
    for cls, d in POS_DIRS.items():
        files = list_images(d)
        tr, va, te = _split_70_15_15(files)
        for split_name, bag in zip((tr, va, te), (train, val, test)):
            for p in split_name:
                labels = {c: 0 for c in CLASSES}
                labels[cls] = 1
                bag.append({"path": str(p), "labels": labels, "source": f"pos_{cls}"})

    # Plain negatives
    files = list_images(NEG_DIR)
    tr, va, te = _split_70_15_15(files)
    for split_files, bag in zip((tr, va, te), (train, val, test)):
        for p in split_files:
            bag.append({"path": str(p), "labels": {c: 0 for c in CLASSES}, "source": "neg"})

    # Hard negatives
    for d in HARD_NEG_DIRS:
        files = list_images(d)
        if not files:
            continue
        tr, va, te = _split_70_15_15(files)
        for split_files, bag in zip((tr, va, te), (train, val, test)):
            for p in split_files:
                bag.append({"path": str(p), "labels": {c: 0 for c in CLASSES},
                             "source": f"hard_neg_{d.name}"})

    return {"train": train, "val": val, "test": test}


def save_split(split, path: Path = SPLITS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: Path = SPLITS_PATH):
    with open(path) as f:
        return json.load(f)


# ---------- crop + augmentation ----------

def br_roi(img_bgr: np.ndarray, frac: float = ROI_FRAC) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    rh = max(1, int(h * frac))
    rw = max(1, int(w * frac))
    return img_bgr[h - rh:, w - rw:]


def random_crop_and_resize(img_bgr: np.ndarray, target: int, *, augment: bool, rng: random.Random) -> np.ndarray:
    """BR-anchored crop: every crop is guaranteed to contain the bottom-right corner of
    the original image (where the watermark lives). Augmentation only varies how much
    surrounding context is visible (zoom level), not whether the watermark is in frame."""
    h, w = img_bgr.shape[:2]
    if augment:
        # Vary the crop size (zoom): anywhere from ROI_FRAC to ROI_FRAC*1.6 of the image.
        # All crops still anchored at the bottom-right corner.
        frac = ROI_FRAC * (1.0 + rng.uniform(-0.20, 0.60))   # ~0.20 .. 0.40
    else:
        frac = ROI_FRAC
    crop_h = max(2, int(h * frac))
    crop_w = max(2, int(w * frac))
    crop = img_bgr[h - crop_h:, w - crop_w:]
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


def color_jitter(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    f = img_bgr.astype(np.float32)
    f *= 1.0 + rng.uniform(-0.20, 0.20)             # brightness
    mean = f.mean(axis=(0, 1), keepdims=True)
    f = mean + (f - mean) * (1.0 + rng.uniform(-0.20, 0.20))   # contrast
    return np.clip(f, 0, 255).astype(np.uint8)


def maybe_jpeg(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    q = rng.randint(60, 95)
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img_bgr
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def small_rotation(img_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Tiny rotation kept for natural-looking jitter; ±2.5° so watermark stays visually
    intact (a 5° rotation can shift the BR corner enough to push the wordmark out of frame)."""
    angle = rng.uniform(-2.5, 2.5)
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------- dataset ----------

class WatermarkDataset(Dataset):
    def __init__(self, entries, *, augment: bool, input_size: int = INPUT_SIZE, seed: int = 0):
        self.entries = entries
        self.augment = augment
        self.input_size = input_size
        self._rng_seed = seed

    def __len__(self):
        return len(self.entries)

    def _load_bgr(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return img

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = self._load_bgr(e["path"])
        # Per-sample RNG so workers + epoch reshuffles still see varied augmentation
        rng = random.Random((self._rng_seed * 31 + idx) ^ 0x9E3779B1)
        crop = random_crop_and_resize(img, self.input_size, augment=self.augment, rng=rng)
        if self.augment:
            crop = small_rotation(crop, rng)
            crop = color_jitter(crop, rng)
            if rng.random() < 0.5:
                crop = maybe_jpeg(crop, rng)
        # to CHW float32 in [0,1], then ImageNet normalize. The MobileNetV3
        # backbone expects RGB; OpenCV reads BGR — convert before normalizing.
        from .model import NORM_MEAN, NORM_STD
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = (x - np.array(NORM_MEAN, dtype=np.float32)) / np.array(NORM_STD, dtype=np.float32)
        x = np.transpose(x, (2, 0, 1)).copy()
        y = np.array([e["labels"][c] for c in CLASSES], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y), e["path"]
