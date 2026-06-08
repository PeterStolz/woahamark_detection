"""Color-template-matching detector for the DALL·E 2 horizontal color-bar watermark.

The template is a strip of five highly-saturated colored squares — geometric
edge matching is weak (only the rectangular outlines), but RGB color matching
is decisive because the colors are very specific.

We use cv2.matchTemplate on multi-channel BGR (OpenCV processes channels
independently and sums). A small scale ladder around 1.0 covers normal-size
images; the template is already small (80×16) so we don't need wide scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import ROI_FRAC, SCALES, TEMPLATE_PATH


def load_template_bgr(path: Path = TEMPLATE_PATH) -> np.ndarray:
    """Load the 80x16 template as BGR (drop alpha — it's all 255 anyway)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3].copy()
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def crop_br_roi(image_bgr: np.ndarray, roi_frac: float = ROI_FRAC) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    rh = max(1, int(h * roi_frac))
    rw = max(1, int(w * roi_frac))
    return image_bgr[h - rh:, w - rw:]


@dataclass
class ScoreConfig:
    roi_frac: float = ROI_FRAC
    scales: Tuple[float, ...] = SCALES
    saturation_gate: bool = False
    """If True, zero out ROI pixels whose HSV saturation is below sat_min before matching."""
    sat_min: int = 100


def _maybe_gate(roi_bgr: np.ndarray, cfg: ScoreConfig) -> np.ndarray:
    if not cfg.saturation_gate:
        return roi_bgr
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] >= cfg.sat_min).astype(np.uint8)
    return roi_bgr * mask[..., None]


def score(image_bgr: np.ndarray, template_bgr: np.ndarray,
          cfg: Optional[ScoreConfig] = None) -> float:
    """Multi-channel NCC against the colored-bar template, max over scales."""
    cfg = cfg or ScoreConfig()
    roi = crop_br_roi(image_bgr, cfg.roi_frac)
    roi = _maybe_gate(roi, cfg)
    rh, rw = roi.shape[:2]
    th0, tw0 = template_bgr.shape[:2]

    best = -np.inf
    for s in cfg.scales:
        th = max(1, int(round(th0 * s)))
        tw = max(1, int(round(tw0 * s)))
        if th >= rh or tw >= rw:
            continue
        tpl_s = cv2.resize(template_bgr, (tw, th), interpolation=cv2.INTER_AREA)
        try:
            res = cv2.matchTemplate(roi, tpl_s, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        m = float(res.max())
        if m > best:
            best = m
    return best if np.isfinite(best) else 0.0


def classify(image_bgr, template_bgr, threshold: float, cfg=None):
    s = score(image_bgr, template_bgr, cfg)
    return s, int(s >= threshold)


# Locked default — V0 baseline tuned on val (val score-gap tiebreaker).
DEFAULT_THRESHOLD = 0.59


def default_config_and_template():
    return ScoreConfig(), load_template_bgr()
