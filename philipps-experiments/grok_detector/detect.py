"""Single-image scoring + classification for the Grok watermark.

Phase 3: a configurable scoring function that supports several cheap variants
(high-pass pre-filter, distance-transform matching, ROI fraction sweep,
multi-template). Defaults preserve the Phase 1 baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import CANNY_HI, CANNY_LO, ROI_FRAC, SCALES, TEMPLATE_PATH


# ---------- template loading ----------

def load_template_gray(path: Path = TEMPLATE_PATH) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        a = img[..., 3:4].astype(np.float32) / 255.0
        comp = (bgr * a).astype(np.uint8)
        gray = cv2.cvtColor(comp, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    return gray


def load_templates(paths: List[Path]) -> List[np.ndarray]:
    return [load_template_gray(p) for p in paths]


def real_template_paths(root: Optional[Path] = None) -> List[Path]:
    if root is None:
        root = Path(__file__).resolve().parent / "real_templates"
    return sorted(root.glob("real_template_*.png"))


# ---------- final shipped config (Phase 3 winner: V8_realonly_narrow) ----------

# Real-sample templates capture the watermark at its rendered native size,
# so we don't need the master template here and we don't need wide scales.
DEFAULT_NARROW_SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)
DEFAULT_THRESHOLD = 0.27   # tuned on val; do not retune on test


def default_config_and_template() -> tuple["ScoreConfig", np.ndarray]:
    """Returns (cfg, primary_template) for the locked V8 configuration.

    Loads real-sample crops from grok_detector/real_templates/. The first one
    is used as the primary template arg to score(); the rest go via cfg.extra_templates.
    Falls back to the master template if no real crops are present.
    """
    paths = real_template_paths()
    if paths:
        tpls = load_templates(paths)
        primary, extras = tpls[0], tpls[1:]
    else:
        primary, extras = load_template_gray(), []
    cfg = ScoreConfig(scales=DEFAULT_NARROW_SCALES, extra_templates=extras)
    return cfg, primary


# ---------- ROI ----------

def crop_br_roi(image_bgr: np.ndarray, roi_frac: float = ROI_FRAC) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    rh = max(1, int(h * roi_frac))
    rw = max(1, int(w * roi_frac))
    return image_bgr[h - rh:, w - rw:]


def high_pass(gray: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Subtract Gaussian-blurred copy and shift back to mid-gray for Canny."""
    f = gray.astype(np.float32)
    blur = cv2.GaussianBlur(f, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    hp = f - blur
    return np.clip(hp + 128.0, 0, 255).astype(np.uint8)


# ---------- scoring config ----------

@dataclass
class ScoreConfig:
    roi_frac: float = ROI_FRAC
    canny_lo: int = CANNY_LO
    canny_hi: int = CANNY_HI
    scales: Tuple[float, ...] = SCALES
    high_pass_sigma: Optional[float] = None     # if set, apply ROI -= GaussianBlur(σ) before Canny
    use_distance_transform: bool = False        # match Canny template against DT of (255-ROI edges)
    extra_templates: List[np.ndarray] = field(default_factory=list)  # multi-template ladder


# ---------- scoring ----------

def _prep_roi_edges(roi_bgr: np.ndarray, cfg: ScoreConfig) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if cfg.high_pass_sigma is not None:
        gray = high_pass(gray, cfg.high_pass_sigma)
    return cv2.Canny(gray, cfg.canny_lo, cfg.canny_hi)


def _score_template(roi_edges: np.ndarray, tpl_edges: np.ndarray,
                    scales: Tuple[float, ...],
                    use_distance_transform: bool) -> float:
    rh, rw = roi_edges.shape[:2]
    th0, tw0 = tpl_edges.shape[:2]
    if use_distance_transform:
        # DT of the inverse of edges (i.e., 0 at edges, growing elsewhere).
        # Lower DT-values where roi has edges; correlating template (binary edges)
        # against the *negative* DT gives a peak where template edges align.
        dt = cv2.distanceTransform(255 - roi_edges, distanceType=cv2.DIST_L2, maskSize=3)
        dt_neg = -dt  # higher when closer to an edge
        target = dt_neg.astype(np.float32)
    else:
        target = roi_edges
    best = -np.inf
    for s in scales:
        th = max(1, int(round(th0 * s)))
        tw = max(1, int(round(tw0 * s)))
        if th >= rh or tw >= rw:
            continue
        tpl_s = cv2.resize(tpl_edges, (tw, th), interpolation=cv2.INTER_AREA)
        if tpl_s.sum() == 0:
            continue
        if not use_distance_transform and roi_edges.sum() == 0:
            continue
        try:
            res = cv2.matchTemplate(target, tpl_s.astype(target.dtype), cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        m = float(res.max())
        if m > best:
            best = m
    return best if np.isfinite(best) else 0.0


def score(image_bgr: np.ndarray,
          template_gray: np.ndarray,
          cfg: Optional[ScoreConfig] = None) -> float:
    """Phase 1-compatible signature; cfg adds Phase 3 knobs."""
    cfg = cfg or ScoreConfig()
    roi = crop_br_roi(image_bgr, cfg.roi_frac)
    roi_edges = _prep_roi_edges(roi, cfg)
    templates_gray = [template_gray] + list(cfg.extra_templates)
    best = 0.0
    for tg in templates_gray:
        tpl_edges = cv2.Canny(tg, cfg.canny_lo, cfg.canny_hi)
        s = _score_template(roi_edges, tpl_edges, cfg.scales, cfg.use_distance_transform)
        if s > best:
            best = s
    return best


def classify(image_bgr: np.ndarray,
             template_gray: np.ndarray,
             threshold: float,
             cfg: Optional[ScoreConfig] = None) -> tuple[float, int]:
    s = score(image_bgr, template_gray, cfg)
    return s, int(s >= threshold)


def score_path(path: str | Path,
               template_gray: Optional[np.ndarray] = None,
               cfg: Optional[ScoreConfig] = None) -> float:
    if template_gray is None:
        template_gray = load_template_gray()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return 0.0
    return score(img, template_gray, cfg)
