"""
watermark_aug.py — Watermark augmentation for synthetic training data.

Applies known watermarks to clean images with realistic blending.
Supports both realistic (fixed position) and arbitrary position modes
for localization-based training.

Watermark properties (derived from real image analysis):
- dalle:         Opaque 80x16 color strip, flush bottom-right
- gemini:        White 4-point star, ~45% opacity, 22x22 on 1024px, bottom-right
- grok:          White text+logo, ~45% opacity, ~150x50 on 1024px, bottom-right
- minimax_hailuo: White text, ~45% opacity, ~200x25 on 1024px, bottom-right
- text_tpdne:    White text, ~45% opacity, ~330x25 on 1024px, top center
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path


class WatermarkAugmenter:
    """Apply synthetic watermarks to clean images."""

    def __init__(self, templates_dir: str = "images/watermarks"):
        self.templates_dir = Path(templates_dir)
        self.templates = {}
        self._load_templates()

    def _load_template(self, name, filename):
        path = self.templates_dir / filename
        if path.exists():
            tmpl = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if tmpl is not None:
                self.templates[name] = tmpl

    def _load_templates(self):
        self._load_template("dalle", "dalle_watermark.png")
        self._load_template("gemini", "gemini_watermark.png")
        self._load_template("grok", "grok_watermark.png")
        self._load_template("minimax_hailuo", "hailuoaixminimax_watermark.png")
        self._load_template("text_tpdne", "this-person-does-not-exist_watermark.png")

    def _alpha_composite(self, result, tmpl_bgr, tmpl_alpha, x1, y1):
        """Alpha-composite a watermark onto the image at (x1, y1)."""
        h, w = result.shape[:2]
        th, tw = tmpl_bgr.shape[:2]
        # Clip to image bounds
        sx = max(0, -x1)
        sy = max(0, -y1)
        ex = min(tw, w - x1)
        ey = min(th, h - y1)
        if ex <= sx or ey <= sy:
            return
        ix1, iy1 = x1 + sx, y1 + sy
        ix2, iy2 = x1 + ex, y1 + ey
        roi = result[iy1:iy2, ix1:ix2].astype(np.float32)
        fg = tmpl_bgr[sy:ey, sx:ex].astype(np.float32)
        a = tmpl_alpha[sy:ey, sx:ex, np.newaxis]
        result[iy1:iy2, ix1:ix2] = (roi * (1 - a) + fg * a).astype(np.uint8)

    def _get_position(self, h, w, wm_h, wm_w, location, rng):
        """Compute paste position. location='realistic' or 'random'."""
        if location == "random":
            x = rng.integers(0, max(1, w - wm_w))
            y = rng.integers(0, max(1, h - wm_h))
            return x, y
        # Default: realistic position based on watermark type is handled per-method
        return None, None

    def apply_dalle(self, img, rng=None, position="realistic"):
        if "dalle" not in self.templates:
            return img
        rng = rng or np.random.default_rng()
        result = img.copy()
        h, w = result.shape[:2]
        tmpl = self.templates["dalle"]
        scale = w / 1024.0
        tw = max(int(80 * scale), 20)
        th = max(int(16 * scale), 4)
        resized = cv2.resize(tmpl[:, :, :3], (tw, th))

        if position == "random":
            x, y = self._get_position(h, w, th, tw, "random", rng)
        else:
            x = w - tw - rng.integers(0, max(1, int(3 * scale)))
            y = h - th - rng.integers(0, max(1, int(2 * scale)))

        # Opaque paste
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + tw), min(h, y + th)
        pw, ph = x2 - x1, y2 - y1
        if pw > 0 and ph > 0:
            result[y1:y2, x1:x2] = cv2.resize(resized, (pw, ph))
        return result

    def apply_gemini(self, img, rng=None, position="realistic"):
        if "gemini" not in self.templates:
            return img
        rng = rng or np.random.default_rng()
        result = img.copy()
        h, w = result.shape[:2]
        tmpl = self.templates["gemini"]
        star_size = max(int(22 * (w / 1024.0)), 8)
        resized = cv2.resize(tmpl, (star_size, star_size), interpolation=cv2.INTER_AREA)

        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        bgr = np.full_like(resized[:, :, :3], 255, dtype=np.float32)
        opacity = 0.45 + rng.uniform(-0.08, 0.08)
        alpha = alpha * opacity

        if position == "random":
            x, y = self._get_position(h, w, star_size, star_size, "random", rng)
        else:
            x = w - rng.integers(25, 55) - star_size // 2
            y = h - rng.integers(25, 45) - star_size // 2

        self._alpha_composite(result, bgr, alpha, x, y)
        return result

    def apply_grok(self, img, rng=None, position="realistic"):
        if "grok" not in self.templates:
            return img
        rng = rng or np.random.default_rng()
        result = img.copy()
        h, w = result.shape[:2]
        tmpl = self.templates["grok"]
        scale = w / 1024.0
        tw = max(int(150 * scale), 40)
        th = max(int(50 * scale), 15)
        resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)

        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        bgr = np.full((th, tw, 3), 255, dtype=np.float32)
        opacity = 0.45 + rng.uniform(-0.08, 0.08)
        alpha = alpha * opacity

        if position == "random":
            x, y = self._get_position(h, w, th, tw, "random", rng)
        else:
            x = w - tw - rng.integers(5, 25)
            y = h - th - rng.integers(5, 25)

        self._alpha_composite(result, bgr, alpha, x, y)
        return result

    def apply_minimax(self, img, rng=None, position="realistic"):
        if "minimax_hailuo" not in self.templates:
            return img
        rng = rng or np.random.default_rng()
        result = img.copy()
        h, w = result.shape[:2]
        tmpl = self.templates["minimax_hailuo"]
        scale = w / 1024.0
        tw = max(int(200 * scale), 50)
        th = max(int(25 * scale), 8)
        resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)

        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        bgr = np.full((th, tw, 3), 255, dtype=np.float32)
        opacity = 0.45 + rng.uniform(-0.08, 0.08)
        alpha = alpha * opacity

        if position == "random":
            x, y = self._get_position(h, w, th, tw, "random", rng)
        else:
            x = w - tw - rng.integers(5, 20)
            y = h - th - rng.integers(5, 20)

        self._alpha_composite(result, bgr, alpha, x, y)
        return result

    def apply_tpdne(self, img, rng=None, position="realistic"):
        if "text_tpdne" not in self.templates:
            return img
        rng = rng or np.random.default_rng()
        result = img.copy()
        h, w = result.shape[:2]
        tmpl = self.templates["text_tpdne"]
        scale = w / 1024.0
        tw = max(int(330 * scale), 80)
        th = max(int(25 * scale), 8)
        resized = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)

        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        # TPDNE template has dark text — on real images the text is WHITE
        bgr = np.full((th, tw, 3), 255, dtype=np.float32)
        opacity = 0.45 + rng.uniform(-0.08, 0.08)
        alpha = alpha * opacity

        if position == "random":
            x, y = self._get_position(h, w, th, tw, "random", rng)
        else:
            x = (w - tw) // 2 + rng.integers(-20, 20)
            y = rng.integers(3, 15)

        self._alpha_composite(result, bgr, alpha, x, y)
        return result

    def augment(self, img, watermark_type, rng=None, position="realistic"):
        """Apply a watermark. position='realistic' or 'random'."""
        rng = rng or np.random.default_rng()
        methods = {
            "dalle": self.apply_dalle,
            "gemini": self.apply_gemini,
            "grok": self.apply_grok,
            "minimax_hailuo": self.apply_minimax,
            "text_tpdne": self.apply_tpdne,
        }
        if watermark_type not in methods:
            raise ValueError(f"Unknown watermark type: {watermark_type}")
        return methods[watermark_type](img, rng, position)

    def augment_random_position(self, img, watermark_type, rng=None):
        """Apply watermark at a random position (for localization training)."""
        return self.augment(img, watermark_type, rng, position="random")


if __name__ == "__main__":
    import os
    aug = WatermarkAugmenter()
    clean_files = sorted(os.listdir("images/no_watermark/"))
    if clean_files:
        clean_path = f"images/no_watermark/{clean_files[0]}"
        clean = cv2.imread(clean_path)
        if clean is None:
            clean = cv2.cvtColor(np.array(Image.open(clean_path).convert("RGB")), cv2.COLOR_RGB2BGR)

        for wm_type in ["dalle", "gemini", "grok", "minimax_hailuo", "text_tpdne"]:
            result = aug.augment(clean, wm_type)
            out = f"/tmp/test_aug_{wm_type}.png"
            cv2.imwrite(out, result)
            print(f"Wrote {out} ({result.shape[1]}x{result.shape[0]})")

        # Also test random positions
        for wm_type in ["gemini", "grok"]:
            rng = np.random.default_rng(123)
            result = aug.augment_random_position(clean, wm_type, rng)
            out = f"/tmp/test_aug_{wm_type}_random.png"
            cv2.imwrite(out, result)
            print(f"Wrote {out} (random position)")
