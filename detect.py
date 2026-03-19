"""
detect.py — Watermark detection pipeline.
THIS FILE IS MODIFIED BY THE AGENT. Everything is fair game.

Required contract:
  - Must define `detect(image_path: str) -> dict`
  - Return dict must contain at minimum: {"binary": "clean" | "watermarked"}
  - Optionally: {"binary": ..., "label": "<class>", "confidence": float}
  - Optional `setup(train_set, templates)` called once before evaluation

The baseline is intentionally naive — a random coin flip.
The agent's job is to replace this with real detection approaches.
"""

import random

# ─────────────────────────────────────────────────────────────
# State (populated by setup())
# ─────────────────────────────────────────────────────────────

TRAIN_SET = []
TEMPLATES = {}


def setup(train_set: list[dict], templates: dict[str, str]):
    """Called once before evaluation. Use this for:
    - Loading/training models
    - Precomputing template features
    - Building reference databases
    - Any one-time expensive work

    Args:
        train_set: list of {"path": str, "label": str, "binary_label": str}
        templates: dict of {template_name: template_path}
    """
    global TRAIN_SET, TEMPLATES
    TRAIN_SET = train_set
    TEMPLATES = templates


def detect(image_path: str) -> dict:
    """Detect watermarks in an image.

    Args:
        image_path: path to the image file

    Returns:
        dict with keys:
            "binary": "clean" or "watermarked"
            "label": specific watermark class (e.g. "dalle", "gemini", "grok", ...)
                     or "clean" if no watermark detected
            "confidence": float 0.0 - 1.0
    """
    # ── BASELINE: random coin flip ──
    # This is the worst possible detector. The agent should replace this
    # with real detection logic. Some approaches to consider:
    # - Image processing (edges, frequency, color analysis)
    # - OCR for text-based watermarks
    # - Template matching against known watermarks
    # - Feature extraction + classification
    # - Ensemble of multiple signals

    if random.random() > 0.5:
        return {"binary": "watermarked", "label": "watermarked", "confidence": 0.5}
    else:
        return {"binary": "clean", "label": "clean", "confidence": 0.5}
