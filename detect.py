"""
detect.py — Watermark detection pipeline.
THIS FILE IS MODIFIED BY THE AGENT. Everything is fair game.

Experiment 1: Random Forest on border-region features + template match scores.
"""

import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier

TRAIN_SET = []
TEMPLATES = {}
MODEL = None
TEMPLATE_INFO = []

TEMPLATE_LABEL_MAP = {
    "dalle_watermark": "dalle",
    "gemini_watermark": "gemini",
    "grok_watermark": "grok",
    "hailuoai_watermark": "minimax_hailuo",
    "hailuoaixminimax_watermark": "minimax_hailuo",
    "minimax_watermark": "minimax_hailuo",
    "openai_watermark": "openai_logo",
    "sora_watermark": "sora",
    "this-person-does-not-exist_watermark": "text_tpdne",
}


def preprocess_templates(templates):
    """Load templates and create edge maps at multiple scales."""
    info = []
    for name, path in templates.items():
        tmpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if tmpl is None:
            continue

        if len(tmpl.shape) == 3 and tmpl.shape[2] == 4:
            alpha = tmpl[:, :, 3].astype(float) / 255.0
            gray = cv2.cvtColor(tmpl[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray = (gray.astype(float) * alpha).astype(np.uint8)
        elif len(tmpl.shape) == 3:
            gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        else:
            gray = tmpl

        edges = cv2.Canny(gray, 30, 100)
        orig_h, orig_w = gray.shape

        if orig_w > 1000:
            target_widths = [20, 35, 55]
        elif orig_w > 300:
            target_widths = [80, 140, 220]
        else:
            target_widths = [50, 80, 130]

        scales = []
        for tw in target_widths:
            s = tw / orig_w
            th = max(int(orig_h * s), 5)
            if th < 5:
                continue
            scaled = cv2.resize(edges, (tw, th))
            scales.append(scaled)

        info.append({
            "name": name,
            "label": TEMPLATE_LABEL_MAP.get(name, "unknown"),
            "scales": scales,
        })
    return info


def get_template_scores(img_edges, h, w):
    """Best template match score per template across regions and scales."""
    regions = [
        img_edges[h * 2 // 3:, w // 2:],       # bottom-right
        img_edges[h * 2 // 3:, :w // 2],        # bottom-left
        img_edges[:h // 4, :],                   # top strip
        img_edges[h * 3 // 4:, :],               # bottom strip
    ]

    scores = {}
    for ti in TEMPLATE_INFO:
        best = 0.0
        for tmpl in ti["scales"]:
            for region in regions:
                if tmpl.shape[0] > region.shape[0] or tmpl.shape[1] > region.shape[1]:
                    continue
                try:
                    result = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
                    s = float(result.max())
                    if s > best:
                        best = s
                except Exception:
                    pass
        scores[ti["name"]] = best
    return scores


def extract_features(img_gray, img_edges, h, w, tmpl_scores):
    """Build feature vector from region stats + template scores."""
    feats = []
    ch = max(h // 6, 10)
    cw = max(w // 6, 10)

    region_specs = [
        (0, ch, 0, cw),                    # top-left
        (0, ch, w - cw, w),                # top-right
        (h - ch, h, 0, cw),                # bottom-left
        (h - ch, h, w - cw, w),            # bottom-right
        (0, ch, 0, w),                      # top strip
        (h - ch, h, 0, w),                  # bottom strip
    ]

    for y1, y2, x1, x2 in region_specs:
        r = img_gray[y1:y2, x1:x2]
        e = img_edges[y1:y2, x1:x2]
        feats.append(float(r.mean()))
        feats.append(float(r.std()))
        feats.append(float(np.percentile(r, 95)))
        feats.append(float(np.percentile(r, 5)))
        feats.append(float(e.mean()) / 255.0)

    feats.append(float(img_gray.mean()))
    feats.append(float(img_gray.std()))
    feats.append(float(h))
    feats.append(float(w))
    feats.append(float(h) / float(w) if w > 0 else 1.0)

    for name in sorted(tmpl_scores.keys()):
        feats.append(tmpl_scores[name])

    return feats


def load_image(image_path, max_dim=512):
    """Load, resize, return gray + edges + dims."""
    img = cv2.imread(image_path)
    if img is None:
        pil_img = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return gray, edges, h, w


def setup(train_set: list[dict], templates: dict[str, str]):
    global TRAIN_SET, TEMPLATES, MODEL, TEMPLATE_INFO
    TRAIN_SET = train_set
    TEMPLATES = templates

    TEMPLATE_INFO = preprocess_templates(templates)

    X, y = [], []
    for sample in train_set:
        try:
            gray, edges, h, w = load_image(sample["path"])
            ts = get_template_scores(edges, h, w)
            feats = extract_features(gray, edges, h, w, ts)
            X.append(feats)
            y.append(sample["label"])
        except Exception:
            continue

    MODEL = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    MODEL.fit(np.array(X), y)


def detect(image_path: str) -> dict:
    """Detect watermarks using trained Random Forest."""
    try:
        gray, edges, h, w = load_image(image_path)
        ts = get_template_scores(edges, h, w)
        feats = extract_features(gray, edges, h, w, ts)

        pred = MODEL.predict([feats])[0]
        proba = MODEL.predict_proba([feats])[0]
        confidence = float(proba.max())
        binary = "clean" if pred == "clean" else "watermarked"

        return {"binary": binary, "label": pred, "confidence": confidence}
    except Exception:
        return {"binary": "clean", "label": "clean", "confidence": 0.0}
