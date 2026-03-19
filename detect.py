"""
detect.py — Watermark detection pipeline.
THIS FILE IS MODIFIED BY THE AGENT. Everything is fair game.

Experiment 2: Enhanced features — color (HSV), finer corner regions,
aspect ratio, gradient orientation, plus template match scores.
"""

import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import GradientBoostingClassifier

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
            target_widths = [15, 25, 40, 60]
        elif orig_w > 300:
            target_widths = [60, 100, 160, 250]
        else:
            target_widths = [40, 70, 110, 160]

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


def region_features(gray_region, edge_region, color_region_hsv=None):
    """Extract stats from a single region."""
    feats = []
    # Grayscale stats
    feats.append(float(gray_region.mean()))
    feats.append(float(gray_region.std()))
    feats.append(float(np.percentile(gray_region, 95)))
    feats.append(float(np.percentile(gray_region, 5)))
    feats.append(float(np.percentile(gray_region, 95) - np.percentile(gray_region, 5)))  # range

    # Edge stats
    feats.append(float(edge_region.mean()) / 255.0)

    # Gradient orientation — horizontal vs vertical edges
    sobel_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
    horiz_energy = float(np.abs(sobel_y).mean())
    vert_energy = float(np.abs(sobel_x).mean())
    feats.append(horiz_energy)
    feats.append(vert_energy)
    feats.append(horiz_energy / (vert_energy + 1e-6))  # text tends to be horizontal

    # Color stats (HSV)
    if color_region_hsv is not None:
        feats.append(float(color_region_hsv[:, :, 0].mean()))  # hue
        feats.append(float(color_region_hsv[:, :, 0].std()))
        feats.append(float(color_region_hsv[:, :, 1].mean()))  # saturation
        feats.append(float(color_region_hsv[:, :, 1].std()))
        feats.append(float(color_region_hsv[:, :, 2].mean()))  # value
        feats.append(float(color_region_hsv[:, :, 2].std()))
    else:
        feats.extend([0.0] * 6)

    return feats


def extract_features(img_gray, img_edges, img_hsv, h, w, tmpl_scores):
    """Build feature vector from region stats + template scores."""
    feats = []

    # Coarse regions (1/6 of image)
    ch = max(h // 6, 10)
    cw = max(w // 6, 10)

    coarse_specs = [
        (0, ch, 0, cw),                    # top-left corner
        (0, ch, w - cw, w),                # top-right corner
        (h - ch, h, 0, cw),                # bottom-left corner
        (h - ch, h, w - cw, w),            # bottom-right corner
        (0, ch, 0, w),                      # top strip
        (h - ch, h, 0, w),                  # bottom strip
    ]

    for y1, y2, x1, x2 in coarse_specs:
        feats.extend(region_features(
            img_gray[y1:y2, x1:x2],
            img_edges[y1:y2, x1:x2],
            img_hsv[y1:y2, x1:x2],
        ))

    # Fine-grained bottom-right corner (1/10 of image — where most logos sit)
    fh = max(h // 10, 8)
    fw = max(w // 10, 8)
    feats.extend(region_features(
        img_gray[h - fh:, w - fw:],
        img_edges[h - fh:, w - fw:],
        img_hsv[h - fh:, w - fw:],
    ))

    # Fine-grained top strip (where TPDNE text sits — top 1/12)
    th = max(h // 12, 8)
    feats.extend(region_features(
        img_gray[:th, :],
        img_edges[:th, :],
        img_hsv[:th, :],
    ))

    # Global features
    feats.append(float(img_gray.mean()))
    feats.append(float(img_gray.std()))
    feats.append(float(h))
    feats.append(float(w))
    feats.append(float(h) / float(w) if w > 0 else 1.0)

    # Global color features
    feats.append(float(img_hsv[:, :, 1].mean()))  # overall saturation
    feats.append(float(img_hsv[:, :, 1].std()))

    # Template match scores
    for name in sorted(tmpl_scores.keys()):
        feats.append(tmpl_scores[name])

    return feats


def load_image(image_path, max_dim=512):
    """Load, resize, return gray + edges + hsv + dims."""
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return gray, edges, hsv, h, w


def setup(train_set: list[dict], templates: dict[str, str]):
    global TRAIN_SET, TEMPLATES, MODEL, TEMPLATE_INFO
    TRAIN_SET = train_set
    TEMPLATES = templates

    TEMPLATE_INFO = preprocess_templates(templates)

    X, y = [], []
    for sample in train_set:
        try:
            gray, edges, hsv, h, w = load_image(sample["path"])
            ts = get_template_scores(edges, h, w)
            feats = extract_features(gray, edges, hsv, h, w, ts)
            X.append(feats)
            y.append(sample["label"])
        except Exception:
            continue

    MODEL = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    MODEL.fit(np.array(X), y)


def detect(image_path: str) -> dict:
    """Detect watermarks using trained classifier."""
    try:
        gray, edges, hsv, h, w = load_image(image_path)
        ts = get_template_scores(edges, h, w)
        feats = extract_features(gray, edges, hsv, h, w, ts)

        pred = MODEL.predict([feats])[0]
        proba = MODEL.predict_proba([feats])[0]
        confidence = float(proba.max())
        binary = "clean" if pred == "clean" else "watermarked"

        return {"binary": binary, "label": pred, "confidence": confidence}
    except Exception:
        return {"binary": "clean", "label": "clean", "confidence": 0.0}
