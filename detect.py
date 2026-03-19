"""
detect.py — Watermark detection pipeline.
THIS FILE IS MODIFIED BY THE AGENT. Everything is fair game.

Experiment 3: Add local contrast features, DCT-based frequency features,
grayscale template matching, and class-balanced training.
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
    """Load templates and create edge + grayscale maps at multiple scales."""
    info = []
    for name, path in templates.items():
        tmpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if tmpl is None:
            continue

        if len(tmpl.shape) == 3 and tmpl.shape[2] == 4:
            alpha = tmpl[:, :, 3].astype(float) / 255.0
            gray = cv2.cvtColor(tmpl[:, :, :3], cv2.COLOR_BGR2GRAY)
            gray_masked = (gray.astype(float) * alpha).astype(np.uint8)
        elif len(tmpl.shape) == 3:
            gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
            gray_masked = gray
        else:
            gray = tmpl
            gray_masked = gray

        edges = cv2.Canny(gray_masked, 30, 100)
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
            scaled_edges = cv2.resize(edges, (tw, th))
            scaled_gray = cv2.resize(gray_masked, (tw, th))
            scales.append((scaled_edges, scaled_gray))

        info.append({
            "name": name,
            "label": TEMPLATE_LABEL_MAP.get(name, "unknown"),
            "scales": scales,
        })
    return info


def get_template_scores(img_edges, img_gray, h, w):
    """Best template match score per template (both edge and grayscale matching)."""
    edge_regions = [
        img_edges[h * 2 // 3:, w // 2:],
        img_edges[h * 2 // 3:, :w // 2],
        img_edges[:h // 4, :],
        img_edges[h * 3 // 4:, :],
    ]
    gray_regions = [
        img_gray[h * 2 // 3:, w // 2:],
        img_gray[h * 2 // 3:, :w // 2],
        img_gray[:h // 4, :],
        img_gray[h * 3 // 4:, :],
    ]

    scores = {}
    for ti in TEMPLATE_INFO:
        best_edge = 0.0
        best_gray = 0.0
        for tmpl_edge, tmpl_gray in ti["scales"]:
            for region in edge_regions:
                if tmpl_edge.shape[0] > region.shape[0] or tmpl_edge.shape[1] > region.shape[1]:
                    continue
                try:
                    result = cv2.matchTemplate(region, tmpl_edge, cv2.TM_CCOEFF_NORMED)
                    s = float(result.max())
                    if s > best_edge:
                        best_edge = s
                except Exception:
                    pass
            for region in gray_regions:
                if tmpl_gray.shape[0] > region.shape[0] or tmpl_gray.shape[1] > region.shape[1]:
                    continue
                try:
                    result = cv2.matchTemplate(region, tmpl_gray, cv2.TM_CCOEFF_NORMED)
                    s = float(result.max())
                    if s > best_gray:
                        best_gray = s
                except Exception:
                    pass
        scores[ti["name"] + "_edge"] = best_edge
        scores[ti["name"] + "_gray"] = best_gray
    return scores


def local_contrast_features(gray_region):
    """Compute local contrast: ratio of pixels brighter than local neighborhood."""
    if gray_region.size < 100:
        return [0.0, 0.0, 0.0]
    gray_f = gray_region.astype(np.float32)
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    diff = gray_f - local_mean

    bright_ratio = float((diff > 15).sum()) / diff.size
    dark_ratio = float((diff < -15).sum()) / diff.size
    contrast_std = float(diff.std())
    return [bright_ratio, dark_ratio, contrast_std]


def dct_features(gray_region, block_size=32):
    """Compute DCT energy distribution features."""
    if gray_region.shape[0] < block_size or gray_region.shape[1] < block_size:
        return [0.0, 0.0, 0.0]

    # Take a central block
    cy, cx = gray_region.shape[0] // 2, gray_region.shape[1] // 2
    half = block_size // 2
    block = gray_region[cy - half:cy + half, cx - half:cx + half].astype(np.float32)
    dct = cv2.dct(block)

    # Energy in low, mid, high frequency bands
    total = float(np.abs(dct).sum()) + 1e-6
    low = float(np.abs(dct[:block_size // 4, :block_size // 4]).sum())
    mid = float(np.abs(dct[block_size // 4:block_size // 2, block_size // 4:block_size // 2]).sum())
    high = float(np.abs(dct[block_size // 2:, block_size // 2:]).sum())

    return [low / total, mid / total, high / total]


def region_features(gray_region, edge_region, hsv_region=None):
    """Extract stats from a single region."""
    feats = []
    feats.append(float(gray_region.mean()))
    feats.append(float(gray_region.std()))
    feats.append(float(np.percentile(gray_region, 95)))
    feats.append(float(np.percentile(gray_region, 5)))
    feats.append(float(np.percentile(gray_region, 95) - np.percentile(gray_region, 5)))
    feats.append(float(edge_region.mean()) / 255.0)

    sobel_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
    horiz_energy = float(np.abs(sobel_y).mean())
    vert_energy = float(np.abs(sobel_x).mean())
    feats.append(horiz_energy)
    feats.append(vert_energy)
    feats.append(horiz_energy / (vert_energy + 1e-6))

    # Local contrast
    feats.extend(local_contrast_features(gray_region))

    if hsv_region is not None:
        feats.append(float(hsv_region[:, :, 0].mean()))
        feats.append(float(hsv_region[:, :, 0].std()))
        feats.append(float(hsv_region[:, :, 1].mean()))
        feats.append(float(hsv_region[:, :, 1].std()))
        feats.append(float(hsv_region[:, :, 2].mean()))
        feats.append(float(hsv_region[:, :, 2].std()))
    else:
        feats.extend([0.0] * 6)

    return feats


def extract_features(img_gray, img_edges, img_hsv, h, w, tmpl_scores):
    """Build feature vector."""
    feats = []

    ch = max(h // 6, 10)
    cw = max(w // 6, 10)

    coarse_specs = [
        (0, ch, 0, cw),
        (0, ch, w - cw, w),
        (h - ch, h, 0, cw),
        (h - ch, h, w - cw, w),
        (0, ch, 0, w),
        (h - ch, h, 0, w),
    ]

    for y1, y2, x1, x2 in coarse_specs:
        feats.extend(region_features(
            img_gray[y1:y2, x1:x2],
            img_edges[y1:y2, x1:x2],
            img_hsv[y1:y2, x1:x2],
        ))

    # Fine-grained bottom-right (1/10)
    fh = max(h // 10, 8)
    fw = max(w // 10, 8)
    feats.extend(region_features(
        img_gray[h - fh:, w - fw:],
        img_edges[h - fh:, w - fw:],
        img_hsv[h - fh:, w - fw:],
    ))

    # Fine-grained top strip (1/12)
    th = max(h // 12, 8)
    feats.extend(region_features(
        img_gray[:th, :],
        img_edges[:th, :],
        img_hsv[:th, :],
    ))

    # DCT features for bottom-right and top regions
    feats.extend(dct_features(img_gray[h * 2 // 3:, w // 2:]))
    feats.extend(dct_features(img_gray[:h // 4, :]))

    # Global features
    feats.append(float(img_gray.mean()))
    feats.append(float(img_gray.std()))
    feats.append(float(h))
    feats.append(float(w))
    feats.append(float(h) / float(w) if w > 0 else 1.0)
    feats.append(float(img_hsv[:, :, 1].mean()))
    feats.append(float(img_hsv[:, :, 1].std()))

    # Global local contrast
    feats.extend(local_contrast_features(img_gray))

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
            ts = get_template_scores(edges, gray, h, w)
            feats = extract_features(gray, edges, hsv, h, w, ts)
            X.append(feats)
            y.append(sample["label"])
        except Exception:
            continue

    # Compute class weights for balancing
    from collections import Counter
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    sample_weights = [total / (n_classes * counts[label]) for label in y]

    MODEL = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    MODEL.fit(np.array(X), y, sample_weight=sample_weights)


def detect(image_path: str) -> dict:
    """Detect watermarks using trained classifier."""
    try:
        gray, edges, hsv, h, w = load_image(image_path)
        ts = get_template_scores(edges, gray, h, w)
        feats = extract_features(gray, edges, hsv, h, w, ts)

        pred = MODEL.predict([feats])[0]
        proba = MODEL.predict_proba([feats])[0]
        confidence = float(proba.max())
        binary = "clean" if pred == "clean" else "watermarked"

        return {"binary": binary, "label": pred, "confidence": confidence}
    except Exception:
        return {"binary": "clean", "label": "clean", "confidence": 0.0}
