"""
detect.py — Watermark detection pipeline.
THIS FILE IS MODIFIED BY THE AGENT. Everything is fair game.

Experiment 4: Full template matching + histogram/unsharp/cross-region features
+ GBT classifier with class balancing. Fast GBT inference with max ~30ms/image.
"""

import numpy as np
from PIL import Image
import cv2
from sklearn.ensemble import GradientBoostingClassifier

TRAIN_SET = []
TEMPLATES = {}
MODEL = None
BINARY_MODEL = None
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
    """Load templates with edge + grayscale at multiple scales."""
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

        # More scales for better matching
        if orig_w > 1000:
            target_widths = [15, 25, 40, 60, 80]
        elif orig_w > 300:
            target_widths = [60, 100, 160, 220, 300]
        else:
            target_widths = [40, 70, 110, 160, 200]

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
    """Best template match score per template — edge + grayscale, 4 regions."""
    edge_regions = [
        img_edges[h * 2 // 3:, w // 2:],       # bottom-right
        img_edges[h * 2 // 3:, :w // 2],        # bottom-left
        img_edges[:h // 4, :],                   # top strip
        img_edges[h * 3 // 4:, :],               # bottom strip
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
    """Compute local contrast features."""
    if gray_region.size < 100:
        return [0.0, 0.0, 0.0]
    gray_f = gray_region.astype(np.float32)
    kernel = np.ones((7, 7), np.float32) / 49
    local_mean = cv2.filter2D(gray_f, -1, kernel)
    diff = gray_f - local_mean
    return [
        float((diff > 15).sum()) / diff.size,
        float((diff < -15).sum()) / diff.size,
        float(diff.std()),
    ]


def dct_features(gray_region, block_size=32):
    """Compute DCT energy distribution features."""
    if gray_region.shape[0] < block_size or gray_region.shape[1] < block_size:
        return [0.0, 0.0, 0.0]
    cy, cx = gray_region.shape[0] // 2, gray_region.shape[1] // 2
    half = block_size // 2
    block = gray_region[cy - half:cy + half, cx - half:cx + half].astype(np.float32)
    dct = cv2.dct(block)
    total = float(np.abs(dct).sum()) + 1e-6
    low = float(np.abs(dct[:block_size // 4, :block_size // 4]).sum())
    mid = float(np.abs(dct[block_size // 4:block_size // 2, block_size // 4:block_size // 2]).sum())
    high = float(np.abs(dct[block_size // 2:, block_size // 2:]).sum())
    return [low / total, mid / total, high / total]


def unsharp_features(gray_region):
    """Detect overlay artifacts by comparing sharp vs blurred."""
    if gray_region.shape[0] < 10 or gray_region.shape[1] < 10:
        return [0.0, 0.0]
    blurred = cv2.GaussianBlur(gray_region, (5, 5), 1.5)
    diff = gray_region.astype(float) - blurred.astype(float)
    return [float(np.abs(diff).mean()), float((np.abs(diff) > 10).sum() / diff.size)]


def region_features(gray_region, edge_region, hsv_region=None):
    """Extract comprehensive region features."""
    feats = []
    # Basic stats
    feats.append(float(gray_region.mean()))
    feats.append(float(gray_region.std()))
    feats.append(float(np.percentile(gray_region, 95)))
    feats.append(float(np.percentile(gray_region, 5)))
    feats.append(float(np.percentile(gray_region, 95) - np.percentile(gray_region, 5)))
    feats.append(float(edge_region.mean()) / 255.0)

    # Gradient orientation
    sobel_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
    horiz_energy = float(np.abs(sobel_y).mean())
    vert_energy = float(np.abs(sobel_x).mean())
    feats.append(horiz_energy)
    feats.append(vert_energy)
    feats.append(horiz_energy / (vert_energy + 1e-6))

    # Local contrast
    feats.extend(local_contrast_features(gray_region))

    # Unsharp overlay detection
    feats.extend(unsharp_features(gray_region))

    # Pixel histogram
    hist, _ = np.histogram(gray_region.ravel(), bins=8, range=(0, 256), density=True)
    feats.extend(hist.tolist())

    # Color stats
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
    """Build complete feature vector."""
    feats = []

    ch = max(h // 6, 10)
    cw = max(w // 6, 10)

    # Coarse regions (6 regions)
    coarse_specs = [
        (0, ch, 0, cw),                    # top-left
        (0, ch, w - cw, w),                # top-right
        (h - ch, h, 0, cw),                # bottom-left
        (h - ch, h, w - cw, w),            # bottom-right
        (0, ch, 0, w),                      # top strip
        (h - ch, h, 0, w),                  # bottom strip
    ]

    for y1, y2, x1, x2 in coarse_specs:
        feats.extend(region_features(
            img_gray[y1:y2, x1:x2],
            img_edges[y1:y2, x1:x2],
            img_hsv[y1:y2, x1:x2],
        ))

    # Fine-grained bottom-right (1/10 — typical watermark location)
    fh = max(h // 10, 8)
    fw = max(w // 10, 8)
    feats.extend(region_features(
        img_gray[h - fh:, w - fw:],
        img_edges[h - fh:, w - fw:],
        img_hsv[h - fh:, w - fw:],
    ))

    # Fine-grained top strip (1/12 — TPDNE text location)
    th = max(h // 12, 8)
    feats.extend(region_features(
        img_gray[:th, :],
        img_edges[:th, :],
        img_hsv[:th, :],
    ))

    # Very fine bottom-right (1/15 — tiny gemini star)
    fh2 = max(h // 15, 6)
    fw2 = max(w // 15, 6)
    feats.extend(region_features(
        img_gray[h - fh2:, w - fw2:],
        img_edges[h - fh2:, w - fw2:],
        img_hsv[h - fh2:, w - fw2:],
    ))

    # DCT features for key regions
    feats.extend(dct_features(img_gray[h * 2 // 3:, w // 2:]))
    feats.extend(dct_features(img_gray[:h // 4, :]))

    # Cross-region contrast (corner vs center)
    center = img_gray[h // 3:2 * h // 3, w // 3:2 * w // 3]
    center_mean = float(center.mean())
    center_std = float(center.std())
    br_corner = img_gray[h - ch:, w - cw:]
    tl_corner = img_gray[:ch, :cw]
    top_strip = img_gray[:ch, :]
    feats.append(float(br_corner.mean()) - center_mean)
    feats.append(float(tl_corner.mean()) - center_mean)
    feats.append(float(br_corner.std()) - center_std)
    feats.append(float(top_strip.mean()) - center_mean)

    # Global features
    feats.append(float(img_gray.mean()))
    feats.append(float(img_gray.std()))
    feats.append(float(h))
    feats.append(float(w))
    feats.append(float(h) / float(w) if w > 0 else 1.0)
    feats.append(float(img_hsv[:, :, 1].mean()))
    feats.append(float(img_hsv[:, :, 1].std()))
    feats.extend(local_contrast_features(img_gray))

    # Template match scores
    for name in sorted(tmpl_scores.keys()):
        feats.append(tmpl_scores[name])

    return feats


def load_image(image_path, max_dim=768):
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
    global TRAIN_SET, TEMPLATES, MODEL, BINARY_MODEL, TEMPLATE_INFO
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

    # Class-balanced sample weights with extra boost for minority classes
    from collections import Counter
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    # Stronger weighting: use sqrt to further boost rare classes
    weight_map = {label: (total / (n_classes * count)) ** 1.2 for label, count in counts.items()}
    sample_weights = np.array([weight_map[label] for label in y])

    X_arr = np.array(X)

    # Stage 1: Binary classifier (clean vs. watermarked) — more sensitive
    y_binary = ["clean" if label == "clean" else "watermarked" for label in y]
    binary_counts = Counter(y_binary)
    binary_weights = np.array([
        (total / (2 * binary_counts[label])) ** 1.3  # strong boost for watermarked
        for label in y_binary
    ])

    BINARY_MODEL = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        subsample=0.8,
    )
    BINARY_MODEL.fit(X_arr, y_binary, sample_weight=binary_weights)

    # Stage 2: Multi-class classifier
    MODEL = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
        subsample=0.8,
        min_samples_leaf=3,
    )
    MODEL.fit(X_arr, y, sample_weight=sample_weights)


def detect(image_path: str) -> dict:
    """Detect watermarks using two-stage classification."""
    try:
        gray, edges, hsv, h, w = load_image(image_path)
        ts = get_template_scores(edges, gray, h, w)
        feats = extract_features(gray, edges, hsv, h, w, ts)

        # Stage 1: Binary detection (sensitive — catches more watermarks)
        binary_proba = BINARY_MODEL.predict_proba([feats])[0]
        binary_classes = BINARY_MODEL.classes_
        wm_idx = list(binary_classes).index("watermarked")
        wm_prob = binary_proba[wm_idx]

        # Stage 2: Multi-class prediction
        pred = MODEL.predict([feats])[0]
        proba = MODEL.predict_proba([feats])[0]
        multi_confidence = float(proba.max())

        # Decision logic: if binary detector says watermarked with decent confidence,
        # trust the multi-class prediction even if it says clean
        if pred == "clean" and wm_prob > 0.55:
            # Binary says watermarked but multi-class says clean
            # Use multi-class probas excluding clean
            classes = MODEL.classes_
            clean_idx = list(classes).index("clean")
            proba_no_clean = proba.copy()
            proba_no_clean[clean_idx] = 0
            if proba_no_clean.max() > 0.05:
                best_wm_idx = int(np.argmax(proba_no_clean))
                pred = classes[best_wm_idx]
                multi_confidence = float(proba_no_clean[best_wm_idx])

        binary = "clean" if pred == "clean" else "watermarked"
        return {"binary": binary, "label": pred, "confidence": multi_confidence}
    except Exception:
        return {"binary": "clean", "label": "clean", "confidence": 0.0}
