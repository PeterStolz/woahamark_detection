"""Phase 1 evaluation: tune threshold on val, report on test, save montages + CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from .config import DEBUG_DIR, ROOT
from .data import load_split
from .detect import ScoreConfig, load_template_gray, score


ScoreFn = Callable[[np.ndarray], float]


def score_split(items: List[Tuple[str, int]], score_fn: ScoreFn) -> Tuple[List[float], List[int], List[str]]:
    scores, labels, paths = [], [], []
    for path, label in tqdm(items, desc="scoring"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            scores.append(0.0)
        else:
            scores.append(float(score_fn(img)))
        labels.append(int(label))
        paths.append(path)
    return scores, labels, paths


def best_threshold_macro_f1(scores: List[float], labels: List[int],
                             lo: float = 0.20, hi: float = 0.90, step: float = 0.02) -> Tuple[float, float]:
    s = np.asarray(scores)
    y = np.asarray(labels)
    best_t, best_f1 = lo, -1.0
    t = lo
    while t <= hi + 1e-9:
        pred = (s >= t).astype(int)
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
        t += step
    return best_t, best_f1


def report(scores, labels, threshold, name="test"):
    s = np.asarray(scores)
    y = np.asarray(labels)
    pred = (s >= threshold).astype(int)
    macro = f1_score(y, pred, average="macro", zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    try:
        auc = roc_auc_score(y, s)
    except ValueError:
        auc = float("nan")
    print(f"\n=== {name} report (threshold={threshold:.3f}) ===")
    print(f"macro F1: {macro:.4f}  ROC-AUC: {auc:.4f}")
    print(f"class 0 (neg): precision={p[0]:.4f} recall={r[0]:.4f} f1={f1[0]:.4f}")
    print(f"class 1 (pos): precision={p[1]:.4f} recall={r[1]:.4f} f1={f1[1]:.4f}")
    print(f"confusion matrix [rows=true 0/1, cols=pred 0/1]:\n{cm}")
    return {
        "macro_f1": macro, "auc": auc,
        "p0": p[0], "r0": r[0], "f0": f1[0],
        "p1": p[1], "r1": r[1], "f1": f1[1],
        "cm": cm.tolist(),
        "threshold": threshold,
    }


def make_montage(paths_scores, max_n=32, cols=8, tile=160) -> np.ndarray:
    items = paths_scores[:max_n]
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * tile, cols * tile, 3), 30, dtype=np.uint8)
    for i, (p, sc) in enumerate(items):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        s = tile / max(h, w)
        rw, rh = max(1, int(w * s)), max(1, int(h * s))
        small = cv2.resize(img, (rw, rh))
        r, c = divmod(i, cols)
        y0 = r * tile + (tile - rh) // 2
        x0 = c * tile + (tile - rw) // 2
        canvas[y0:y0 + rh, x0:x0 + rw] = small
        cv2.putText(canvas, f"{sc:.2f}", (c * tile + 4, r * tile + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_failure_montages(paths, labels, scores, threshold, out_dir: Path, tag: str = "phase1"):
    out_dir.mkdir(parents=True, exist_ok=True)
    s = np.asarray(scores)
    y = np.asarray(labels)
    pred = (s >= threshold).astype(int)

    # FN: true positive, predicted negative; rank by lowest score (worst)
    fn_idx = np.where((y == 1) & (pred == 0))[0]
    fn_sorted = sorted(fn_idx.tolist(), key=lambda i: s[i])
    fn_items = [(paths[i], float(s[i])) for i in fn_sorted]

    # FP: true negative, predicted positive; rank by highest score (worst)
    fp_idx = np.where((y == 0) & (pred == 1))[0]
    fp_sorted = sorted(fp_idx.tolist(), key=lambda i: -s[i])
    fp_items = [(paths[i], float(s[i])) for i in fp_sorted]

    if fn_items:
        cv2.imwrite(str(out_dir / f"{tag}_worst_fn.jpg"), make_montage(fn_items, 32))
    if fp_items:
        cv2.imwrite(str(out_dir / f"{tag}_worst_fp.jpg"), make_montage(fp_items, 32))
    print(f"saved montages: FN={len(fn_items)} FP={len(fp_items)} -> {out_dir}")
    return fn_items, fp_items


def write_scores_csv(out_path: Path, paths, labels, scores, threshold):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "score", "prediction"])
        for p, lab, sc in zip(paths, labels, scores):
            w.writerow([p, lab, f"{sc:.6f}", int(sc >= threshold)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--phase-tag", default="phase1")
    ap.add_argument("--threshold", type=float, default=None,
                    help="If given, skip val tuning and use this threshold.")
    ap.add_argument("--sweep-lo", type=float, default=0.05)
    ap.add_argument("--sweep-hi", type=float, default=0.95)
    ap.add_argument("--baseline", action="store_true",
                    help="Use the Phase 1 baseline config (master template, wide scales).")
    args = ap.parse_args()

    split = load_split()
    if args.baseline:
        template = load_template_gray()
        cfg = ScoreConfig()
    else:
        from .detect import default_config_and_template
        cfg, template = default_config_and_template()
    score_fn = lambda img: score(img, template, cfg)

    if args.threshold is None:
        val_scores, val_labels, _ = score_split(split["val"], score_fn)
        thr, val_f1 = best_threshold_macro_f1(val_scores, val_labels, lo=args.sweep_lo, hi=args.sweep_hi)
        print(f"val best threshold={thr:.3f}  val macro F1={val_f1:.4f}")
    else:
        thr = args.threshold
        print(f"using provided threshold={thr:.3f}")

    eval_items = split[args.split]
    scores_, labels, paths = score_split(eval_items, score_fn)
    metrics = report(scores_, labels, thr, name=args.split)

    debug_phase = DEBUG_DIR / args.phase_tag
    save_failure_montages(paths, labels, scores_, thr, debug_phase, tag=args.phase_tag)
    csv_path = debug_phase / f"{args.split}_scores.csv"
    write_scores_csv(csv_path, paths, labels, scores_, thr)
    print(f"scores csv: {csv_path}")

    metrics_path = debug_phase / f"{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
