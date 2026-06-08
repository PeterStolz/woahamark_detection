"""Eval the locked V5 config on the full watermark_gemini + no_watermark folders."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from .config import DEBUG_DIR, NEG_DIR, POS_DIR
from .data import list_images, load_split
from .detect import DEFAULT_THRESHOLD, default_config_and_template, score
from .evaluate import save_failure_montages


def main():
    cfg, template = default_config_and_template()
    threshold = DEFAULT_THRESHOLD

    pos_paths = list_images(POS_DIR)
    neg_paths = list_images(NEG_DIR)
    items = [(str(p), 1) for p in pos_paths] + [(str(p), 0) for p in neg_paths]
    print(f"scoring {len(pos_paths)} positives + {len(neg_paths)} negatives = {len(items)} images")

    split = load_split()
    split_of = {p: name for name, entries in split.items() for p, _ in entries}

    paths, labels, scores = [], [], []
    for path, lab in tqdm(items, desc="full"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        scores.append(float(score(img, template, cfg)) if img is not None else 0.0)
        labels.append(lab); paths.append(path)

    s = np.asarray(scores); y = np.asarray(labels)
    pred = (s >= threshold).astype(int)

    def report_subset(mask, name):
        if mask.sum() == 0:
            print(f"  {name}: empty"); return
        ys, ps, ss = y[mask], pred[mask], s[mask]
        f1 = f1_score(ys, ps, average="macro", zero_division=0)
        try:
            auc = roc_auc_score(ys, ss) if len(set(ys.tolist())) == 2 else float("nan")
        except ValueError:
            auc = float("nan")
        cm = confusion_matrix(ys, ps, labels=[0, 1])
        prfs = precision_recall_fscore_support(ys, ps, labels=[0, 1], zero_division=0)
        n_pos = int((ys == 1).sum()); n_neg = int((ys == 0).sum())
        print(f"\n=== {name} (n={len(ys)}, pos={n_pos}, neg={n_neg}) ===")
        print(f"  macro F1: {f1:.4f}  AUC: {auc:.4f}")
        print(f"  neg P/R/F1: {prfs[0][0]:.4f} / {prfs[1][0]:.4f} / {prfs[2][0]:.4f}")
        print(f"  pos P/R/F1: {prfs[0][1]:.4f} / {prfs[1][1]:.4f} / {prfs[2][1]:.4f}")
        print(f"  confusion:\n{cm}")

    report_subset(np.ones_like(y, dtype=bool), "FULL")
    for split_name in ("train", "val", "test"):
        mask = np.array([split_of.get(p) == split_name for p in paths])
        report_subset(mask, f"split={split_name}")

    out = DEBUG_DIR / "full_eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "label", "score", "prediction"])
        for p, lab, sc, pr in zip(paths, labels, scores, pred.tolist()):
            w.writerow([p, split_of.get(p, "unknown"), lab, f"{sc:.6f}", int(pr)])
    print(f"\nwrote {out / 'all_scores.csv'}")
    save_failure_montages(paths, labels, scores, threshold, out, tag="full")


if __name__ == "__main__":
    main()
