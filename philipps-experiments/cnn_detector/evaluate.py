"""Evaluate a trained checkpoint on a chosen split. Threshold per class loaded from ckpt
unless `--retune` is given (which re-runs the val sweep)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .config import CHECKPOINTS_DIR, CLASSES, DEBUG_DIR, INPUT_SIZE, SEED
from .data import WatermarkDataset, load_split
from .model import WatermarkCNN
from .train import evaluate as run_inference, pick_device, sigmoid, best_threshold_per_class


def make_montage(items, max_n=32, cols=8, tile=160):
    items = items[:max_n]
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


def report_per_class(scores, labels, thresholds, name="test"):
    print(f"\n=== {name} report ===")
    out = {}
    for k, c in enumerate(CLASSES):
        sc = scores[:, k]; y = labels[:, k]; thr = thresholds[k]
        pred = (sc >= thr).astype(int)
        macro = f1_score(y, pred, average="macro", zero_division=0)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        try:
            auc = roc_auc_score(y, sc)
        except ValueError:
            auc = float("nan")
        print(f"\n  [{c}] threshold={thr:.3f}  macro F1={macro:.4f}  AUC={auc:.4f}")
        print(f"    neg P/R/F1: {p[0]:.4f} / {r[0]:.4f} / {f1[0]:.4f}")
        print(f"    pos P/R/F1: {p[1]:.4f} / {r[1]:.4f} / {f1[1]:.4f}")
        print(f"    confusion (rows=true 0/1, cols=pred 0/1):\n{cm}")
        out[c] = {"threshold": thr, "macro_f1": macro, "auc": auc,
                  "p0": float(p[0]), "r0": float(r[0]), "f0": float(f1[0]),
                  "p1": float(p[1]), "r1": float(r[1]), "f1": float(f1[1]),
                  "cm": cm.tolist()}
    out["mean_macro_f1"] = float(np.mean([out[c]["macro_f1"] for c in CLASSES]))
    print(f"\n  mean macro F1 across classes = {out['mean_macro_f1']:.4f}")
    return out


def save_failure_montages(paths, labels, scores, thresholds, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(CLASSES):
        sc = scores[:, k]; y = labels[:, k]; thr = thresholds[k]
        pred = (sc >= thr).astype(int)
        fn_idx = np.where((y == 1) & (pred == 0))[0]
        fn = sorted(fn_idx.tolist(), key=lambda i: sc[i])
        fn_items = [(paths[i], float(sc[i])) for i in fn]
        fp_idx = np.where((y == 0) & (pred == 1))[0]
        fp = sorted(fp_idx.tolist(), key=lambda i: -sc[i])
        fp_items = [(paths[i], float(sc[i])) for i in fp]
        if fn_items:
            cv2.imwrite(str(out_dir / f"{c}_worst_fn.jpg"), make_montage(fn_items, 32))
        if fp_items:
            cv2.imwrite(str(out_dir / f"{c}_worst_fp.jpg"), make_montage(fp_items, 32))
        print(f"  [{c}] FN={len(fn_items)} FP={len(fp_items)}")


def write_scores_csv(out_path: Path, paths, labels, scores, thresholds):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["path"]
        for c in CLASSES:
            header += [f"label_{c}", f"score_{c}", f"pred_{c}"]
        w.writerow(header)
        for i, p in enumerate(paths):
            row = [p]
            for k, c in enumerate(CLASSES):
                row += [int(labels[i, k]), f"{scores[i, k]:.6f}",
                        int(scores[i, k] >= thresholds[k])]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--ckpt", default=str(CHECKPOINTS_DIR / "cnn_v3.pt"))
    ap.add_argument("--phase-tag", default="phase1")
    ap.add_argument("--retune", action="store_true",
                    help="Re-tune per-class thresholds on val instead of using checkpoint values.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = WatermarkCNN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    classes = ckpt.get("classes", list(CLASSES))
    assert classes == list(CLASSES), f"checkpoint classes {classes} != {CLASSES}"

    split = load_split()
    val_ds = WatermarkDataset(split["val"], augment=False, input_size=INPUT_SIZE, seed=SEED + 1)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    if args.retune:
        val_logits, val_labels = run_inference(model, val_loader, device)
        val_scores = sigmoid(val_logits)
        per_cls = best_threshold_per_class(val_scores, val_labels)
        thresholds = [t for t, _ in per_cls]
        print(f"re-tuned thresholds: {dict(zip(CLASSES, thresholds))}")
    else:
        thresholds = [ckpt["thresholds"][c] for c in CLASSES]
        print(f"checkpoint thresholds: {dict(zip(CLASSES, thresholds))}")

    eval_ds = WatermarkDataset(split[args.split], augment=False, input_size=INPUT_SIZE, seed=SEED + 2)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    paths = [e["path"] for e in split[args.split]]
    logits, labels = run_inference(model, eval_loader, device)
    scores = sigmoid(logits)

    metrics = report_per_class(scores, labels, thresholds, name=args.split)
    out = DEBUG_DIR / args.phase_tag
    out.mkdir(parents=True, exist_ok=True)
    save_failure_montages(paths, labels, scores, thresholds, out)
    write_scores_csv(out / f"{args.split}_scores.csv", paths, labels, scores, thresholds)
    with open(out / f"{args.split}_metrics.json", "w") as f:
        json.dump({"thresholds": dict(zip(CLASSES, thresholds)), **metrics}, f, indent=2)
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
