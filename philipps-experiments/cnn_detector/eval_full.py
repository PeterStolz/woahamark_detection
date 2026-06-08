"""Full-folder eval of the locked CNN. Reports per-class on grok/gemini across
all 241 grok + 116 gemini + 656 plain-neg + 317 hard-neg images = 1330 images."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    CHECKPOINTS_DIR,
    CLASSES,
    DEBUG_DIR,
    HARD_NEG_DIRS,
    INPUT_SIZE,
    NEG_DIR,
    POS_DIRS,
    SEED,
)
from .data import WatermarkDataset, list_images, load_split
from .evaluate import save_failure_montages
from .model import WatermarkCNN
from .train import evaluate as run_inference, pick_device, sigmoid


def collect_all_entries():
    """Build a flat (path, labels, source) list across the entire dataset."""
    entries = []
    for cls, d in POS_DIRS.items():
        for p in list_images(d):
            labels = {c: 0 for c in CLASSES}
            labels[cls] = 1
            entries.append({"path": str(p), "labels": labels, "source": f"pos_{cls}"})
    for p in list_images(NEG_DIR):
        entries.append({"path": str(p), "labels": {c: 0 for c in CLASSES}, "source": "neg"})
    for d in HARD_NEG_DIRS:
        for p in list_images(d):
            entries.append({"path": str(p), "labels": {c: 0 for c in CLASSES},
                             "source": f"hard_neg_{d.name}"})
    return entries


def main():
    device = pick_device()
    ckpt_path = CHECKPOINTS_DIR / "cnn_v3.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WatermarkCNN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    thresholds = [ckpt["thresholds"][c] for c in CLASSES]
    print(f"checkpoint thresholds: {dict(zip(CLASSES, thresholds))}")

    entries = collect_all_entries()
    print(f"scoring {len(entries)} images")

    split = load_split()
    split_of = {e["path"]: name for name, items in split.items() for e in items}

    ds = WatermarkDataset(entries, augment=False, input_size=INPUT_SIZE, seed=SEED + 3)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
    all_logits, all_labels = run_inference(model, loader, device)
    all_scores = sigmoid(all_logits)

    paths = [e["path"] for e in entries]
    labels = all_labels
    scores = all_scores
    pred = np.array([[scores[i, k] >= thresholds[k] for k in range(len(CLASSES))] for i in range(len(entries))], dtype=int)

    def report_subset(mask, name):
        if mask.sum() == 0:
            return
        print(f"\n=== {name} (n={mask.sum()}) ===")
        for k, c in enumerate(CLASSES):
            sc = scores[mask, k]; y = labels[mask, k]; pr = pred[mask, k]
            f1 = f1_score(y, pr, average="macro", zero_division=0)
            try:
                auc = roc_auc_score(y, sc) if len(set(y.tolist())) == 2 else float("nan")
            except ValueError:
                auc = float("nan")
            cm = confusion_matrix(y, pr, labels=[0, 1])
            prfs = precision_recall_fscore_support(y, pr, labels=[0, 1], zero_division=0)
            n_pos = int((y == 1).sum())
            print(f"  [{c}] macro_F1={f1:.4f} AUC={auc:.4f} pos={n_pos} thr={thresholds[k]:.3f}")
            print(f"    neg P/R/F1: {prfs[0][0]:.4f} / {prfs[1][0]:.4f} / {prfs[2][0]:.4f}")
            print(f"    pos P/R/F1: {prfs[0][1]:.4f} / {prfs[1][1]:.4f} / {prfs[2][1]:.4f}")
            print(f"    confusion:\n{cm}")

    report_subset(np.ones(len(entries), dtype=bool), "FULL")
    for split_name in ("train", "val", "test"):
        mask = np.array([split_of.get(p) == split_name for p in paths])
        report_subset(mask, f"split={split_name}")

    out = DEBUG_DIR / "full_eval"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "all_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["path", "split", "source"]
        for c in CLASSES:
            header += [f"label_{c}", f"score_{c}", f"pred_{c}"]
        w.writerow(header)
        for i, e in enumerate(entries):
            row = [e["path"], split_of.get(e["path"], "unknown"), e["source"]]
            for k, c in enumerate(CLASSES):
                row += [int(labels[i, k]), f"{scores[i, k]:.6f}", int(pred[i, k])]
            w.writerow(row)
    print(f"\nwrote {out / 'all_scores.csv'}")
    save_failure_montages(paths, labels, scores, thresholds, out)


if __name__ == "__main__":
    main()
