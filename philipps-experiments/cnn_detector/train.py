"""Phase 1: train the multi-label CNN on grok+gemini.

Logs train/val loss + per-class val macro F1 each epoch. Saves best checkpoint
by mean(per-class val macro F1) plus per-class thresholds tuned on val.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from .config import CHECKPOINTS_DIR, CLASSES, DEBUG_DIR, INPUT_SIZE, SEED
from .data import WatermarkDataset, load_split
from .model import WatermarkCNN, count_params


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def per_class_pos_weight(entries):
    """sqrt(neg/pos) — softer than full ratio; full ratio destabilized v1 training."""
    weights = []
    for c in CLASSES:
        n_pos = sum(1 for e in entries if e["labels"][c] == 1)
        n_neg = len(entries) - n_pos
        weights.append(max(1.0, (n_neg / max(1, n_pos)) ** 0.5))
    return torch.tensor(weights, dtype=torch.float32)


def best_threshold_per_class(scores: np.ndarray, labels: np.ndarray, lo=0.05, hi=0.95, step=0.02):
    """For each class column, return (best_threshold, best_f1)."""
    out = []
    n_classes = scores.shape[1]
    for k in range(n_classes):
        best_t, best_f1 = lo, -1.0
        t = lo
        while t <= hi + 1e-9:
            pred = (scores[:, k] >= t).astype(int)
            f1 = f1_score(labels[:, k], pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
            t += step
        out.append((best_t, best_f1))
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", type=str, default="cnn_v1")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    split = load_split()
    train_ds = WatermarkDataset(split["train"], augment=True, input_size=INPUT_SIZE, seed=SEED)
    val_ds = WatermarkDataset(split["val"], augment=False, input_size=INPUT_SIZE, seed=SEED + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    model = WatermarkCNN(n_classes=len(CLASSES)).to(device)
    print(f"model params: {count_params(model):,}")

    pos_w = per_class_pos_weight(split["train"]).to(device)
    print(f"pos weights: {pos_w.tolist()}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    log = {"epochs": []}
    best_score = -1.0
    best_epoch = -1
    best_thresholds = [0.5] * len(CLASSES)
    epochs_without_improve = 0

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DEBUG_DIR / "phase1" / f"{args.tag}_train_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        train_loss = float(np.mean(losses))

        # validation
        val_logits, val_labels = evaluate(model, val_loader, device)
        val_scores = sigmoid(val_logits)
        per_cls = best_threshold_per_class(val_scores, val_labels)
        per_cls_f1 = [f1 for _, f1 in per_cls]
        per_cls_thr = [t for t, _ in per_cls]
        try:
            per_cls_auc = [roc_auc_score(val_labels[:, k], val_scores[:, k]) for k in range(len(CLASSES))]
        except ValueError:
            per_cls_auc = [float("nan")] * len(CLASSES)
        mean_f1 = float(np.mean(per_cls_f1))

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1_per_class": per_cls_f1,
            "val_threshold_per_class": per_cls_thr,
            "val_auc_per_class": per_cls_auc,
            "mean_val_macro_f1": mean_f1,
            "lr": opt.param_groups[0]["lr"],
            "time_s": time.time() - t0,
        }
        log["epochs"].append(epoch_log)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        line = (f"epoch {epoch:3d}  loss={train_loss:.4f}  "
                + "  ".join(f"{c}: F1={per_cls_f1[i]:.4f}/AUC={per_cls_auc[i]:.4f}/thr={per_cls_thr[i]:.2f}"
                            for i, c in enumerate(CLASSES))
                + f"  mean_F1={mean_f1:.4f}  ({epoch_log['time_s']:.1f}s)")
        print(line)

        improved = mean_f1 > best_score + 1e-6
        if improved:
            best_score = mean_f1
            best_epoch = epoch
            best_thresholds = per_cls_thr
            epochs_without_improve = 0
            ckpt_path = CHECKPOINTS_DIR / f"{args.tag}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "thresholds": dict(zip(CLASSES, per_cls_thr)),
                "input_size": INPUT_SIZE,
                "classes": list(CLASSES),
                "epoch": epoch,
                "val_macro_f1_per_class": dict(zip(CLASSES, per_cls_f1)),
            }, ckpt_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"early stop at epoch {epoch} (best epoch {best_epoch}, mean_F1={best_score:.4f})")
                break

    log["best_epoch"] = best_epoch
    log["best_mean_val_macro_f1"] = best_score
    log["best_thresholds"] = dict(zip(CLASSES, best_thresholds))
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"best epoch {best_epoch}: mean val macro F1 = {best_score:.4f}, "
          f"thresholds = {dict(zip(CLASSES, best_thresholds))}")
    print(f"saved checkpoint -> {CHECKPOINTS_DIR / (args.tag + '.pt')}")


if __name__ == "__main__":
    main()
