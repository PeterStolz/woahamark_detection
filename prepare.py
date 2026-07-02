"""
prepare.py — Fixed evaluation harness for woahamark_detection.
DO NOT MODIFY. The agent only modifies detect.py.

Handles:
  - Image discovery and labeling from directory structure
  - Train/val splitting (stratified, deterministic)
  - Calling the agent's detect() function
  - Computing all metrics (binary + multi-class)
  - Enforcing the time budget
  - Reporting results
"""

import os
import sys
import json
import time
import signal
import hashlib
import argparse
import importlib
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from tabulate import tabulate

# ─────────────────────────────────────────────────────────────
# Constants — do not change
# ─────────────────────────────────────────────────────────────

IMAGES_DIR = Path("images")
TEMPLATES_DIR = IMAGES_DIR / "watermarks"

# Time budget in seconds (wall clock for the full eval run)
TIME_BUDGET_SECONDS = 240  # 4 minutes (bumped from 2 by Peter, 2026-07-02 —
# shared box under external load pushed runs to the 120s cliff)

# Random seed for reproducible splits
SPLIT_SEED = 42
VAL_FRACTION = 0.20

# Directory name -> label mapping
DIR_LABEL_MAP = {
    "no_watermark": "clean",
    "watermark_dalle": "dalle",
    "watermark_gemini": "gemini",
    "watermark_grok": "grok",
    "watermark_minimax_hailuoAI": "minimax_hailuo",
    "watermark_openai_logo": "openai_logo",
    "watermark_sora": "sora",
    "watermark_text_this-person-does-not-exist.com": "text_tpdne",
}

# For binary evaluation: anything not "clean" is "watermarked"
BINARY_POS = "watermarked"
BINARY_NEG = "clean"

# Supported image extensions
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


# ─────────────────────────────────────────────────────────────
# Image discovery
# ─────────────────────────────────────────────────────────────

def discover_images() -> list[dict]:
    """Walk the images/ directory and build a labeled dataset.

    Returns a list of dicts: {"path": str, "label": str, "binary_label": str}
    """
    dataset = []
    for dir_name, label in DIR_LABEL_MAP.items():
        dir_path = IMAGES_DIR / dir_name
        if not dir_path.is_dir():
            continue
        for f in sorted(dir_path.iterdir()):
            if f.suffix.lower() in IMG_EXTENSIONS and f.is_file():
                binary = BINARY_NEG if label == "clean" else BINARY_POS
                dataset.append({
                    "path": str(f),
                    "label": label,
                    "binary_label": binary,
                })
    return dataset


def get_templates() -> dict[str, str]:
    """Return {template_name: path} for all reference watermark templates."""
    templates = {}
    if TEMPLATES_DIR.is_dir():
        for f in sorted(TEMPLATES_DIR.iterdir()):
            if f.suffix.lower() in IMG_EXTENSIONS and f.is_file():
                templates[f.stem] = str(f)
    return templates


def split_dataset(dataset: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split, deterministic."""
    if len(dataset) < 5:
        # Too few images — use all for val (bootstrap mode)
        return [], dataset

    labels = [d["label"] for d in dataset]
    label_counts = Counter(labels)

    # If any class has only 1 sample, we can't stratify — fall back to random
    min_count = min(label_counts.values())
    stratify = labels if min_count >= 2 else None

    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=VAL_FRACTION,
        random_state=SPLIT_SEED,
        stratify=stratify,
    )
    train = [dataset[i] for i in train_idx]
    val = [dataset[i] for i in val_idx]
    return train, val


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

def run_evaluation(detect_fn, val_set: list[dict], time_limit: float) -> dict:
    """Run detect_fn on each val image, collect predictions, compute metrics.

    detect_fn signature: detect(image_path: str) -> dict
        Must return at minimum:
            {"binary": "clean" | "watermarked"}
        Optionally:
            {"binary": ..., "label": "<specific_class>", "confidence": float}

    Returns a results dict with all metrics.
    """
    y_true_multi = []
    y_pred_multi = []
    y_true_binary = []
    y_pred_binary = []
    confidences = []
    errors = []
    per_image_results = []
    start_time = time.time()

    for i, sample in enumerate(val_set):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            errors.append(f"TIME BUDGET EXCEEDED after {i}/{len(val_set)} images")
            break

        path = sample["path"]
        true_label = sample["label"]
        true_binary = sample["binary_label"]

        try:
            result = detect_fn(path)
        except Exception as e:
            errors.append(f"CRASH on {path}: {e}")
            # On crash, predict "clean" (worst case for a detector)
            result = {"binary": "clean", "label": "clean", "confidence": 0.0}

        # Extract predictions
        pred_binary = result.get("binary", "clean")
        pred_label = result.get("label", pred_binary)
        confidence = result.get("confidence", 0.0)

        # Normalize predictions
        if pred_binary not in (BINARY_POS, BINARY_NEG):
            pred_binary = BINARY_NEG  # default to clean on garbage output

        # For multi-class: if the detector only does binary, map watermarked -> "unknown_watermark"
        if pred_label in (BINARY_POS, BINARY_NEG):
            pred_label_multi = "clean" if pred_label == BINARY_NEG else "unknown_watermark"
        else:
            pred_label_multi = pred_label

        y_true_binary.append(true_binary)
        y_pred_binary.append(pred_binary)
        y_true_multi.append(true_label)
        y_pred_multi.append(pred_label_multi)
        confidences.append(confidence)

        per_image_results.append({
            "path": path,
            "true_label": true_label,
            "pred_label": pred_label_multi,
            "true_binary": true_binary,
            "pred_binary": pred_binary,
            "confidence": confidence,
            "correct_binary": true_binary == pred_binary,
            "correct_multi": true_label == pred_label_multi,
        })

    total_time = time.time() - start_time
    n_evaluated = len(y_true_binary)

    # ── Binary metrics ──
    binary_acc = accuracy_score(y_true_binary, y_pred_binary) if n_evaluated > 0 else 0
    binary_f1 = f1_score(y_true_binary, y_pred_binary, pos_label=BINARY_POS, zero_division=0) if n_evaluated > 0 else 0
    binary_precision = precision_score(y_true_binary, y_pred_binary, pos_label=BINARY_POS, zero_division=0) if n_evaluated > 0 else 0
    binary_recall = recall_score(y_true_binary, y_pred_binary, pos_label=BINARY_POS, zero_division=0) if n_evaluated > 0 else 0

    # ── Multi-class metrics ──
    all_labels = sorted(set(y_true_multi + y_pred_multi))
    macro_f1 = f1_score(y_true_multi, y_pred_multi, average="macro", zero_division=0) if n_evaluated > 0 else 0
    weighted_f1 = f1_score(y_true_multi, y_pred_multi, average="weighted", zero_division=0) if n_evaluated > 0 else 0
    multi_acc = accuracy_score(y_true_multi, y_pred_multi) if n_evaluated > 0 else 0

    # Per-class report
    cls_report = classification_report(
        y_true_multi, y_pred_multi,
        labels=all_labels,
        zero_division=0,
        output_dict=True,
    ) if n_evaluated > 0 else {}

    # Confusion matrix
    cm = confusion_matrix(y_true_multi, y_pred_multi, labels=all_labels) if n_evaluated > 0 else np.array([])

    return {
        # Primary metric (what we optimize)
        "macro_f1": macro_f1,
        # Secondary metrics
        "weighted_f1": weighted_f1,
        "multi_acc": multi_acc,
        "binary_f1": binary_f1,
        "binary_acc": binary_acc,
        "binary_precision": binary_precision,
        "binary_recall": binary_recall,
        # Details
        "n_evaluated": n_evaluated,
        "n_total": len(val_set),
        "total_time_s": total_time,
        "avg_time_per_image_ms": (total_time / n_evaluated * 1000) if n_evaluated > 0 else 0,
        "errors": errors,
        "per_class_report": cls_report,
        "confusion_matrix": cm.tolist() if isinstance(cm, np.ndarray) else [],
        "confusion_labels": all_labels,
        "per_image_results": per_image_results,
    }


def print_results(results: dict):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print("  WOAHAMARK DETECTION — EVALUATION RESULTS")
    print("=" * 70)

    # Primary metric — big and bold
    print(f"\n  >>> macro_f1: {results['macro_f1']:.4f} <<<  (PRIMARY METRIC)")
    print(f"  >>> binary_f1: {results['binary_f1']:.4f} <<<  (SECONDARY METRIC)")

    # Summary table
    summary = [
        ["Multi-class Accuracy", f"{results['multi_acc']:.4f}"],
        ["Multi-class Macro F1", f"{results['macro_f1']:.4f}"],
        ["Multi-class Weighted F1", f"{results['weighted_f1']:.4f}"],
        ["Binary Accuracy", f"{results['binary_acc']:.4f}"],
        ["Binary F1", f"{results['binary_f1']:.4f}"],
        ["Binary Precision", f"{results['binary_precision']:.4f}"],
        ["Binary Recall", f"{results['binary_recall']:.4f}"],
        ["Images Evaluated", f"{results['n_evaluated']}/{results['n_total']}"],
        ["Total Time", f"{results['total_time_s']:.1f}s"],
        ["Avg per Image", f"{results['avg_time_per_image_ms']:.1f}ms"],
    ]
    print("\n" + tabulate(summary, headers=["Metric", "Value"], tablefmt="simple"))

    # Per-class breakdown
    cls = results.get("per_class_report", {})
    if cls:
        rows = []
        for label in sorted(cls.keys()):
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            row = cls[label]
            rows.append([
                label,
                f"{row.get('precision', 0):.3f}",
                f"{row.get('recall', 0):.3f}",
                f"{row.get('f1-score', 0):.3f}",
                int(row.get('support', 0)),
            ])
        print("\nPer-class breakdown:")
        print(tabulate(rows, headers=["Class", "Prec", "Recall", "F1", "Support"], tablefmt="simple"))

    # Confusion matrix
    cm = results.get("confusion_matrix", [])
    labels = results.get("confusion_labels", [])
    if cm and labels:
        print("\nConfusion Matrix:")
        # Truncate long labels
        short_labels = [l[:12] for l in labels]
        header = [""] + short_labels
        rows = []
        for i, row in enumerate(cm):
            rows.append([short_labels[i]] + [str(v) for v in row])
        print(tabulate(rows, headers=header, tablefmt="simple"))

    # Errors
    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for err in results["errors"][:10]:
            print(f"  - {err}")
        if len(results["errors"]) > 10:
            print(f"  ... and {len(results['errors']) - 10} more")

    print("\n" + "=" * 70)


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Woahamark Detection — Evaluation Harness")
    parser.add_argument("--verify", action="store_true", help="Verify setup and print dataset stats")
    parser.add_argument("--run", action="store_true", help="Run evaluation using detect.py")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val", help="Which split to evaluate on")
    parser.add_argument("--json", action="store_true", help="Output results as JSON (for automation)")
    args = parser.parse_args()

    # Discover dataset
    dataset = discover_images()
    templates = get_templates()

    if not dataset:
        print("ERROR: No images found in images/ directory.")
        print("Expected structure:")
        for dir_name in DIR_LABEL_MAP:
            print(f"  images/{dir_name}/  (put images here)")
        sys.exit(1)

    train_set, val_set = split_dataset(dataset)

    if args.verify or (not args.run):
        print("\n=== Dataset Summary ===")
        label_counts = Counter(d["label"] for d in dataset)
        binary_counts = Counter(d["binary_label"] for d in dataset)
        print(f"Total images: {len(dataset)}")
        print(f"  Train: {len(train_set)}, Val: {len(val_set)}")
        print(f"\nBinary distribution:")
        for k, v in sorted(binary_counts.items()):
            print(f"  {k}: {v}")
        print(f"\nMulti-class distribution:")
        for k, v in sorted(label_counts.items()):
            print(f"  {k}: {v}")
        print(f"\nTemplates found: {len(templates)}")
        for name, path in sorted(templates.items()):
            print(f"  {name}: {path}")
        print("\nSetup OK. Run with --run to evaluate detect.py")

        if not args.run:
            sys.exit(0)

    if args.run:
        # Import detect.py dynamically
        try:
            if "detect" in sys.modules:
                del sys.modules["detect"]
            detect_module = importlib.import_module("detect")
            detect_fn = detect_module.detect
        except ImportError as e:
            print(f"ERROR: Could not import detect.py: {e}")
            sys.exit(1)
        except AttributeError:
            print("ERROR: detect.py must define a `detect(image_path: str) -> dict` function")
            sys.exit(1)

        # Allow detect.py an optional setup() call
        setup_fn = getattr(detect_module, "setup", None)
        if callable(setup_fn):
            print("Running detect.setup()...")
            try:
                setup_fn(
                    train_set=train_set,
                    templates=templates,
                )
            except Exception as e:
                print(f"WARNING: setup() failed: {e}")
                traceback.print_exc()

        # Pick evaluation set
        if args.split == "train":
            eval_set = train_set
        elif args.split == "all":
            eval_set = dataset
        else:
            eval_set = val_set

        print(f"\nEvaluating on {len(eval_set)} images (split={args.split})...")
        results = run_evaluation(detect_fn, eval_set, TIME_BUDGET_SECONDS)

        if args.json:
            # Strip per_image_results for compact output
            out = {k: v for k, v in results.items() if k != "per_image_results"}
            print(json.dumps(out, indent=2, default=str))
        else:
            print_results(results)

        # Print the one-liner the agent needs for results.tsv
        print(f"\nmacro_f1:{results['macro_f1']:.4f}")
        print(f"binary_f1:{results['binary_f1']:.4f}")
        print(f"multi_acc:{results['multi_acc']:.4f}")
        print(f"time_s:{results['total_time_s']:.1f}")


if __name__ == "__main__":
    main()
