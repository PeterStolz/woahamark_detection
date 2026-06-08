"""Phase 3: run independent variants on val, log macro F1 to experiments.csv,
then evaluate the best on test."""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np

from .config import EXPERIMENTS_CSV, ROOT, SCALES
from .data import load_split
from .detect import ScoreConfig, load_template_gray, load_templates, score
from .evaluate import (
    best_threshold_macro_f1,
    report,
    save_failure_montages,
    score_split,
    write_scores_csv,
)


REAL_TEMPLATE_DIR = ROOT / "grok_detector" / "real_templates"


def make_score_fn(template, cfg: ScoreConfig) -> Callable:
    return lambda img: score(img, template, cfg)


def eval_variant(name: str, cfg: ScoreConfig, template, val_items,
                 sweep_lo=0.05, sweep_hi=0.95):
    """Score val with this cfg, find best threshold, return record."""
    fn = make_score_fn(template, cfg)
    scores, labels, _ = score_split(val_items, fn)
    thr, f1 = best_threshold_macro_f1(scores, labels, lo=sweep_lo, hi=sweep_hi)
    print(f"  variant={name}: val_macro_f1={f1:.4f} threshold={thr:.3f}")
    return {
        "name": name,
        "val_macro_f1": f1,
        "val_threshold": thr,
        "cfg": cfg,
    }


def cfg_summary(cfg: ScoreConfig) -> str:
    parts = [
        f"roi={cfg.roi_frac}",
        f"scales={cfg.scales}",
        f"hp={cfg.high_pass_sigma}",
        f"dt={cfg.use_distance_transform}",
        f"extra_tpls={len(cfg.extra_templates)}",
    ]
    return " ".join(parts)


def main():
    split = load_split()
    template = load_template_gray()
    val_items = split["val"]
    test_items = split["test"]

    real_paths = sorted(REAL_TEMPLATE_DIR.glob("real_template_*.png"))
    real_templates = load_templates(real_paths) if real_paths else []
    print(f"loaded {len(real_templates)} real-sample templates from {REAL_TEMPLATE_DIR}")

    # The watermark renders at ~0.20 of master template; spec ladder bottoms at 0.4.
    # Multi-template variant uses real-sample crops at native size, which
    # internally still uses cfg.scales (small relative to a 75x30 template).
    # For master-template-only variants, we only have the spec'd ladder.
    base = ScoreConfig(scales=tuple(SCALES))

    runs = []

    # V0: phase 1 baseline (re-eval on val for direct comparison)
    runs.append(eval_variant("V0_baseline", base, template, val_items))

    # V1: high-pass pre-filter
    runs.append(eval_variant("V1_highpass", replace(base, high_pass_sigma=3.0),
                              template, val_items))

    # V2: distance-transform matching
    runs.append(eval_variant("V2_distancetransform", replace(base, use_distance_transform=True),
                              template, val_items))

    # V3: ROI fraction sweep
    for rf in (0.12, 0.18, 0.25):
        runs.append(eval_variant(f"V3_roi{int(rf*100)}", replace(base, roi_frac=rf),
                                  template, val_items))

    # V4: multi-template (master + N real-sample crops). Real crops are small (~75x30)
    # so cfg.scales applied to them spans 0.4-2.0 -> 30x12 to 150x60: brackets the
    # observed ~75x30 rendered watermark. Master template kept as-is.
    if real_templates:
        runs.append(eval_variant("V4_multi_template", replace(base, extra_templates=real_templates),
                                  template, val_items))

    # Combos worth trying once individual ones are scored
    if real_templates:
        runs.append(eval_variant("V14_hp_multi",
                                  replace(base, high_pass_sigma=3.0, extra_templates=real_templates),
                                  template, val_items))
        runs.append(eval_variant("V124_hp_dt_multi",
                                  replace(base, high_pass_sigma=3.0, use_distance_transform=True,
                                          extra_templates=real_templates),
                                  template, val_items))

    # V5: master template + small scales added (0.18, 0.22, 0.30) — match the
    # observed rendered size from the master directly, no real templates.
    runs.append(eval_variant(
        "V5_master_smallscales",
        replace(base, scales=(0.18, 0.22, 0.30, 0.4, 0.6, 0.8, 1.0)),
        template, val_items))

    # V6: master + small scales + multi-template (everything that should help).
    if real_templates:
        runs.append(eval_variant(
            "V6_smallscales_multi",
            replace(base, scales=(0.18, 0.22, 0.30, 0.4, 0.6, 0.8, 1.0),
                    extra_templates=real_templates),
            template, val_items))

        # V7: same as V6 + high-pass
        runs.append(eval_variant(
            "V7_smallscales_multi_hp",
            replace(base, scales=(0.18, 0.22, 0.30, 0.4, 0.6, 0.8, 1.0),
                    extra_templates=real_templates, high_pass_sigma=3.0),
            template, val_items))

        # V8: real templates only, narrow scales (~1.0 ± 15%). Drops the master
        # template, since real crops already encode the rendered size.
        narrow = (0.85, 0.92, 1.0, 1.08, 1.15)
        runs.append(eval_variant(
            "V8_realonly_narrow",
            ScoreConfig(scales=narrow, extra_templates=real_templates[1:],
                        roi_frac=base.roi_frac),
            real_templates[0], val_items))

        # V9: V8 + high-pass
        runs.append(eval_variant(
            "V9_realonly_narrow_hp",
            ScoreConfig(scales=narrow, extra_templates=real_templates[1:],
                        roi_frac=base.roi_frac, high_pass_sigma=3.0),
            real_templates[0], val_items))

        # V10: master+real, narrow scales (master at 1.0 keeps the original
        # full-size template option).
        runs.append(eval_variant(
            "V10_narrow_multi",
            replace(base, scales=narrow, extra_templates=real_templates),
            template, val_items))

    # Persist to experiments.csv (append rows)
    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["phase", "variant", "roi_frac", "scales", "high_pass_sigma",
                        "use_dt", "n_extra_templates", "val_macro_f1", "val_threshold", "notes"])
        for r in runs:
            cfg = r["cfg"]
            w.writerow([
                "phase3", r["name"], cfg.roi_frac, str(cfg.scales),
                cfg.high_pass_sigma, cfg.use_distance_transform, len(cfg.extra_templates),
                f"{r['val_macro_f1']:.4f}", f"{r['val_threshold']:.3f}", "",
            ])

    # Pick best by val macro F1
    best = max(runs, key=lambda r: r["val_macro_f1"])
    print(f"\n>> best variant: {best['name']} val_macro_f1={best['val_macro_f1']:.4f} "
          f"thr={best['val_threshold']:.3f}")

    # Eval best on test
    fn = make_score_fn(template, best["cfg"])
    scores_, labels, paths = score_split(test_items, fn)
    metrics = report(scores_, labels, best["val_threshold"], name="test")

    out = ROOT / "debug" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    save_failure_montages(paths, labels, scores_, best["val_threshold"], out, tag="phase3")
    write_scores_csv(out / "test_scores.csv", paths, labels, scores_, best["val_threshold"])
    import json as _json
    with open(out / "test_metrics.json", "w") as f:
        _json.dump({**metrics, "variant": best["name"], "cfg_summary": cfg_summary(best["cfg"])}, f, indent=2)
    with open(out / "best_variant.txt", "w") as f:
        f.write(f"{best['name']}\nthreshold={best['val_threshold']}\n{cfg_summary(best['cfg'])}\n")
    print(f"saved phase3 artifacts under {out}")


if __name__ == "__main__":
    main()
