"""Phase 3: Gemini variant runner. Mirrors grok_detector.phase3_run."""
from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

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


REAL_TEMPLATE_DIR = ROOT / "gemini_detector" / "real_templates"


def make_score_fn(template, cfg):
    return lambda img: score(img, template, cfg)


def eval_variant(name, cfg, template, val_items, sweep_lo=0.05, sweep_hi=0.95):
    fn = make_score_fn(template, cfg)
    scores, labels, _ = score_split(val_items, fn)
    thr, f1 = best_threshold_macro_f1(scores, labels, lo=sweep_lo, hi=sweep_hi)
    print(f"  variant={name}: val_macro_f1={f1:.4f} threshold={thr:.3f}")
    return {"name": name, "val_macro_f1": f1, "val_threshold": thr, "cfg": cfg}


def cfg_summary(cfg):
    return (f"roi={cfg.roi_frac} scales={cfg.scales} hp={cfg.high_pass_sigma} "
            f"dt={cfg.use_distance_transform} extra_tpls={len(cfg.extra_templates)}")


def main():
    split = load_split()
    template = load_template_gray()
    val_items = split["val"]
    test_items = split["test"]

    real_paths = sorted(REAL_TEMPLATE_DIR.glob("real_template_*.png"))
    real_templates = load_templates(real_paths) if real_paths else []
    print(f"loaded {len(real_templates)} real-sample templates")

    base = ScoreConfig(scales=tuple(SCALES))
    runs = []

    runs.append(eval_variant("V0_baseline_master_widescales", base, template, val_items))

    runs.append(eval_variant("V1_highpass_master",
                              replace(base, high_pass_sigma=3.0), template, val_items))

    runs.append(eval_variant("V2_distancetransform_master",
                              replace(base, use_distance_transform=True), template, val_items))

    for rf in (0.12, 0.18, 0.25):
        runs.append(eval_variant(f"V3_master_roi{int(rf*100)}",
                                  replace(base, roi_frac=rf), template, val_items))

    if real_templates:
        narrow = (0.85, 0.92, 1.0, 1.08, 1.15)
        wide_for_real = tuple(SCALES)  # 0.4..2.0 — for the small (27x27) real templates this is 11x11..54x54

        # V4: master + 3 real templates, master scales (will skip on master since it doesn't fit)
        runs.append(eval_variant("V4_multi_master_widescales",
                                  replace(base, extra_templates=real_templates),
                                  template, val_items))

        # V5: real-only, narrow scales
        runs.append(eval_variant("V5_realonly_narrow",
                                  ScoreConfig(scales=narrow, extra_templates=real_templates[1:]),
                                  real_templates[0], val_items))

        # V6: real-only, narrow scales + high-pass
        runs.append(eval_variant("V6_realonly_narrow_hp",
                                  ScoreConfig(scales=narrow,
                                              extra_templates=real_templates[1:],
                                              high_pass_sigma=3.0),
                                  real_templates[0], val_items))

        # V7: real-only, wider scales (covers some sparkle size variation)
        runs.append(eval_variant("V7_realonly_wider",
                                  ScoreConfig(scales=(0.7, 0.85, 1.0, 1.15, 1.3),
                                              extra_templates=real_templates[1:]),
                                  real_templates[0], val_items))

        # V8: V6 (HP) but with wider scales
        runs.append(eval_variant("V8_realonly_wider_hp",
                                  ScoreConfig(scales=(0.7, 0.85, 1.0, 1.15, 1.3),
                                              extra_templates=real_templates[1:],
                                              high_pass_sigma=3.0),
                                  real_templates[0], val_items))

        # V9: real-only narrow + ROI 12% (smaller area to suppress clutter)
        runs.append(eval_variant("V9_realonly_narrow_roi12",
                                  ScoreConfig(scales=narrow,
                                              extra_templates=real_templates[1:],
                                              roi_frac=0.12),
                                  real_templates[0], val_items))

        # V10: real-only narrow + ROI 25%
        runs.append(eval_variant("V10_realonly_narrow_roi25",
                                  ScoreConfig(scales=narrow,
                                              extra_templates=real_templates[1:],
                                              roi_frac=0.25),
                                  real_templates[0], val_items))

        # V11: real-only narrow + DT (expect bad)
        runs.append(eval_variant("V11_realonly_narrow_dt",
                                  ScoreConfig(scales=narrow,
                                              extra_templates=real_templates[1:],
                                              use_distance_transform=True),
                                  real_templates[0], val_items))

    # Append all runs to gemini_experiments.csv
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

    best = max(runs, key=lambda r: r["val_macro_f1"])
    print(f"\n>> best: {best['name']} val_macro_f1={best['val_macro_f1']:.4f} "
          f"thr={best['val_threshold']:.3f}")
    print(f"   cfg: {cfg_summary(best['cfg'])}")

    # Eval best on test
    fn = make_score_fn(template if best["cfg"].extra_templates is None else None, best["cfg"])
    # need a proper template — use the variant's primary template:
    # Reconstruct based on which variant we picked. Variants V5..V11 use real_templates[0].
    if best["name"].startswith(("V5", "V6", "V7", "V8", "V9", "V10", "V11")):
        primary = real_templates[0]
    else:
        primary = template
    fn = make_score_fn(primary, best["cfg"])
    scores_, labels, paths = score_split(test_items, fn)
    metrics = report(scores_, labels, best["val_threshold"], name="test")

    out = ROOT / "debug" / "gemini" / "phase3"
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
