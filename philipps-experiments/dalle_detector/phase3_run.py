"""Phase 3: try cheap variants. Already val=1.0 from Phase 1; goal is test FP reduction."""
from __future__ import annotations

import csv
from dataclasses import replace

from .config import EXPERIMENTS_CSV, ROOT, SCALES
from .data import load_split
from .detect import ScoreConfig, load_template_bgr, score
from .evaluate import (
    best_threshold_macro_f1,
    report,
    save_failure_montages,
    score_split,
    write_scores_csv,
)


def make_score_fn(template, cfg):
    return lambda img: score(img, template, cfg)


def eval_variant(name, cfg, template, val_items):
    fn = make_score_fn(template, cfg)
    scores, labels, _ = score_split(val_items, fn)
    thr, f1 = best_threshold_macro_f1(scores, labels)
    print(f"  {name}: val_macro_f1={f1:.4f} threshold={thr:.3f}")
    return {"name": name, "val_macro_f1": f1, "val_threshold": thr, "cfg": cfg}


def main():
    split = load_split()
    template = load_template_bgr()
    val_items = split["val"]
    test_items = split["test"]

    base = ScoreConfig()
    runs = []
    runs.append(eval_variant("V0_baseline", base, template, val_items))
    runs.append(eval_variant("V1_satgate100", replace(base, saturation_gate=True, sat_min=100),
                              template, val_items))
    runs.append(eval_variant("V2_satgate150", replace(base, saturation_gate=True, sat_min=150),
                              template, val_items))
    runs.append(eval_variant("V3_roi12", replace(base, roi_frac=0.12), template, val_items))
    runs.append(eval_variant("V4_roi25", replace(base, roi_frac=0.25), template, val_items))
    runs.append(eval_variant("V5_satgate100_roi12",
                              replace(base, saturation_gate=True, sat_min=100, roi_frac=0.12),
                              template, val_items))

    EXPERIMENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["phase", "variant", "roi_frac", "scales", "saturation_gate", "sat_min",
                        "val_macro_f1", "val_threshold", "notes"])
        for r in runs:
            cfg = r["cfg"]
            w.writerow(["phase3", r["name"], cfg.roi_frac, str(cfg.scales),
                        cfg.saturation_gate, cfg.sat_min,
                        f"{r['val_macro_f1']:.4f}", f"{r['val_threshold']:.3f}", ""])

    # Tiebreak by largest val score-gap between positives and negatives:
    # we re-score val for each variant once more and compute the gap.
    def gap(cfg):
        sc, lab, _ = score_split(val_items, make_score_fn(template, cfg))
        import numpy as np
        s = np.asarray(sc); y = np.asarray(lab)
        if (y == 1).any() and (y == 0).any():
            return float(s[y == 1].min() - s[y == 0].max())
        return -1.0

    tied = [r for r in runs if abs(r["val_macro_f1"] - max(rr["val_macro_f1"] for rr in runs)) < 1e-9]
    print(f"\n{len(tied)} variants tied at val F1; using val score-gap as tiebreaker:")
    scored = [(gap(r["cfg"]), r) for r in tied]
    for g, r in scored:
        print(f"  gap={g:.4f}  {r['name']}")
    best = max(scored, key=lambda gr: gr[0])[1]
    print(f"\n>> best: {best['name']} val_f1={best['val_macro_f1']:.4f} thr={best['val_threshold']:.3f}")

    fn = make_score_fn(template, best["cfg"])
    scores_, labels, paths = score_split(test_items, fn)
    metrics = report(scores_, labels, best["val_threshold"], name="test")

    out = ROOT / "debug" / "dalle" / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    save_failure_montages(paths, labels, scores_, best["val_threshold"], out, tag="phase3")
    write_scores_csv(out / "test_scores.csv", paths, labels, scores_, best["val_threshold"])
    import json as _json
    with open(out / "test_metrics.json", "w") as f:
        _json.dump({**metrics, "variant": best["name"]}, f, indent=2)


if __name__ == "__main__":
    main()
