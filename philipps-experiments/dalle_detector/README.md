# DALL·E color-bar watermark detector

CPU-only classical detector for the bottom-right DALL·E 2 color-strip watermark (yellow/cyan/green/orange/blue square sequence). Standalone — independent of `grok_detector/`, `gemini_detector/`, and the planned CNN.

## Final result

**ROC-AUC = 1.000 across all 667 images.** No false negative anywhere — every one of the 11 dalle positives ranks above every one of the 656 negatives. The val-tuned threshold (0.59) is slightly conservative, leading to 12 FPs across the full dataset; with only 2 val positives we cannot safely tighten further without leaking test data.

| split | n | pos | neg | macro F1 | AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| train | 466 | 7 | 459 | 0.7739 | 1.000 | 11 | 0 |
| val | 100 | 2 | 98 | 1.0000 | 1.000 | 0 | 0 |
| test | 101 | 2 | 99 | **0.8975** | **1.000** | 1 | 0 |
| **FULL (667)** | **667** | **11** | **656** | **0.8189** | **1.000** | **12** | **0** |

The macro F1 numbers are dragged down purely by precision-on-positives because there are only 11 dalle images total — every FP costs a lot. **AUC = 1.000 means the score is a perfect classifier with the right threshold**: lowest dalle score = 0.774 (across all 11), highest neg score = 0.672. Any threshold in `[0.673, 0.773]` would give zero errors. Val sweep landed at 0.59 because val negatives only reach 0.59; we don't know about the higher-scoring negatives until we look at the rest of the data, which we can't do without leaking.

## Approach

The dalle watermark is **5 highly-saturated colored squares** with very specific BGR values per band. Edge-based template matching (the trick used for grok / gemini) is weak here — only the 4 inter-square borders give edges, and a tiny rectangle of edges is easy to hit by chance. Color is the discriminative feature.

**Multi-channel `TM_CCOEFF_NORMED`** on the BGR template directly. OpenCV processes each channel independently and sums, which is exactly what we want: the cross-correlation only spikes where all three channels match in the right pattern.

Pipeline:
1. Bottom-right ROI 18% (no Canny).
2. For scale `s` in `(0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)`: resize template to 80s × 16s and run `cv2.matchTemplate(roi_bgr, tpl_bgr, TM_CCOEFF_NORMED)`. Track max.
3. Score = max NCC across scales.

That's it — no edge detection, no high-pass, no real-sample crops. The colors are specific enough that the master template works perfectly without modification.

## Variant log (val macro F1)

```
V0_baseline           1.0000  thr=0.59  gap=0.4033  ← shipped (largest val gap)
V1_satgate100         1.0000  thr=0.59  gap=0.2904
V2_satgate150         1.0000  thr=0.57  gap=0.0395
V3_roi12              1.0000  thr=0.59  gap=0.4033
V4_roi25              1.0000  thr=0.59  gap=0.4033
V5_satgate100_roi12   1.0000  thr=0.59  gap=0.2904
```

All six variants tied at val F1 = 1.000 — the val score-gap (lowest val pos − highest val neg) was the tiebreaker. Saturation-gating the ROI before matching reduced the gap (it suppressed not just background clutter but parts of the watermark too), so we shipped the unmodified V0.

## Final config

```python
# dalle_detector/detect.py
DEFAULT_THRESHOLD = 0.59
ROI_FRAC          = 0.18
SCALES            = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)
saturation_gate   = False
```

## Files

```
dalle_detector/
  config.py           # paths, ROI/scale defaults
  data.py             # custom 7/2/2 split (≥2 pos in val and test)
  detect.py           # ScoreConfig + score() + default_config_and_template()
  evaluate.py         # python -m dalle_detector.evaluate --split test
  cli.py              # python -m dalle_detector.cli <image>
  phase0.py           # inventory + split
  phase3_run.py       # variant runner with val-gap tiebreak
  eval_full.py        # full-folder eval
  copy_misclassified.py
  showcase.py
splits/dalle_split.json
dalle_experiments.csv
debug/dalle/
  phase0/  phase1/  phase3/  phase4/
  full_eval/
    all_scores.csv
    full_worst_fp.jpg
    false_positives/   # 12 imgs, prefixed with rank+score
  showcase.jpg
```

## How to run

```bash
# single-image classification
micromamba run -n deepfake-detector-app \
  python -m dalle_detector.cli "$(ls images/watermark_dalle/*.webp | head -1)"

# test-set eval (re-tunes threshold on val first)
micromamba run -n deepfake-detector-app python -m dalle_detector.evaluate --split test

# full-folder eval
micromamba run -n deepfake-detector-app python -m dalle_detector.eval_full

# copy misclassified into per-error folders
micromamba run -n deepfake-detector-app python -m dalle_detector.copy_misclassified

# best/worst showcase
micromamba run -n deepfake-detector-app python -m dalle_detector.showcase
```

## Known failure modes (all 12 FPs, 0 FNs)

Every FP is a high-saturation multi-colored patch in the bottom-right corner of an unrelated image. Examples in `debug/dalle/full_eval/false_positives/`:
- AP-press images with an "AP" red logo + colored chyron
- A "K + flushed-face emoji" meme — the emoji's pink/red and the white K give a multi-colored signature
- Medical / cosmetic photos with bright colored UI badges
- Stock-photo watermarks with logo coloration

All sit between 0.59 (val-tuned threshold) and 0.67 (highest neg). With 11 positives and ~656 negatives, a slightly higher threshold would clean these up — the AUC is already 1.000.

If a tighter threshold is wanted in production (e.g. 0.70 to eliminate all visible FPs while keeping all 11 TPs), it can be set via the `--threshold` flag; just note that this number is *not* val-tuned.

## Discipline notes

- Threshold 0.59 chosen on val only (sweep 0.05–0.95, step 0.02; tied variants ranked by val score-gap).
- No edits to `grok_detector/` or `gemini_detector/`.
- All paths via `dalle_detector/config.py`. Every script runs as `python -m dalle_detector.<x>`.
- Custom 7/2/2 split is deterministic: `seed=42` shuffle, then enforce `n_pos_val == n_pos_test == max(2, round(0.15 * n_pos))`. Saved to `splits/dalle_split.json`.
