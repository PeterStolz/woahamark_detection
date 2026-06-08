# Gemini sparkle watermark detector

CPU-only binary classifier for the bottom-right Gemini 4-pointed-sparkle watermark. Standalone — independent of `grok_detector/`. No deep learning.

## Final result

Locked config: **V5_realonly_narrow** — three real-sample sparkle crops (27×27 px), narrow scale band, 18% bottom-right ROI, threshold 0.41 (val-tuned).

| split | n | pos | neg | macro F1 | AUC | FP | FN |
|---|---|---|---|---|---|---|---|
| train | 540 | 81 | 459 | 0.8894 | 0.940 | 21 | 11 |
| val | 115 | 17 | 98 | **1.0000** | 1.000 | 0 | 0 |
| test | 117 | 18 | 99 | **0.9095** | 0.993 | 5 | 1 |
| **FULL (all 772)** | **772** | **116** | **656** | **0.9081** | **0.957** | **26** | **12** |

Test split: macro F1 = 0.9095, AUC = **0.9933**, recall on positives = 0.944. The five test FPs all sit just over the 0.41 threshold (0.41–0.53).

## Approach (mirrors what worked for Grok)

1. **Master template is way too big.** `gemini_watermark.png` is 2144×2169; the rendered sparkle in real images is ~22×27 px at scale ~0.011 of the master. The Phase 1 spec ladder (min 0.4) means the smallest tried template is 858 px wide — bigger than the entire 184×184 ROI. `cv2.matchTemplate` skipped at every scale, so every Phase 1 score = 0.0 (AUC = 0.50).
2. **Use real-sample sparkle crops at native size.** `extract_templates.py` runs a wide-scale (0.006–0.030) match across train positives, picks the highest-confidence three from distinct source images, and writes them as `gemini_detector/real_templates/real_template_{00,01,02}.png`.
3. **Narrow scale band on real templates only.** Scales `(0.85, 0.92, 1.0, 1.08, 1.15)` ≈ 23×23 to 31×31 px. Wider scales only multiply false-positive opportunities.
4. **No high-pass / no distance-transform.** Both lowered val F1 here (HP went 1.0000 → 0.9823, DT went 1.0000 → 0.9469). The sparkle is a high-contrast white star on tinted backgrounds; Canny already extracts it cleanly.

Final scoring: max NCC across (3 templates × 5 scales) = 15 small `matchTemplate` calls per image.

## Final config

```python
# gemini_detector/detect.py
DEFAULT_NARROW_SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)
DEFAULT_THRESHOLD     = 0.41        # tuned on val only

# ScoreConfig
roi_frac           = 0.18
canny_lo, canny_hi = 80, 160
high_pass_sigma    = None           # disabled
use_distance_transform = False      # disabled
```

## Files

```
gemini_detector/
  config.py
  data.py                  # 70/15/15 stratified split, seed=42
  detect.py                # ScoreConfig + score() + default_config_and_template()
  evaluate.py              # python -m gemini_detector.evaluate --split test
  cli.py                   # python -m gemini_detector.cli <image_path>
  phase0.py                # inventory + split
  phase2_diagnose.py       # ROI/Canny montages + wide-scale probe
  phase3_run.py            # variant runner -> gemini_experiments.csv
  extract_templates.py     # automated real-sample crop harvest
  eval_full.py             # full-folder eval
  copy_misclassified.py    # sort FN/FP into folders
  showcase.py              # best/worst montage with Canny views
  real_templates/          # 3 cropped sparkle templates from train positives

splits/gemini_split.json
gemini_experiments.csv
debug/gemini/
  phase0/  phase1/  phase2/notes.md  phase3/  phase4/
  full_eval/
    all_scores.csv
    full_worst_fn.jpg, full_worst_fp.jpg
    false_negatives/   # 12 mis-classified positives, prefixed with rank+score
    false_positives/   # 26 mis-classified negatives
  showcase.jpg
```

## How to run

```bash
# single image
micromamba run -n deepfake-detector-app \
  python -m gemini_detector.cli "images/watermark_gemini/Generated Image September 07, 2025 - 11_07AM.jpeg"

# test-split eval (re-tunes threshold on val first)
micromamba run -n deepfake-detector-app python -m gemini_detector.evaluate --split test

# Phase 1 baseline reproduction (master template, wide scales)
micromamba run -n deepfake-detector-app python -m gemini_detector.evaluate --split test --baseline

# full-folder eval + montages + scores csv
micromamba run -n deepfake-detector-app python -m gemini_detector.eval_full

# copy misclassified images into per-error folders
micromamba run -n deepfake-detector-app python -m gemini_detector.copy_misclassified

# best/worst showcase (uses phase4 test_scores.csv)
micromamba run -n deepfake-detector-app python -m gemini_detector.showcase

# re-extract real-sample templates from train positives
micromamba run -n deepfake-detector-app python -m gemini_detector.extract_templates --n 3 --min-score 0.55
```

## Variant log (val macro F1, 17 positives / 98 negatives)

| variant | val macro F1 | notes |
|---|---|---|
| V0 baseline (master, scales 0.4–2.0) | 0.4601 | degenerate — template never fits ROI |
| V1 / V2 / V3 (HP, DT, ROI sweep on master) | 0.4601 | same — master is just too big |
| V4 master + 3 real templates, wide scales | 0.9637 | first big jump — real templates unlock matching |
| **V5 real-only, narrow scales** | **1.0000** | **shipped** |
| V6 V5 + high-pass | 0.9823 | HP slightly hurts |
| V7 real-only, scales 0.7–1.3 | 0.9832 | wider band loses ~0.02 |
| V9 V5 + ROI 12% | 1.0000 | tied; 18% kept (most defensible default) |
| V10 V5 + ROI 25% | 1.0000 | tied; 25% adds clutter for marginal gain |
| V11 V5 + DT | 0.9469 | DT hurts (same as Grok) |

Full table in `gemini_experiments.csv`. Three variants tied at val 1.0; we shipped the simplest (V5: ROI 18%, narrow scales, no HP/DT, no master template).

## Known failure modes

**12 false negatives across the full 116 positives.** Sample inspection: smooth-luminance backgrounds where the translucent white sparkle has too little Canny contrast to register, plus a couple where the BR ROI clips the sparkle. Threshold could be lowered to recover most of them, but with the small val/test positive pool (17/18) we conservatively kept the val-optimum.

**26 false positives across the full 656 negatives** (4.0% FP rate, vs Grok's 0.15%). The Gemini sparkle is geometrically simple — a 4-stroke, 4-fold-symmetric concave star — so its Canny pattern is easy to mimic. Common triggers visible in `debug/gemini/full_eval/false_positives/`:
- Press-wire text overlays / magazine BR text (letterforms with crossing strokes mimic the sparkle)
- Decorative graphics (tattoo / face-paint, jewelry, snowflake or asterisk-like marks)
- Small bright glints / lens flares in CG art
- Logo/icon shapes with diagonal strokes

These are expected from the diagnosis (Phase 2 notes) — there is no realistic way to remove them via Canny edge matching alone without losing real positives. Above 0.97 macro F1 would likely require either (a) color-aware matching that exploits the sparkle's faint white tint, or (b) a small CNN.

## Discipline notes

- Threshold 0.41 chosen on val only (best of 0.05–0.95 sweep, 0.02 step). Test held out.
- Every script runs as `python -m gemini_detector.<x>`. No hardcoded absolute paths in committed files (paths derive from `gemini_detector/config.py`).
- No edits to `grok_detector/` were made.
