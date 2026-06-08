# Multi-label CNN watermark detector (grok + gemini)

PyTorch CNN trained on Apple MPS, two sigmoid heads. Standalone — independent of `grok_detector/`, `gemini_detector/`, `dalle_detector/`.

## Final result (test split, n=200; per-class as labelled)

| | grok | gemini |
|---|---|---|
| n positives | 36 | 18 |
| **macro F1** | **0.9290** | **0.8889** |
| ROC-AUC | 0.9937 | 0.9472 |
| pos P / R / F1 | 0.814 / 0.972 / 0.886 | 0.727 / 0.889 / 0.800 |
| neg P / R / F1 | 0.994 / 0.951 / 0.972 | 0.989 / 0.967 / 0.978 |
| FN / FP | 1 / 8 | 2 / 6 |

Test mean macro F1 across the two heads = **0.9090**.

## CNN vs classical (head-to-head, test split)

| metric | grok classical | **grok CNN** | gemini classical | **gemini CNN** |
|---|---|---|---|---|
| macro F1 | 0.9291 | **0.9290** (tied) | 0.9095 | 0.8889 |
| ROC-AUC | 0.9248 | **0.9937** (+0.07) | 0.9933 | 0.9472 |
| FP | 0 | 8 | 5 | 6 |
| FN | 7 | 1 | 1 | 2 |

The CNN trades off precision vs recall differently from the classical detectors (lower FN, higher FP). On Grok, AUC jumps from 0.92 → 0.99 — the CNN's score is a much stronger ranking signal even though final F1 ties. On Gemini, the classical color-aware narrow-scale template matcher remains slightly ahead in both F1 (0.91 vs 0.89) and AUC (0.99 vs 0.95) — that watermark is so specific that template matching is hard to beat.

## Full-folder eval (1330 images: 241 grok pos + 116 gemini pos + 656 plain neg + 317 hard neg)

| class | macro F1 | AUC | FN | FP |
|---|---|---|---|---|
| grok | 0.9688 | 0.9965 | 8 | 17 |
| gemini | 0.9618 | 0.9926 | 2 | 15 |

Train F1: 0.987 / 0.980. Val F1: 0.921 / 0.955. Test F1: 0.929 / 0.889. The train→val gap suggests mild overfitting — expected with ~250 grok / 100 gemini training positives.

## Training journey

Three runs, each addressing a real issue from the previous one. The "lessons learned" file is `experiments/cnn_journey.md`-equivalent — see below.

### v1 (from-scratch 1.5M-param CNN, aggressive augmentation)
- Test mean F1 = **0.55** — basically broken.
- Diagnosis: random crops within the BR window were *losing the watermark* (not BR-anchored). The model received "watermark-positive" labels on crops with no watermark in them, blocking learning.

### v2 (same model, BR-anchored crops, softer pos weights, longer training)
- Best at epoch 42, test mean F1 = **0.63**.
- Diagnosis: from-scratch CNN with 250 positives per class and heavy augmentation does not learn good features in any reasonable time. Loss kept dropping but val F1 plateaued at ≈ 0.74.

### v3 (MobileNetV3-small backbone, ImageNet weights, head fine-tune)
- Best at epoch 17, test mean F1 = **0.91**.
- Drop-in change of architecture from from-scratch → pretrained backbone. ImageNet features bootstrap the representation; the model only has to learn a small head on top. Reached the v2 final score in ~3 epochs and kept going.

## Architecture

- **Backbone**: `torchvision.models.mobilenet_v3_small(weights=IMAGENET1K_V1)` — 1M params total including head.
- **Head**: GAP → Linear(576, 128) → Hardswish → Dropout(0.2) → Linear(128, 2). Two sigmoid outputs, one per class.
- **Input**: 192×192 RGB ImageNet-normalized. Crop is the **bottom-right 18 % of the image at eval time**, anchored at the corner; **20–40 % at train time** (varied zoom level), still corner-anchored. Heavy augmentation can never drop the watermark out of frame.

## Training setup

```python
loss          = BCEWithLogitsLoss(pos_weight=sqrt(neg/pos) per class)
optimizer     = AdamW(lr=5e-4, weight_decay=1e-4)
schedule      = cosine over args.epochs
batch_size    = 32
augment       = BR-anchored crop ± zoom, ±2.5° rotation, ±20 % brightness/contrast,
                 random JPEG q60–95 (50 %)  --  no horizontal flip
patience      = 8 (early-stop on mean val macro F1 across the two classes)
device        = MPS
threshold     = per-class, tuned on val (0.05–0.95 step 0.02)
```

Best v3 thresholds at epoch 17: grok = 0.35, gemini = 0.33.

## Files

```
cnn_detector/
  config.py             # paths, classes, ROI 0.18 (eval) / 0.20-0.40 (train), input 192
  data.py               # split, BR-anchored crops, ImageNet-normalized augmentations
  model.py              # MobileNetV3-small backbone + small head + ImageNet stats
  train.py              # python -m cnn_detector.train --epochs 25 --tag cnn_v3
  evaluate.py           # python -m cnn_detector.evaluate --split test
  cli.py                # python -m cnn_detector.cli <image>
  phase0.py             # split + augmented batch sanity montage
  eval_full.py          # full-folder eval with per-class metrics
  copy_misclassified.py # copies FN/FP into per-class folders
  showcase.py           # best/worst per-class showcase montage
  README.md
splits/cnn_split.json
cnn_experiments.csv     # not auto-populated — see "training journey" above
debug/cnn/
  phase0/augmented_batch.jpg
  phase1_v3/                       # train_log + test scores + FN/FP montages for v3
  full_eval/
    all_scores.csv
    grok_false_negatives/    grok_false_positives/        # 8 FN, 17 FP
    gemini_false_negatives/  gemini_false_positives/      # 2 FN, 15 FP
  showcase_grok.jpg  showcase_gemini.jpg
checkpoints/cnn_v3.pt   # 1M-param MobileNetV3-small backbone + head, ~4MB
logs/cnn_{v1,v2,v3}_train.log
```

## How to run

```bash
# train (≈ 8 min on Apple MPS, M-series)
micromamba run -n deepfake-detector-app \
  python -m cnn_detector.train --epochs 25 --batch-size 32 --lr 5e-4 --tag cnn_v3 --patience 8

# single image — prints both class scores
micromamba run -n deepfake-detector-app \
  python -m cnn_detector.cli images/watermark_grok/Astronaut.webp

# test set (uses checkpoint thresholds; --retune to re-search on val)
micromamba run -n deepfake-detector-app \
  python -m cnn_detector.evaluate --split test

# full-folder eval (1330 images, both classes)
micromamba run -n deepfake-detector-app python -m cnn_detector.eval_full

# copy misclassified into per-class folders
micromamba run -n deepfake-detector-app python -m cnn_detector.copy_misclassified

# best/worst showcase per class
micromamba run -n deepfake-detector-app python -m cnn_detector.showcase
```

## When to use which detector

The classical detectors and the CNN target the same problems but differently. Recommended use:

| problem | recommended | why |
|---|---|---|
| Grok | **CNN** | Same F1, much higher AUC (0.99 vs 0.93) → more robust to threshold choice, better for ranking |
| Gemini | **classical** (`gemini_detector/`) | Slightly higher F1 + AUC; sparkle is so specific that template matching wins |
| DALL·E | **classical** (`dalle_detector/`) | Color watermark is unmistakable; no point training a CNN |
| New watermark with no clear template | CNN | Add a head to `model.py`, retrain |

You can also **ensemble**: classical for high precision, CNN for high recall, then OR the predictions. Untested but a natural extension.

## Known failure modes

**Grok FNs (8 total)**: same as the classical detector finds — out-of-distribution watermark variants ("GROK X" angular wordmark), images mislabeled as Grok-generated (Grok-app UI screenshots, hockey scenes), and a few low-contrast watermark cases.

**Grok FPs (17 total)**: sit between the val-tuned threshold and the score regime where TPs live. Many are images where text or logos in the BR corner have wordmark-shaped edges (newspaper/magazine layouts, "OPPT WIRD" memes, UI overlays). Notable: the CNN trips on slightly more of these than the classical detector because its features include "any text-like wordmark in the BR" rather than specifically the Grok logo.

**Gemini FNs (2 total)**: low-contrast positives (the sparkle on a near-uniform background). Less of an issue than the classical detector's failures.

**Gemini FPs (15 total)**: small bright glints, dots, and reflective highlights in the BR corner. Same family of FPs the classical detector produces, with slightly different distribution.

## Discipline

- Threshold per class is tuned on val only. Test held out.
- No edits to `grok_detector/`, `gemini_detector/`, or `dalle_detector/`.
- All paths via `cnn_detector/config.py`. Every script runs as `python -m cnn_detector.<x>`.
- Splits are deterministic: `seed=42` shuffle, 70/15/15 stratified per source folder. Saved to `splits/cnn_split.json`.
- Checkpoint saves model state + per-class thresholds + class names + epoch metadata.
