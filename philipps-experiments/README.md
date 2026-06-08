# Philipp's experiments — per-watermark detectors

A self-contained set of watermark detectors built alongside the main `woahamark_detection`
autoresearch loop, but with a **different methodology**. Where the parent repo iterates a single
multi-class `detect.py` against `prepare.py`'s harness, this folder contains **four standalone,
per-watermark detectors**, each developed and evaluated independently with its own deterministic
split and eval harness.

Nothing here touches the parent repo's `detect.py` / `prepare.py`. It is a separate body of work,
dropped in as an archive so the approaches and their measured performance are preserved.

## TL;DR — what's here and how it performs

| detector | watermark | method | type | test macro F1 | test AUC | full-folder macro F1 |
|---|---|---|---|---|---|---|
| `grok_detector/` | xAI Grok wordmark | Canny + real-template NCC, narrow scales | classical | **0.9291** | 0.9248 | 0.9039 |
| `gemini_detector/` | Google Gemini sparkle | Canny + real-template NCC, narrow scales | classical | **0.9095** | 0.9933 | 0.9081 |
| `dalle_detector/` | DALL·E color bars | multi-channel BGR NCC (color, not edges) | classical | **0.8975** | 1.0000 | 0.8189 |
| `cnn_detector/` | Grok **and** Gemini (multi-label) | MobileNetV3-small, ImageNet-pretrained, fine-tuned | learned | grok **0.9290** / gemini **0.8889** | grok 0.9937 / gemini 0.9472 | grok **0.9688** / gemini **0.9618** |

All figures are on **held-out test splits** (thresholds tuned on val only). "Full-folder" = every image
in the relevant class folders, including hard negatives.

## The one insight that made the classical detectors work

Each `images/watermarks/*.png` master template is **far larger than the watermark as actually
rendered** in real images. Matching the master directly fails:

| watermark | master size | rendered size | scale of master |
|---|---|---|---|
| Grok | 346 × 122 | ~75 × 30 px | ~0.20 |
| Gemini | 2144 × 2169 | ~22 × 27 px | ~0.011 |

With the spec's default scale ladder (min 0.4), the *smallest* template tried is still bigger than the
bottom-right ROI, so `cv2.matchTemplate` is skipped at every scale and **every score is 0.0** (degenerate
all-negative classifier, AUC 0.50). This is visible in the first rows of every `*_experiments.csv`.

**Fix:** crop real watermark instances out of train positives at native size (`extract_templates.py`),
then match those crops with a **narrow scale band** `(0.85, 0.92, 1.0, 1.08, 1.15)`. This was the single
biggest lever for both Grok (V4→V8) and Gemini (V0→V5), taking val macro F1 from ~0.46 to ~0.90–1.00.
Wider scales only multiply false-positive opportunities. High-pass and distance-transform pre-filters
were tried and **hurt** in both cases (see the variant logs).

DALL·E is the exception: its watermark is **5 saturated colored squares**, so *color* is discriminative,
not edges. There we run multi-channel `TM_CCOEFF_NORMED` on the BGR master directly — no Canny, no
real-sample crops needed (AUC = 1.000 out of the box).

## The CNN, and classical-vs-learned

`cnn_detector/` is a single multi-label MobileNetV3-small (two sigmoid heads: grok, gemini) fine-tuned
on Apple MPS. It exists because edge-based matching has a hard ceiling on **low-contrast watermarks**
(a faint light-gray Grok wordmark on a white background never enters the Canny edge map, so no ROI size
or scale ladder can recover it). The CNN doesn't depend on an edge map and handles those cases.

Head-to-head on the test split:

| metric | grok classical | **grok CNN** | gemini classical | **gemini CNN** |
|---|---|---|---|---|
| macro F1 | 0.9291 | 0.9290 (tied) | **0.9095** | 0.8889 |
| ROC-AUC | 0.9248 | **0.9937** | **0.9933** | 0.9472 |
| FP / FN | 0 / 7 | 8 / 1 | 5 / 1 | 6 / 2 |

Takeaways (also in `cnn_detector/README.md`):
- **Grok → prefer the CNN.** Same F1, far higher AUC (0.92 → 0.99) → much more robust to threshold choice,
  and it recovers the low-contrast FNs the classical detector structurally cannot. On the full 1330-image
  set the CNN gets 8 FN vs the classical detector's 61.
- **Gemini → prefer classical.** The sparkle is so specific that narrow-scale template matching edges out
  the CNN on both F1 and AUC.
- **DALL·E → classical only.** Color watermark is unmistakable; not worth a CNN (only 11 positives exist).

The CNN training journey (v1 broken at F1 0.55 → v2 plateau 0.63 → v3 0.91) is documented in
`cnn_detector/README.md`. Two breakthroughs: **bottom-right-anchored crop augmentation** (so heavy
augmentation can never crop the watermark out of frame) and switching from a **from-scratch CNN to a
pretrained backbone**.

## Layout

```
philipps-experiments/
  README.md                  ← this file
  grok_detector/             ← classical, Grok wordmark
  gemini_detector/           ← classical, Gemini sparkle  (has its own README)
  dalle_detector/            ← classical, DALL·E color bars (has its own README)
  cnn_detector/              ← MobileNetV3-small, grok+gemini (has its own README)
  splits/                    ← deterministic split JSONs (seed=42), one per detector
  checkpoints/cnn_v3.pt      ← trained CNN weights (force-added; *.pt is gitignored)
  experiments.csv            ← Grok variant log
  gemini_experiments.csv     ← Gemini variant log
  dalle_experiments.csv      ← DALL·E variant log
  requirements.txt           ← classical-detector deps (CNN additionally needs torch+torchvision)
  images/                    ← symlink to the shared dataset (gitignored; recreate locally)
```

Each detector is a Python package run as a module. They share a path convention:
`ROOT = Path(__file__).resolve().parent.parent`, i.e. paths resolve relative to **this folder**. They
expect `philipps-experiments/images/` (the symlink) and `philipps-experiments/splits/` to be present.

## Running it

The classical detectors run in the parent repo's `woahamark` env (numpy/opencv/scikit-learn). The CNN
additionally needs PyTorch — it was developed in a separate `deepfake-detector-app` env with
`torch`+`torchvision` and Apple MPS. Pick whichever env has torch for CNN commands.

First, make sure the dataset is reachable (the `images` symlink points at `../../images`):

```bash
cd philipps-experiments
ls images/   # should list no_watermark/, watermark_grok/, ... watermarks/
```

Then, from inside `philipps-experiments/`:

```bash
# --- classical detectors (no torch needed) ---
python -m grok_detector.evaluate   --split test     # Grok test metrics
python -m gemini_detector.evaluate --split test     # Gemini test metrics
python -m dalle_detector.evaluate  --split test     # DALL·E test metrics
python -m grok_detector.eval_full                   # full-folder eval + montages

python -m grok_detector.cli  images/watermark_grok/<some_image>
python -m dalle_detector.cli "$(ls images/watermark_dalle/* | head -1)"

# --- CNN (needs torch + torchvision) ---
python -m cnn_detector.evaluate --split test        # uses thresholds baked into cnn_v3.pt
python -m cnn_detector.cli   images/watermark_grok/<some_image>
python -m cnn_detector.eval_full                    # full-folder, both classes
# retrain (~8 min on Apple MPS):
python -m cnn_detector.train --epochs 25 --batch-size 32 --lr 5e-4 --tag cnn_v3 --patience 8
```

Per-detector READMEs (`gemini_detector/README.md`, `dalle_detector/README.md`,
`cnn_detector/README.md`) have the full variant tables, configs, and failure-mode analyses. Grok's
config is summarized below since it has no standalone README.

## Grok detector config (no per-detector README)

Shipped variant **V8_realonly_narrow** (`experiments.csv`):

```python
ROI_FRAC   = 0.18          # bottom-right 18%
CANNY      = (80, 160)
SCALES     = (0.85, 0.92, 1.0, 1.08, 1.15)
THRESHOLD  = 0.27          # tuned on val only
templates  = grok_detector/real_templates/real_template_{00,01,02}.png   # ~75×30 px, cropped from train positives
```

Score = max `TM_CCOEFF_NORMED` over (3 real templates × 5 scales) on the Canny edge map of the BR ROI.
Test: macro F1 0.9291, AUC 0.9248, FP 0 / FN 7. Full-folder (897 imgs): macro F1 0.9039, 1 FP / 61 FN.

**Known limitation (investigated):** the 61 full-folder FNs are dominated by **low-contrast watermarks**
(faint light-gray wordmark on white/smooth backgrounds) where the logo never registers in the Canny edge
map. This is a fundamental limit of edge matching, not an ROI-size problem — widening the ROI or scale
ladder only matches background edge noise. This is exactly why the CNN exists for Grok.
(`grok_detector/diagnose_fns.py` reproduces the per-FN diagnostic.)

## Discipline / reproducibility

- **Thresholds tuned on val only.** Test splits held out throughout. Where variants tie on val F1, the
  tiebreaker is the val score-gap (lowest positive − highest negative).
- **Deterministic splits.** `seed=42`, 70/15/15 stratified per source folder (DALL·E uses a custom 7/2/2
  that forces ≥2 positives into val and test, given only 11 positives exist). Saved in `splits/`.
- **No hardcoded absolute paths.** Everything derives from each package's `config.py`.
- **Every script runs as `python -m <package>.<module>`.**
- `debug/` montages, logs, and `results.tsv`-style outputs are **not committed** (regenerable via the
  `eval_full` / `showcase` / `copy_misclassified` scripts). The CNN weight `cnn_v3.pt` **is** committed
  (force-added past the `*.pt` gitignore) so the CNN runs from a fresh clone without retraining.
