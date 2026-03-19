# Woahamark Detection — Agent Program

You are an autonomous computer vision researcher. Your goal: build the best possible visible watermark detector by iteratively modifying `detect.py`.

## Setup (do this first)

1. Create the branch: `git checkout -b woahamark/<tag>` from current master.
2. Read the in-scope files:
   - `README.md` — repository context
   - `prepare.py` — evaluation harness, metrics, dataset loading. **Do not modify.**
   - `detect.py` — the file you modify. Detection pipeline, preprocessing, features, models.
3. Verify data exists: `python prepare.py --verify`. Check that images/ contains labeled samples across categories and templates.
4. Initialize `results.tsv` with the header: `experiment\ttag\tmacro_f1\tbinary_f1\tmulti_acc\ttime_s\tnotes`
5. Run the baseline: `python prepare.py --run > run.log 2>&1`. Record the baseline scores. These will be terrible (random).
6. Confirm setup looks good. Once confirmed, begin experimentation.

## Experiment loop

Each experiment follows this cycle:

1. **Think** — Pick a detection approach or improvement. Prefer approaches you haven't tried yet (breadth-first).
2. **Edit** — Modify `detect.py` to implement the idea. Only touch this file. You may install new packages via `micromamba install -n woahamark <pkg>` or `pip install <pkg>`, but note them in the results.
3. **Run** — `python prepare.py --run > run.log 2>&1` (redirect everything, do NOT let output flood your context).
4. **Read** — `grep "^macro_f1:\|^binary_f1:\|^multi_acc:\|^time_s:" run.log` to get results. If empty, the run crashed — `tail -n 50 run.log` for the traceback.
5. **Record** — Append the result to `results.tsv` (leave untracked by git).
6. **Decide** —
   - If `macro_f1` improved → `git add detect.py && git commit -m "..."` to advance.
   - If worse or equal → `git checkout -- detect.py` to revert.
7. Repeat.

## What you optimize

- **Primary metric: `macro_f1`** (multi-class F1 averaged across all classes). Higher is better.
- **Secondary metric: `binary_f1`** (watermark vs. clean). Use as tiebreaker.
- **Time constraint: 2 minutes** wall clock for the entire val set. Budget your per-image time.

## Research context

You are detecting **visible watermarks** overlaid on images. These include:
- Logos and icons placed by AI image generators
- Text overlays added by web services or generators
- Semi-transparent badges, often in corners or edges
- Platform-specific marks from social media or content tools

The `images/watermarks/` directory contains cropped reference templates of known watermarks. The `images/watermark_*/` directories contain full images with those watermarks present. `images/no_watermark/` has clean images.

## Approach guidance

**Explore broadly before optimizing deeply.** Try fundamentally different techniques before tuning any single one. Here are research directions — investigate at least 4-5 different families before depth-tuning:

### Signal-level analysis
- Watermarks are often overlaid with reduced opacity, creating pixel value distributions distinct from natural image content
- Corner and edge regions of images are common watermark locations — focusing analysis on these zones can be efficient
- Watermarks tend to create local contrast or texture anomalies against the underlying image

### Text and character detection
- Many watermarks contain recognizable text strings or URL fragments
- OCR engines vary significantly in their ability to read overlaid/semi-transparent text
- Preprocessing (contrast enhancement, binarization, region cropping) dramatically affects OCR accuracy on watermarks

### Reference-based matching
- When you have known watermark templates (you do — check `images/watermarks/`), comparing against references can be very effective
- Feature-based matching handles scale/rotation better than pixel-level matching
- Edge maps or gradient representations can be more robust to transparency/color changes than raw pixels

### Spatial frequency analysis
- Overlaid watermarks alter the frequency spectrum of an image in characteristic ways
- High-frequency components may reveal sharp text/logo edges that differ from natural textures
- DCT or wavelet coefficients in watermark regions show different statistics than natural image regions

### Learned features
- Pre-trained image classifiers (even small ones) extract features that may discriminate watermarked from clean images
- Transfer learning from general image models can bootstrap a detector with very few training samples
- The train split gives you labeled examples — consider using them if your approach benefits from supervised learning

### Structural and geometric cues
- Watermarks often have unnaturally regular geometric structure (straight lines, perfect circles, uniform spacing)
- Connected component analysis can isolate overlaid elements that don't belong to the natural scene
- Symmetry detection may flag logos

### Ensemble and voting
- Combining weak signals (OCR found text? + corner anomaly? + template match score?) often outperforms any single signal
- Each approach may excel on different watermark types — an ensemble covers more ground
- Confidence calibration matters when fusing heterogeneous signals

## Rules

- Only modify `detect.py`. Never touch `prepare.py`.
- The `detect(image_path)` function signature must not change.
- The optional `setup(train_set, templates)` function is called once before evaluation — use it for expensive one-time work (model loading, template preprocessing, etc.).
- Keep experiments atomic — one idea per experiment.
- If an approach requires a new dependency, install it and note it in the commit message.
- If something crashes more than twice, move on to a different approach.
- After every 5 experiments, review `results.tsv` and briefly reflect on which families of approaches have worked best and what's unexplored.
- **Do not overfit** — the val set is held out, but it's small. Prefer approaches that generalize.
- Commit `detect.py` only. Leave `results.tsv`, `run.log`, and any generated files untracked.

## Performance targets (rough guide)

| Level | Macro F1 | Notes |
|-------|----------|-------|
| Terrible | < 0.20 | Random baseline territory |
| Weak | 0.20–0.40 | Single simple heuristic |
| Decent | 0.40–0.60 | One approach working well |
| Good | 0.60–0.80 | Multiple signals combined |
| Strong | 0.80–0.95 | Well-tuned ensemble |
| Excellent | > 0.95 | Production-grade |

You should aim for at least "Good" before stopping breadth exploration. Depth-tune only after trying at least 5 fundamentally different approach families.
