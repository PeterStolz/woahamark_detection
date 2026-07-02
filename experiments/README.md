# experiments/ — wild benchmarks, training-data recipes, analyses (exp12–16)

Artifacts from the 2026-07-01/02 autoresearch session that took `detect.py` from
val macro F1 0.9666 → 0.9809 and wild false-positive rate ~40% → ~2.5%.
Everything here is either a *script* (deterministic, seeded) or a *manifest*
(sha256-addressed pointers into the data lake) — the pixels themselves are not
committed.

## Where the actual data lives

| artifact | location | notes |
|---|---|---|
| labeled dataset `images/` (1333 imgs) | untracked; Peter's & Philipp's machines | THE fragile artifact — see "reproducibility" below |
| data lake | `/truenas-big/images/<sha[:2]>/<sha[2:4]>/<sha256>.<ext>` | content-addressed, referenced by all manifests |
| catalog metadata | `/truenas-big/catalog/images/source_dataset_name=*/…parquet` | source of all wild sampling |
| YOLO v3 training data (yolo_ds_v3/, yolo_ds/) | `/truenas-big/misc/woahamark_detection/` | full copy incl. labels; rebuildable from scripts below |
| sora video frames + templates | `/truenas-big/misc/woahamark_detection/` | sora_mp4_frames/ (from images/watermark_sora mp4s), real_templates/, tick_candidates/ |
| trained weights | `yolo_watermark.pt` (v1), `yolo_watermark_v3.pt` (v3.2) | committed in repo root |

## Wild benchmark

- `build_wild_manifest.py` → `wild_manifest.tsv` — 540 images across 13 catalog
  partitions. `weak_label` column was manually corrected twice: only the `sora`
  partition carries visible watermarks (OpenAI flower + tick-row); 12 of the
  detector's "FPs" turned out to be real watermarks (creator signatures, stock
  marks, OpenAI tick-rows on gpt_4o images) — see `fp24_reviewed.tsv`.
- `build_wild_v2.py` → `wild_manifest_v2.tsv` — 1140-image expansion (11 more
  partitions + 100 fresh sora frames disjoint from v3 training).
- `wild_eval.py` — runs the repo pipeline over a manifest, reports per-partition
  flag rates. Point it at either manifest.
- `videoufo_stress.tsv` / `videoufo_results.tsv` — 120 caption-heavy real video
  frames (hard-negative stress set): ~2.5% true FP.
- `robustness_eval.py` / `robustness_results.tsv` — JPEG-50/30 and half-res
  degradation suite (motivated exp16's upscale retry).

## YOLO v3 training data recipe (deterministic, seeded)

1. `build_yolo_dataset.py` — synthetic overlays (7 classes) on ~3400 catalog
   backgrounds (benchmark-disjoint), visibility-checked; plus negatives.
2. `add_real_positives.py` — real train-split positives with YOLO-v1 pseudo-boxes.
3. `v3_data.py` — assembles yolo_ds_v3: drops wrong-style sora synthetics, adds
   207 real sora watermark crops pseudo-labeled from fresh catalog sora frames
   (v1 boxes at conf>0.3), tick/flower composites, subsampled negatives.
4. Training: YOLO11n from `yolo_watermark.pt`, imgsz=1280, scale=0.2, degrees=0,
   erasing=0 (see commit messages of exp13/exp15 for epochs).

Traps discovered (do not repeat): master watermark templates ≠ rendered
appearance (philipps' insight, confirmed); synthetic-only positives + hard
negatives destroy real-watermark recall (v2 failure); v1's tpdne head
pseudo-labels CCTV timestamps as watermarks; whole-frame NCC and periodicity
verifiers are non-discriminative for tick/flower marks.

## Analyses

- `build_cache.py` / `fusion_sweep.py` / `gate_sim.py` / `verify_scores.py` —
  the offline prediction-cache workflow behind exp12: cache all model outputs
  once, sweep fusion rules/thresholds in seconds instead of 10-minute harness runs.
- `wild_results_*.tsv` — per-experiment wild predictions for diffing.

## Reproducibility notes

The one artifact git does not protect is the labeled `images/` dataset. Options
discussed: DVC with a remote on the NAS, or simply an rsync'd copy +
sha256 manifest under `/truenas-big/misc/woahamark_detection/`. Everything else
(training sets, benchmarks) is reconstructible from these scripts + the
content-addressed lake, so committing manifests is sufficient for those.
