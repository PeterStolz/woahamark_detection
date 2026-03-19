# woahamark_detection

Autonomous AI-agent-driven research for visible watermark detection in AI-generated images and videos.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The idea: give an AI agent a real watermark detection problem with labeled data and let it experiment autonomously. It modifies the detection pipeline, runs evaluation, checks if results improved, keeps or discards, and repeats. You wake up to a log of experiments and (hopefully) a better detector.

## How it works

The repo has three files that matter:

* **`prepare.py`** — fixed constants, image loading, ground truth labels, evaluation harness. **Not modified by the agent.**
* **`detect.py`** — the single file the agent edits. Contains the watermark detection pipeline. Everything is fair game: preprocessing, detection algorithms, feature extraction, thresholds, ensembles, post-processing. **This file is edited and iterated on by the agent.**
* **`program.md`** — instructions for the agent. Describes the research context, constraints, and strategy. **This file is edited and iterated on by the human.**

Each experiment runs for a **fixed 2-minute time budget** (wall clock). The primary metric is **macro F1-score** across all classes — higher is better. This makes experiments directly comparable regardless of what approach the agent tries.

## The detection challenge

The `images/` directory contains labeled examples across multiple categories:

| Directory | Label | Description |
|-----------|-------|-------------|
| `no_watermark/` | clean | Images without any watermark |
| `watermark_dalle/` | dalle | DALL-E watermark |
| `watermark_gemini/` | gemini | Google Gemini watermark |
| `watermark_grok/` | grok | xAI Grok watermark |
| `watermark_minimax_hailuoAI/` | minimax_hailuo | Minimax/HailuoAI watermark |
| `watermark_openai_logo/` | openai_logo | OpenAI logo watermark |
| `watermark_sora/` | sora | OpenAI Sora watermark |
| `watermark_text_this-person-does-not-exist.com/` | text_tpdne | Text-based watermark |

The `images/watermarks/` directory contains **cropped reference watermark templates** for each type.

Two evaluation modes:
1. **Binary** — watermarked vs. clean (is there *any* watermark?)
2. **Multi-class** — which specific watermark is it? (or clean)

## Quick start

**Requirements:** Python 3.10+, [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

```bash
# 1. Create and activate the environment
micromamba create -f environment.yml
micromamba activate woahamark

# 2. Verify setup and run baseline
python prepare.py --verify

# 3. Run a single detection experiment (~2 min)
python detect.py
```

## Running the agent

Spin up Claude Code / Codex / Cursor in this repo, then prompt:

```
Read program.md and let's kick off a new experiment! Do the setup first.
```

## Evaluation

`prepare.py` handles all evaluation. It:
1. Loads all labeled images from `images/`
2. Splits into train/val (80/20, stratified, seeded)
3. Calls `detect.py`'s `detect(image_path) -> dict` for each val image
4. Computes metrics: accuracy, macro F1, per-class precision/recall/F1, confusion matrix
5. Reports binary and multi-class results
6. Enforces the 2-minute wall clock budget

The agent's goal is to maximize **macro F1 (multi-class)** as the primary metric, with **binary F1** as the secondary.

## Project structure

```
prepare.py        — constants, image loading, evaluation harness (do not modify)
detect.py         — detection pipeline (agent modifies this)
program.md        — agent instructions
environment.yml   — micromamba environment spec
images/           — labeled image dataset
  no_watermark/   — clean images
  watermark_*/    — watermarked images by type
  watermarks/     — cropped reference templates
results.tsv       — experiment log (untracked)
```

## Design choices

* **Single file to modify.** The agent only touches `detect.py`. Diffs stay reviewable.
* **Fixed time budget.** Every experiment gets exactly 2 minutes. Fast approaches that scan 100 images in 10s compete fairly with heavy approaches that take 90s.
* **Breadth-first.** The `program.md` encourages exploring fundamentally different approaches (OCR, template matching, CNN, edge analysis, frequency domain, etc.) before depth-optimizing any single one.
* **No GPU required.** The baseline is CPU-only. The agent may add GPU support if available, but CPU must remain the fallback.

## License

MIT
