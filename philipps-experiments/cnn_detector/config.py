from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"

# Heads (multi-label) — each gets a sigmoid + per-class threshold.
CLASSES = ("grok", "gemini")

# Real positive folders per class.
POS_DIRS = {
    "grok":   IMAGES_DIR / "watermark_grok",
    "gemini": IMAGES_DIR / "watermark_gemini",
}

# Negatives: real photos with no watermark.
NEG_DIR = IMAGES_DIR / "no_watermark"

# Hard negatives: other watermarks, all labeled [0, 0] for the two heads.
HARD_NEG_DIRS = (
    IMAGES_DIR / "watermark_dalle",
    IMAGES_DIR / "watermark_minimax_hailuoAI",
    IMAGES_DIR / "watermark_openai_logo",
    IMAGES_DIR / "watermark_sora",
    IMAGES_DIR / "watermark_text_this-person-does-not-exist.com",
)

SPLITS_PATH = ROOT / "splits" / "cnn_split.json"
DEBUG_DIR = ROOT / "debug" / "cnn"
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_CSV = ROOT / "cnn_experiments.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ROI is BR 25% — slightly larger than the classical 18% to give shift augmentation headroom.
ROI_FRAC = 0.25
INPUT_SIZE = 192    # 192x192 ROI input

SEED = 42
