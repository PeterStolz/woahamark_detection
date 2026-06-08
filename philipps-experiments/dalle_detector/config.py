from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
POS_DIR = IMAGES_DIR / "watermark_dalle"
NEG_DIR = IMAGES_DIR / "no_watermark"
TEMPLATE_PATH = IMAGES_DIR / "watermarks" / "dalle_watermark.png"
SPLITS_PATH = ROOT / "splits" / "dalle_split.json"
DEBUG_DIR = ROOT / "debug" / "dalle"
EXPERIMENTS_CSV = ROOT / "dalle_experiments.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ROI_FRAC = 0.18
SCALES = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)
SEED = 42
