from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "images"
POS_DIR = IMAGES_DIR / "watermark_grok"
NEG_DIR = IMAGES_DIR / "no_watermark"
TEMPLATE_PATH = IMAGES_DIR / "watermarks" / "grok_watermark.png"
SPLITS_PATH = ROOT / "splits" / "split.json"
DEBUG_DIR = ROOT / "debug"
EXPERIMENTS_CSV = ROOT / "experiments.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ROI_FRAC = 0.18
CANNY_LO = 80
CANNY_HI = 160
SCALES = (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0)
SEED = 42
