import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .config import IMAGE_EXTS, NEG_DIR, POS_DIR, SEED, SPLITS_PATH


def list_images(d: Path) -> List[Path]:
    return sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def make_split(seed: int = SEED) -> Dict[str, List[Tuple[str, int]]]:
    pos = list_images(POS_DIR)
    neg = list_images(NEG_DIR)
    rng = random.Random(seed)

    def split_one(items):
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    pos_tr, pos_va, pos_te = split_one(pos)
    neg_tr, neg_va, neg_te = split_one(neg)

    def to_entries(p_paths, n_paths):
        return [(str(p), 1) for p in p_paths] + [(str(p), 0) for p in n_paths]

    return {
        "train": to_entries(pos_tr, neg_tr),
        "val": to_entries(pos_va, neg_va),
        "test": to_entries(pos_te, neg_te),
    }


def save_split(split, path: Path = SPLITS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)


def load_split(path: Path = SPLITS_PATH) -> Dict[str, List[Tuple[str, int]]]:
    with open(path) as f:
        raw = json.load(f)
    return {k: [(p, int(l)) for p, l in v] for k, v in raw.items()}
