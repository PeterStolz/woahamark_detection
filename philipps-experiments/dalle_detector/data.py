"""DALL·E split. Only 11 positives so we hand-enforce a 7/2/2 split with seed=42
shuffle, ensuring at least 2 positives in val and test."""
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

    rng.shuffle(pos)
    rng.shuffle(neg)

    # Custom positive split: ensure ≥2 in val and test (the dataset is 11 images).
    n_pos = len(pos)
    n_pos_val = max(2, int(round(n_pos * 0.15)))
    n_pos_test = max(2, int(round(n_pos * 0.15)))
    pos_te = pos[:n_pos_test]
    pos_va = pos[n_pos_test:n_pos_test + n_pos_val]
    pos_tr = pos[n_pos_test + n_pos_val:]

    # Negatives: standard 70/15/15
    n = len(neg)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    neg_tr = neg[:n_train]
    neg_va = neg[n_train:n_train + n_val]
    neg_te = neg[n_train + n_val:]

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
