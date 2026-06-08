"""Phase 0: inventory + 7/2/2 split for DALL·E."""
from collections import Counter

from .config import NEG_DIR, POS_DIR, SEED
from .data import list_images, make_split, save_split


def main():
    for label, d in [("pos_dalle", POS_DIR), ("neg_real", NEG_DIR)]:
        files = list_images(d)
        exts = Counter(p.suffix.lower() for p in files)
        print(f"[{label}] {d}: {len(files)} files, exts={dict(exts)}")

    split = make_split(SEED)
    save_split(split)
    print("\nsplit saved to splits/dalle_split.json")
    for name in ("train", "val", "test"):
        items = split[name]
        n_pos = sum(1 for _, l in items if l == 1)
        n_neg = sum(1 for _, l in items if l == 0)
        print(f"  {name}: total={len(items)} pos={n_pos} neg={n_neg}")


if __name__ == "__main__":
    main()
