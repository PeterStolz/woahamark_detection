"""Copy misclassified images per class into debug/cnn/full_eval/{cls}_{fn,fp}/."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .config import CLASSES, DEBUG_DIR


def main():
    scores_csv = DEBUG_DIR / "full_eval" / "all_scores.csv"
    out_root = DEBUG_DIR / "full_eval"
    rows = list(csv.DictReader(open(scores_csv)))

    for c in CLASSES:
        for kind in ("false_negatives", "false_positives"):
            d = out_root / f"{c}_{kind}"
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        fn_rows = sorted(
            [r for r in rows if int(r[f"label_{c}"]) == 1 and int(r[f"pred_{c}"]) == 0],
            key=lambda r: float(r[f"score_{c}"]))
        fp_rows = sorted(
            [r for r in rows if int(r[f"label_{c}"]) == 0 and int(r[f"pred_{c}"]) == 1],
            key=lambda r: -float(r[f"score_{c}"]))

        def copy_to(rows_, dst_dir):
            for i, r in enumerate(rows_):
                src = Path(r["path"])
                sc = float(r[f"score_{c}"])
                prefix = f"{i:03d}_s{sc:.3f}_{r['split']}_{r['source']}"
                safe = src.name.replace("/", "_")
                dst = dst_dir / f"{prefix}_{safe}"
                if len(dst.name) > 200:
                    dst = dst_dir / (dst.name[:190] + dst.suffix)
                shutil.copy2(src, dst)

        fn_dir = out_root / f"{c}_false_negatives"
        fp_dir = out_root / f"{c}_false_positives"
        copy_to(fn_rows, fn_dir)
        copy_to(fp_rows, fp_dir)
        print(f"[{c}] FN={len(fn_rows)} -> {fn_dir}, FP={len(fp_rows)} -> {fp_dir}")

        for path, rs in [(fn_dir / "_index.csv", fn_rows), (fp_dir / "_index.csv", fp_rows)]:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["rank", "score", "split", "source", "path"])
                for i, r in enumerate(rs):
                    w.writerow([i, f"{float(r[f'score_{c}']):.6f}", r["split"], r["source"], r["path"]])


if __name__ == "__main__":
    main()
