"""Copy misclassified images into per-error folders inside debug/dalle/full_eval/."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

from .config import DEBUG_DIR


def main():
    scores_csv = DEBUG_DIR / "full_eval" / "all_scores.csv"
    out_root = DEBUG_DIR / "full_eval"
    fn_dir = out_root / "false_negatives"
    fp_dir = out_root / "false_positives"
    for d in (fn_dir, fp_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(scores_csv)))
    fn_rows = [r for r in rows if int(r["label"]) == 1 and int(r["prediction"]) == 0]
    fp_rows = [r for r in rows if int(r["label"]) == 0 and int(r["prediction"]) == 1]
    fn_rows.sort(key=lambda r: float(r["score"]))
    fp_rows.sort(key=lambda r: -float(r["score"]))

    def copy_one(r, idx, dst_dir):
        src = Path(r["path"])
        sc = float(r["score"])
        prefix = f"{idx:03d}_s{sc:.3f}_{r['split']}"
        safe = src.name.replace("/", "_")
        dst = dst_dir / f"{prefix}_{safe}"
        if len(dst.name) > 200:
            dst = dst_dir / (dst.name[:190] + dst.suffix)
        shutil.copy2(src, dst)

    print(f"copying {len(fn_rows)} false negatives -> {fn_dir}")
    for i, r in enumerate(fn_rows):
        copy_one(r, i, fn_dir)
    print(f"copying {len(fp_rows)} false positives -> {fp_dir}")
    for i, r in enumerate(fp_rows):
        copy_one(r, i, fp_dir)

    def write_index(path: Path, rows_):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "score", "split", "label", "prediction", "path"])
            for i, r in enumerate(rows_):
                w.writerow([i, f"{float(r['score']):.6f}", r["split"], r["label"],
                            r["prediction"], r["path"]])

    write_index(fn_dir / "_index.csv", fn_rows)
    write_index(fp_dir / "_index.csv", fp_rows)


if __name__ == "__main__":
    main()
