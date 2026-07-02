"""Sample catalog partitions into a wild-benchmark manifest with weak labels."""
import glob, os, sys
import pyarrow.parquet as pq
import pandas as pd

IMAGES_ROOT = "/truenas-big/images"
CATALOG = "/truenas-big/catalog/images"

# partition -> (weak_label, n_samples)
# weak_label = what watermark we EXPECT (may be absent; verify visually)
PARTITIONS = {
    "sora": ("sora", 60),
    "nano_150k": ("gemini", 60),
    "pico_banana": ("gemini", 40),
    "dalle3": ("dalle", 40),
    "gpt_4o": ("clean_ai", 40),
    "coco_2017": ("clean", 40),
    "openimages": ("clean", 40),
    "laion400m": ("clean", 40),
    "civitai": ("clean_ai", 40),
    "ideogram_scrape": ("clean_ai", 40),
    "hunyuan": ("clean_ai", 30),
    "wan": ("clean_ai", 30),
    "shareveo3": ("veo", 40),
}

rows = []
for part, (weak, n) in PARTITIONS.items():
    files = sorted(glob.glob(f"{CATALOG}/source_dataset_name={part}/**/*.parquet", recursive=True))
    if not files:
        print(f"WARN no parquet for {part}", file=sys.stderr)
        continue
    got = 0
    for f in files:
        if got >= n:
            break
        t = pq.read_table(f, columns=["sha256", "file_type", "original_filename", "is_frame"])
        df = t.to_pandas()
        # deterministic sample spread across the file
        step = max(1, len(df) // max(1, (n - got)))
        df = df.iloc[::step]
        for _, r in df.iterrows():
            if got >= n:
                break
            s = r.sha256
            ext = r.file_type or "JPEG"
            path = f"{IMAGES_ROOT}/{s[:2]}/{s[2:4]}/{s}.{ext}"
            if not os.path.exists(path):
                # try common ext variants
                cands = glob.glob(f"{IMAGES_ROOT}/{s[:2]}/{s[2:4]}/{s}.*")
                if not cands:
                    continue
                path = cands[0]
            rows.append({"path": path, "partition": part, "weak_label": weak,
                         "original_filename": r.original_filename, "is_frame": r.is_frame})
            got += 1
    print(f"{part}: {got}/{n}")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "wild_manifest.tsv"), sep="\t", index=False)
print(f"total: {len(out)}")
