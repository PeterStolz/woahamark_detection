"""Wild benchmark v2: bigger clean pool (fresh partitions) + fresh sora frames
disjoint from both the v1 benchmark and the v3 training pool."""
import glob, os, sys
import pyarrow.parquet as pq
import pandas as pd

IMAGES_ROOT = "/truenas-big/images"
CATALOG = "/truenas-big/catalog/images"
SCRATCH = os.path.dirname(os.path.abspath(__file__))

old = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")
EXCLUDE = set(os.path.basename(p).split(".")[0] for p in old.path)
# v3 training sora pool came from row len//3 onward of each parquet, first ~1200 found;
# to be safe, sample sora from the SECOND half of each file's tail region and skip
# anything already excluded (training set shas aren't recorded, so also skip the
# exact region: we take rows from 3/4 onward instead).

PARTS_CLEAN = {"journeydb": 120, "sagid": 100, "opensdi": 100, "adobe_firefly": 100,
               "flux2_synthetic": 100, "community_forensics": 100, "lexica_ai_images": 100,
               "gpt_4o": 80, "qwen": 80, "moviegen": 80, "x2edit": 80}

rows = []
for part, n in PARTS_CLEAN.items():
    files = sorted(glob.glob(f"{CATALOG}/source_dataset_name={part}/**/*.parquet", recursive=True))
    got = 0
    for f in files:
        if got >= n: break
        try:
            df = pq.read_table(f, columns=["sha256", "file_type"]).to_pandas()
        except Exception:
            continue
        df = df.iloc[len(df)//4: len(df)//2]  # region unused by any prior sampler
        step = max(1, len(df) // max(1, n - got))
        for _, r in df.iloc[::step].iterrows():
            if got >= n: break
            if r.sha256 in EXCLUDE: continue
            p = f"{IMAGES_ROOT}/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.{r.file_type or 'JPEG'}"
            if not os.path.exists(p):
                c = glob.glob(f"{IMAGES_ROOT}/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.*")
                if not c: continue
                p = c[0]
            rows.append({"path": p, "partition": part, "weak_label": "clean", "original_filename": "", "is_frame": False})
            got += 1
    print(part, got)

# fresh sora: from 3/4 onward of each parquet (training took len//3 onward, first 1200 paths —
# which covered early files; use the LAST files' tails)
files = sorted(glob.glob(f"{CATALOG}/source_dataset_name=sora/**/*.parquet", recursive=True))
got = 0
for f in reversed(files):
    if got >= 100: break
    df = pq.read_table(f, columns=["sha256", "file_type"]).to_pandas()
    df = df.iloc[3*len(df)//4:]
    for _, r in df.iterrows():
        if got >= 100: break
        if r.sha256 in EXCLUDE: continue
        p = f"{IMAGES_ROOT}/{r.sha256[:2]}/{r.sha256[2:4]}/{r.sha256}.{r.file_type or 'JPEG'}"
        if os.path.exists(p):
            rows.append({"path": p, "partition": "sora_fresh", "weak_label": "watermarked_sora_or_openai", "original_filename": "", "is_frame": True})
            got += 1
print("sora_fresh", got)

pd.DataFrame(rows).to_csv(os.path.join(SCRATCH, "wild_manifest_v2.tsv"), sep="\t", index=False)
print("total:", len(rows))
