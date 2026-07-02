"""Run the repo's detect.py on the wild catalog manifest; report per-partition behavior."""
import sys, os, time, json
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")

import pandas as pd
import prepare, detect

SCRATCH = os.path.dirname(os.path.abspath(__file__))

dataset = prepare.discover_images()
train_set, val_set = prepare.split_dataset(dataset)
templates = prepare.get_templates()
print("setup...", flush=True)
detect.setup(train_set=train_set, templates=templates)

df = pd.read_csv(os.path.join(SCRATCH, "wild_manifest.tsv"), sep="\t")
rows = []
t0 = time.time()
for i, r in df.iterrows():
    try:
        res = detect.detect(r.path)
    except Exception as e:
        res = {"binary": "error", "label": str(e)[:60], "confidence": 0.0}
    rows.append({"path": r.path, "partition": r.partition, "weak_label": r.weak_label,
                 "pred_binary": res.get("binary"), "pred_label": res.get("label"),
                 "confidence": res.get("confidence", 0.0),
                 "bbox": json.dumps(res.get("bbox")) if res.get("bbox") else ""})
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(df)} ({time.time()-t0:.0f}s)", flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(SCRATCH, "wild_results_exp14.tsv"), sep="\t", index=False)

print("\n=== per-partition prediction counts ===")
for part, g in out.groupby("partition"):
    wm = (g.pred_binary == "watermarked").mean()
    labs = g[g.pred_binary == "watermarked"].pred_label.value_counts().to_dict()
    print(f"{part:16s} n={len(g):3d} wm_rate={wm:.2f} {labs}")

clean = out[out.weak_label == "clean"]
print(f"\nFP rate on weak-clean: {(clean.pred_binary=='watermarked').mean():.3f} ({(clean.pred_binary=='watermarked').sum()}/{len(clean)})")
sora = out[out.partition == "sora"]
print(f"sora flagged as watermarked: {(sora.pred_binary=='watermarked').mean():.3f}")
print(f"avg time/img: {(time.time()-t0)/len(df)*1000:.0f}ms")
