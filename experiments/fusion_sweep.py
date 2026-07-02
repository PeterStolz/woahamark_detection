"""Offline sweep of fusion rules over the prediction cache.
Objective: val macro_f1 >= baseline while minimizing wild FP rate."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(SCRATCH, "pred_cache.parquet"))
CLASSES = [c[2:] for c in df.columns if c.startswith("p_")]
P = df[[f"p_{c}" for c in CLASSES]].values
CLEAN_IDX = CLASSES.index("clean")

val_mask = (df.set == "val").values
wild_clean_mask = ((df.set == "wild") & df.meta.str.endswith("|clean")).values
sora_mask = ((df.set == "wild") & df.meta.str.contains("sora")).values | (df.set == "sora_vid").values
y_val = df.meta.where(~val_mask, df.meta).values  # val meta is the label itself

def apply_rule(τ_y, τ_b, τ_c, gbt_conf_min=0.005):
    """Vectorized re-implementation of detect() fusion with knobs."""
    pred = np.array(CLASSES)[P.argmax(1)].astype(object)
    # binary override: clean -> best nonclean if wm_prob>τ_b AND cnn gate passes
    nonclean = P.copy(); nonclean[:, CLEAN_IDX] = 0
    best_nc = np.array(CLASSES)[nonclean.argmax(1)]
    ov = (pred == "clean") & (df.wm_prob.values > τ_b) & (nonclean.max(1) > gbt_conf_min) & (df.cnn_prob.values > τ_c)
    pred[ov] = best_nc[ov]
    # cnn gate also applies to direct GBT nonclean prediction when no yolo support
    no_yolo = df.yolo_conf.values <= 0.05
    kill = (pred != "clean") & no_yolo & (df.cnn_prob.values <= τ_c)
    pred[kill] = "clean"
    # yolo override
    yo = df.yolo_conf.values > τ_y
    pred[yo] = df.yolo_label.values[yo]
    return pred

def evaluate(pred):
    val_true = df.meta.values[val_mask]
    val_pred = pred[val_mask]
    macro = f1_score(val_true, val_pred, average="macro", zero_division=0)
    fp = (pred[wild_clean_mask] != "clean").mean()
    sora_rec = (pred[sora_mask] != "clean").mean()
    return macro, fp, sora_rec

# baseline reproduction: τ_y=0.3, τ_b=0.35, τ_c=0 (no gate)
base = evaluate(apply_rule(0.3, 0.35, -1))
print(f"baseline repro: val_macro={base[0]:.4f} wildFP={base[1]:.3f} soraRec={base[2]:.3f}")

rows = []
for τ_y in [0.25, 0.3, 0.4, 0.5]:
    for τ_b in [0.35, 0.5, 0.7, 0.9]:
        for τ_c in [-1, 0.3, 0.5, 0.7, 0.9]:
            m, fp, sr = evaluate(apply_rule(τ_y, τ_b, τ_c))
            rows.append({"ty": τ_y, "tb": τ_b, "tc": τ_c, "val_macro": m, "wild_fp": fp, "sora_rec": sr})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(SCRATCH, "fusion_sweep.csv"), index=False)

good = res[res.val_macro >= base[0] - 1e-9].sort_values("wild_fp")
print("\n=== rules with val_macro >= baseline, sorted by wild FP ===")
print(good.head(15).to_string(index=False))
print("\n=== best wild_fp for small val sacrifices ===")
for tol in [0.0, 0.005, 0.01, 0.02]:
    g = res[res.val_macro >= base[0] - tol].sort_values("wild_fp").head(1)
    print(f"tol={tol}: {g.to_dict('records')}")
