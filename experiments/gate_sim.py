"""Simulate exp12 per-class verification gates on cached predictions + verify scores."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRATCH = os.path.dirname(os.path.abspath(__file__))
df = pd.read_parquet(os.path.join(SCRATCH, "pred_cache.parquet"))
vs = pd.read_csv(os.path.join(SCRATCH, "verify_scores.tsv"), sep="\t")
vcols = [c for c in vs.columns if c.startswith("v_")]
df = df.merge(vs[["path"] + vcols].drop_duplicates("path"), on="path", how="left")

CLASSES = [c[2:] for c in df.columns if c.startswith("p_")]
P = df[[f"p_{c}" for c in CLASSES]].values
CLEAN_IDX = CLASSES.index("clean")

val_mask = (df.set == "val").values
wild_clean_mask = ((df.set == "wild") & df.meta.str.endswith("|clean")).values
sora_mask = ((df.set == "wild") & df.meta.str.contains("sora")).values | (df.set == "sora_vid").values

def baseline_pred():
    pred = np.array(CLASSES)[P.argmax(1)].astype(object)
    nonclean = P.copy(); nonclean[:, CLEAN_IDX] = 0
    best_nc = np.array(CLASSES)[nonclean.argmax(1)]
    ov = (pred == "clean") & (df.wm_prob.values > 0.35) & (nonclean.max(1) > 0.005)
    pred[ov] = best_nc[ov]
    yo = df.yolo_conf.values > 0.3
    pred[yo] = df.yolo_label.values[yo]
    return pred

def gated_pred(gates):  # gates: {class: (vcol, tau)}
    pred = baseline_pred()
    no_yolo = df.yolo_conf.values <= 0.3
    for cls, (vcol, tau) in gates.items():
        v = df[vcol].fillna(1.0).values  # missing verify score (yolo-backed) -> pass
        kill = (pred == cls) & no_yolo & (v < tau)
        pred[kill] = "clean"
    return pred

def evaluate(pred, name):
    val_true = df.meta.values[val_mask]
    macro = f1_score(val_true, pred[val_mask], average="macro", zero_division=0)
    fp = (pred[wild_clean_mask] != "clean").mean()
    sora = (pred[sora_mask] != "clean").mean()
    print(f"{name:40s} val_macro={macro:.4f} wild_fp={fp:.3f} ({int(fp*wild_clean_mask.sum())}/480) sora_rec={sora:.3f}")
    return macro, fp

evaluate(baseline_pred(), "baseline")
for tg in [0.25, 0.30, 0.35]:
    for th in [0.22, 0.30, 0.35]:
        gates = {"grok": ("v_grok", tg), "minimax_hailuo": ("v_minimax_hailuo", th), "dalle": ("v_dalle", 0.9)}
        evaluate(gated_pred(gates), f"gate grok@{tg} hailuo@{th} dalle@0.9")
# grok-only gate
for tg in [0.25, 0.30, 0.35]:
    evaluate(gated_pred({"grok": ("v_grok", tg)}), f"gate grok@{tg} only")
# also gate gemini at high threshold
gates = {"grok": ("v_grok", 0.30), "minimax_hailuo": ("v_minimax_hailuo", 0.22), "dalle": ("v_dalle", 0.9),
         "gemini": ("v_gemini", 0.45)}
evaluate(gated_pred(gates), "gate g@.30 h@.22 d@.9 gem@.45")
