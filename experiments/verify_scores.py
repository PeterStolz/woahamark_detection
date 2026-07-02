"""Philipps-style narrow-scale NCC verification scores for candidate groups."""
import sys, os, glob, json
sys.path.insert(0, "/home/peter/repos/detesia/woahamark_detection")
os.chdir("/home/peter/repos/detesia/woahamark_detection")
import cv2
import numpy as np
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)

# load extracted real templates -> per class list of Canny maps
TPL = {}
for p in glob.glob(os.path.join(SCRATCH, "real_templates", "*.png")):
    lab = os.path.basename(p).rsplit("_", 1)[0]
    g = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    TPL.setdefault(lab, []).append(g)

# dalle: master color template (BGR NCC per philipps)
DALLE = cv2.imread("images/watermarks/dalle_watermark.png", cv2.IMREAD_UNCHANGED)
if DALLE is not None and DALLE.shape[2] == 4:
    DALLE = DALLE[:, :, :3]

ROI = {  # (y_frac_start, y_frac_end, x_frac_start, x_frac_end)
    "grok": (0.70, 1.0, 0.60, 1.0),
    "gemini": (0.70, 1.0, 0.60, 1.0),
    "minimax_hailuo": (0.70, 1.0, 0.55, 1.0),
    "text_tpdne": (0.0, 0.18, 0.0, 1.0),
}

def ncc_score(img_bgr, label):
    """Max TM_CCOEFF_NORMED over templates x narrow scales on Canny maps, at width-1024 normalization."""
    h, w = img_bgr.shape[:2]
    s = 1024.0 / w
    img = cv2.resize(img_bgr, (1024, max(8, int(h * s))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    yf0, yf1, xf0, xf1 = ROI[label]
    roi = gray[int(H*yf0):int(H*yf1), int(W*xf0):int(W*xf1)]
    roi_e = cv2.Canny(roi, 80, 160)
    best = 0.0
    for tg in TPL.get(label, []):
        te0 = cv2.Canny(tg, 80, 160)
        th0, tw0 = te0.shape
        for sc in SCALES:
            tw, th = int(tw0*sc), int(th0*sc)
            if tw < 8 or th < 8 or th > roi_e.shape[0] or tw > roi_e.shape[1]:
                continue
            te = cv2.resize(te0, (tw, th))
            try:
                best = max(best, float(cv2.matchTemplate(roi_e, te, cv2.TM_CCOEFF_NORMED).max()))
            except cv2.error:
                pass
    return best

def dalle_score(img_bgr):
    """Multi-channel BGR NCC on bottom-right ROI, small scale ladder around rendered size (80px on 1024)."""
    if DALLE is None:
        return 0.0
    h, w = img_bgr.shape[:2]
    s = 1024.0 / w
    img = cv2.resize(img_bgr, (1024, max(8, int(h * s))))
    roi = img[int(img.shape[0]*0.85):, int(img.shape[1]*0.70):]
    best = 0.0
    for tw in (60, 70, 80, 90, 100):
        th = max(4, int(DALLE.shape[0] * tw / DALLE.shape[1]))
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue
        t = cv2.resize(DALLE, (tw, th))
        try:
            best = max(best, float(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max()))
        except cv2.error:
            pass
    return best

df = pd.read_parquet(os.path.join(SCRATCH, "pred_cache.parquet"))
CLASSES = [c[2:] for c in df.columns if c.startswith("p_")]
P = df[[f"p_{c}" for c in CLASSES]].values
df["gbt_pred"] = np.array(CLASSES)[P.argmax(1)]

groups = {
    "val_pos_noyolo": df[(df.set == "val") & (df.meta != "clean") & (df.yolo_conf <= 0.3)],
    "val_clean": df[(df.set == "val") & (df.meta == "clean")],
    "wild_clean_fp": df[(df.set == "wild") & df.meta.str.endswith("|clean") & (df.gbt_pred != "clean") & (df.yolo_conf <= 0.3)],
}

rows = []
for gname, g in groups.items():
    for _, r in g.iterrows():
        img = cv2.imread(r.path)
        if img is None:
            continue
        scores = {lab: ncc_score(img, lab) for lab in ROI}
        scores["dalle"] = dalle_score(img)
        rows.append({"group": gname, "path": r.path, "true": r.meta, "gbt_pred": r.gbt_pred,
                     "wm_prob": r.wm_prob, **{f"v_{k}": v for k, v in scores.items()}})
out = pd.DataFrame(rows)
out.to_csv(os.path.join(SCRATCH, "verify_scores.tsv"), sep="\t", index=False)

print("=== val positives without YOLO: verification score for TRUE class ===")
vp = out[out.group == "val_pos_noyolo"]
for _, r in vp.iterrows():
    print(f"true={r.true:16s} gbt={r.gbt_pred:16s} v_true={r[f'v_{r.true}']:.3f} v_grok={r.v_grok:.3f}")

print("\n=== negatives: max verification score across classes (want LOW) ===")
for gname in ("val_clean", "wild_clean_fp"):
    g = out[out.group == gname]
    vmax = g[[c for c in out.columns if c.startswith("v_")]].max(axis=1)
    print(f"{gname}: n={len(g)} q50={vmax.quantile(.5):.3f} q90={vmax.quantile(.9):.3f} q95={vmax.quantile(.95):.3f} max={vmax.max():.3f}")
    for lab in ["grok", "gemini", "minimax_hailuo", "text_tpdne", "dalle"]:
        print(f"   {lab:16s} q90={g['v_'+lab].quantile(.9):.3f} >0.27: {(g['v_'+lab]>0.27).mean():.2f}")
