"""
detect.py — Watermark detection + localization pipeline.
THIS FILE IS MODIFIED BY THE AGENT. Everything is fair game.

Experiment 12: localized-evidence verification gate. The GBT fallback is
confidently wrong on out-of-distribution clean images (40% FP rate on
in-the-wild data), so non-clean predictions without YOLO support must now
pass a narrow-scale NCC verification against real watermark crops
(auto-extracted from train positives via YOLO at setup). Verification uses
Canny edge maps for grok/gemini/hailuo and multi-channel color NCC for
dalle. Cuts wild FP 39.6%->11.3% while raising val macro F1 0.9666->0.9764
(the same gate kills the val clean->grok false positives).
"""

import numpy as np
from PIL import Image
import cv2
import torch
import timm
import torchvision.transforms as T
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path

TRAIN_SET = []
TEMPLATES = {}
MODEL = None
BINARY_MODEL = None
TEMPLATE_INFO = []
WM_CNN = None
WM_CNN_TRANSFORM = None
WM_CNN_DEVICE = None
YOLO_MODEL = None
YOLO_DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

YOLO_CLASSES = ['dalle', 'gemini', 'grok', 'minimax_hailuo', 'text_tpdne']

TEMPLATE_LABEL_MAP = {
    "dalle_watermark": "dalle",
    "gemini_watermark": "gemini",
    "grok_watermark": "grok",
    "hailuoai_watermark": "minimax_hailuo",
    "hailuoaixminimax_watermark": "minimax_hailuo",
    "minimax_watermark": "minimax_hailuo",
    "openai_watermark": "openai_logo",
    "sora_watermark": "sora",
    "this-person-does-not-exist_watermark": "text_tpdne",
}


# ── Template matching (kept for GBT fallback) ──

def preprocess_templates(templates):
    info = []
    for name, path in templates.items():
        tmpl = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if tmpl is None: continue
        if len(tmpl.shape) == 3 and tmpl.shape[2] == 4:
            a = tmpl[:,:,3].astype(float)/255.0
            g = cv2.cvtColor(tmpl[:,:,:3], cv2.COLOR_BGR2GRAY)
            gm = (g.astype(float)*a).astype(np.uint8)
        elif len(tmpl.shape) == 3:
            gm = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
        else:
            gm = tmpl
        edges = cv2.Canny(gm, 30, 100)
        oh, ow = gm.shape
        tws = [15,25,40,60,80] if ow>1000 else ([60,100,160,220,300] if ow>300 else [40,70,110,160,200])
        scales = []
        for tw in tws:
            s=tw/ow; th=max(int(oh*s),5)
            if th<5: continue
            scales.append((cv2.resize(edges,(tw,th)), cv2.resize(gm,(tw,th))))
        info.append({"name":name, "label":TEMPLATE_LABEL_MAP.get(name,"unknown"), "scales":scales})
    return info

def get_template_scores(img_edges, img_gray, h, w):
    er=[img_edges[h*2//3:,w//2:],img_edges[h*2//3:,:w//2],img_edges[:h//4,:],img_edges[h*3//4:,:]]
    gr=[img_gray[h*2//3:,w//2:],img_gray[h*2//3:,:w//2],img_gray[:h//4,:],img_gray[h*3//4:,:]]
    scores={}
    for ti in TEMPLATE_INFO:
        be,bg=0.0,0.0
        for te,tg in ti["scales"]:
            for r in er:
                if te.shape[0]<=r.shape[0] and te.shape[1]<=r.shape[1]:
                    try: be=max(be,float(cv2.matchTemplate(r,te,cv2.TM_CCOEFF_NORMED).max()))
                    except: pass
            for r in gr:
                if tg.shape[0]<=r.shape[0] and tg.shape[1]<=r.shape[1]:
                    try: bg=max(bg,float(cv2.matchTemplate(r,tg,cv2.TM_CCOEFF_NORMED).max()))
                    except: pass
        scores[ti["name"]+"_edge"]=be; scores[ti["name"]+"_gray"]=bg
    return scores


# ── Hand-crafted features (kept for GBT fallback) ──

def local_contrast_features(g):
    if g.size<100: return [0.,0.,0.]
    gf=g.astype(np.float32); lm=cv2.filter2D(gf,-1,np.ones((7,7),np.float32)/49); d=gf-lm
    return [float((d>15).sum())/d.size, float((d<-15).sum())/d.size, float(d.std())]

def dct_features(g, bs=32):
    if g.shape[0]<bs or g.shape[1]<bs: return [0.,0.,0.]
    cy,cx=g.shape[0]//2,g.shape[1]//2; h=bs//2
    b=g[cy-h:cy+h,cx-h:cx+h].astype(np.float32); d=cv2.dct(b); t=float(np.abs(d).sum())+1e-6
    return [float(np.abs(d[:bs//4,:bs//4]).sum())/t,float(np.abs(d[bs//4:bs//2,bs//4:bs//2]).sum())/t,float(np.abs(d[bs//2:,bs//2:]).sum())/t]

def unsharp_features(g):
    if g.shape[0]<10 or g.shape[1]<10: return [0.,0.]
    bl=cv2.GaussianBlur(g,(5,5),1.5); d=g.astype(float)-bl.astype(float)
    return [float(np.abs(d).mean()),float((np.abs(d)>10).sum()/d.size)]

def region_features(g,e,hsv=None):
    f=[float(g.mean()),float(g.std()),float(np.percentile(g,95)),float(np.percentile(g,5)),
       float(np.percentile(g,95)-np.percentile(g,5)),float(e.mean())/255.0]
    sx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3);sy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    he,ve=float(np.abs(sy).mean()),float(np.abs(sx).mean())
    f.extend([he,ve,he/(ve+1e-6)]); f.extend(local_contrast_features(g)); f.extend(unsharp_features(g))
    hist,_=np.histogram(g.ravel(),bins=8,range=(0,256),density=True); f.extend(hist.tolist())
    if hsv is not None:
        f.extend([float(hsv[:,:,0].mean()),float(hsv[:,:,0].std()),float(hsv[:,:,1].mean()),
                  float(hsv[:,:,1].std()),float(hsv[:,:,2].mean()),float(hsv[:,:,2].std())])
    else: f.extend([0.]*6)
    return f

def extract_features(gray, edges, hsv, h, w, ts):
    feats=[]
    ch,cw=max(h//6,10),max(w//6,10)
    for y1,y2,x1,x2 in [(0,ch,0,cw),(0,ch,w-cw,w),(h-ch,h,0,cw),(h-ch,h,w-cw,w),(0,ch,0,w),(h-ch,h,0,w)]:
        feats.extend(region_features(gray[y1:y2,x1:x2],edges[y1:y2,x1:x2],hsv[y1:y2,x1:x2]))
    fh,fw=max(h//10,8),max(w//10,8)
    feats.extend(region_features(gray[h-fh:,w-fw:],edges[h-fh:,w-fw:],hsv[h-fh:,w-fw:]))
    th2=max(h//12,8)
    feats.extend(region_features(gray[:th2,:],edges[:th2,:],hsv[:th2,:]))
    fh2,fw2=max(h//15,6),max(w//15,6)
    feats.extend(region_features(gray[h-fh2:,w-fw2:],edges[h-fh2:,w-fw2:],hsv[h-fh2:,w-fw2:]))
    feats.extend(dct_features(gray[h*2//3:,w//2:])); feats.extend(dct_features(gray[:h//4,:]))
    center=gray[h//3:2*h//3,w//3:2*w//3]; cm,cs=float(center.mean()),float(center.std())
    feats.extend([float(gray[h-ch:,w-cw:].mean())-cm,float(gray[:ch,:cw].mean())-cm,
                  float(gray[h-ch:,w-cw:].std())-cs,float(gray[:ch,:].mean())-cm])
    feats.extend([float(gray.mean()),float(gray.std()),float(h),float(w),
                  float(h)/float(w) if w>0 else 1.0,float(hsv[:,:,1].mean()),float(hsv[:,:,1].std())])
    feats.extend(local_contrast_features(gray))
    for name in sorted(ts.keys()): feats.append(ts[name])
    return feats

def load_image(image_path, max_dim=768):
    img=cv2.imread(image_path)
    if img is None: img=cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")),cv2.COLOR_RGB2BGR)
    h,w=img.shape[:2]
    if max(h,w)>max_dim: s=max_dim/max(h,w); img=cv2.resize(img,(int(w*s),int(h*s)))
    h,w=img.shape[:2]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(gray,50,150)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return gray,edges,hsv,h,w


# ── Pre-trained CNN (boomb0om) ──

def _map_convnext_key(ck):
    if ck.startswith('downsample_layers.0.0.'): return ck.replace('downsample_layers.0.0.','stem.0.')
    if ck.startswith('downsample_layers.0.1.'): return ck.replace('downsample_layers.0.1.','stem.1.')
    for i in range(1,4):
        if ck.startswith(f'downsample_layers.{i}.'): return ck.replace(f'downsample_layers.{i}.',f'stages.{i}.downsample.')
    if ck.startswith('stages.'):
        parts=ck.split('.')
        if len(parts)>=3 and parts[2].isdigit(): parts.insert(2,'blocks')
        return '.'.join(parts).replace('.dwconv.','.conv_dw.').replace('.pwconv1.','.mlp.fc1.').replace('.pwconv2.','.mlp.fc2.')
    if ck.startswith('norm.'): return ck.replace('norm.','head.norm.')
    if ck.startswith('head.'): return 'head.fc.'+ck[5:]
    return ck

def load_watermark_cnn():
    global WM_CNN, WM_CNN_TRANSFORM, WM_CNN_DEVICE
    from huggingface_hub import hf_hub_download
    WM_CNN_DEVICE=torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    mp=hf_hub_download(repo_id='boomb0om/watermark-detectors',filename='convnext-tiny_watermarks_detector.pth',cache_dir='/tmp/hf_cache')
    m=timm.create_model('convnext_tiny',pretrained=False,num_classes=2)
    m.head.fc=torch.nn.Sequential(torch.nn.Linear(768,512),torch.nn.ReLU(),torch.nn.Linear(512,256),torch.nn.ReLU(),torch.nn.Linear(256,2))
    m.load_state_dict({_map_convnext_key(k):v for k,v in torch.load(mp,map_location='cpu',weights_only=True).items()})
    WM_CNN=m.to(WM_CNN_DEVICE).eval()
    WM_CNN_TRANSFORM=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])

def get_wm_cnn_prob(image_path):
    img=Image.open(image_path).convert("RGB")
    with torch.no_grad(): probs=torch.softmax(WM_CNN(WM_CNN_TRANSFORM(img).unsqueeze(0).to(WM_CNN_DEVICE)),dim=1)[0]
    return float(probs[0].item())


# ── YOLO localization ──

def _patch_nms_if_needed():
    # Some torchvision builds lack CUDA NMS kernels for new GPU archs (e.g. sm_120).
    import torchvision
    try:
        torchvision.ops.nms(torch.zeros((1,4),device=YOLO_DEVICE), torch.zeros(1,device=YOLO_DEVICE), 0.5)
    except Exception:
        _nms = torchvision.ops.nms
        torchvision.ops.nms = lambda b, s, t: _nms(b.cpu(), s.cpu(), t).to(b.device)

def load_yolo():
    global YOLO_MODEL
    from ultralytics import YOLO
    _patch_nms_if_needed()
    yolo_path = Path(__file__).parent / "yolo_watermark.pt"
    YOLO_MODEL = YOLO(str(yolo_path))
    print(f"  YOLO loaded: {yolo_path}", flush=True)

def yolo_detect(image_path):
    """Run YOLO, return (label, confidence, bbox) or None."""
    results = YOLO_MODEL.predict(image_path, verbose=False, device=YOLO_DEVICE, imgsz=1280, conf=0.15)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    # Take highest confidence detection
    best_idx = int(boxes.conf.argmax())
    cls = YOLO_CLASSES[int(boxes.cls[best_idx])]
    conf = float(boxes.conf[best_idx])
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    return {"label": cls, "confidence": conf, "bbox": [int(x1), int(y1), int(x2), int(y2)]}


# ── Verification gate (exp12) ──
# Real watermark crops extracted from train positives via YOLO; narrow-scale
# NCC verification for non-clean predictions that lack YOLO support.

REAL_TEMPLATES = {}   # label -> list of grayscale crops (1024-width normalized)
DALLE_MASTER = None   # BGR master template for color NCC

VERIFY_SCALES = (0.85, 0.92, 1.0, 1.08, 1.15)
VERIFY_ROI = {
    "grok": (0.70, 1.0, 0.60, 1.0),
    "gemini": (0.70, 1.0, 0.60, 1.0),
    "minimax_hailuo": (0.70, 1.0, 0.55, 1.0),
}
VERIFY_TAU = {"grok": 0.30, "minimax_hailuo": 0.22, "gemini": 0.45, "dalle": 0.90}

def extract_real_templates(train_set, per_class=40, keep=3):
    """Crop rendered watermarks from train positives using YOLO detections."""
    global REAL_TEMPLATES
    cand = {}
    for s in train_set:
        lab = s["label"]
        if lab not in VERIFY_ROI and lab != "dalle": continue
        cand.setdefault(lab, [])
        if len(cand[lab]) >= per_class: continue
        cand[lab].append(s["path"])
    for lab, paths in cand.items():
        dets = []
        for p in paths:
            try:
                r = yolo_detect(p)
            except Exception:
                continue
            if not r or r["label"] != lab or r["confidence"] < 0.5: continue
            img = cv2.imread(p)
            if img is None: continue
            x1, y1, x2, y2 = r["bbox"]
            crop = img[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0: continue
            s1024 = 1024.0 / img.shape[1]
            crop = cv2.resize(crop, (max(8,int(crop.shape[1]*s1024)), max(8,int(crop.shape[0]*s1024))))
            dets.append((r["confidence"], cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
        dets.sort(key=lambda d: -d[0])
        REAL_TEMPLATES[lab] = [g for _, g in dets[:keep]]

def verify_score(img_bgr, label):
    """Max narrow-scale NCC of the label's real templates in its expected ROI."""
    if label == "dalle":
        if DALLE_MASTER is None: return 1.0
        h, w = img_bgr.shape[:2]
        img = cv2.resize(img_bgr, (1024, max(8, int(h * 1024.0 / w))))
        roi = img[int(img.shape[0]*0.85):, int(img.shape[1]*0.70):]
        best = 0.0
        for tw in (60, 70, 80, 90, 100):
            th = max(4, int(DALLE_MASTER.shape[0] * tw / DALLE_MASTER.shape[1]))
            if th >= roi.shape[0] or tw >= roi.shape[1]: continue
            t = cv2.resize(DALLE_MASTER, (tw, th))
            try: best = max(best, float(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max()))
            except cv2.error: pass
        return best
    tpls = REAL_TEMPLATES.get(label)
    if not tpls: return 1.0  # nothing to verify against — pass through
    h, w = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (1024, max(8, int(h * 1024.0 / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    yf0, yf1, xf0, xf1 = VERIFY_ROI[label]
    roi_e = cv2.Canny(gray[int(H*yf0):int(H*yf1), int(W*xf0):int(W*xf1)], 80, 160)
    best = 0.0
    for tg in tpls:
        te0 = cv2.Canny(tg, 80, 160)
        for sc in VERIFY_SCALES:
            tw, th = int(te0.shape[1]*sc), int(te0.shape[0]*sc)
            if tw < 8 or th < 8 or th > roi_e.shape[0] or tw > roi_e.shape[1]: continue
            te = cv2.resize(te0, (tw, th))
            try: best = max(best, float(cv2.matchTemplate(roi_e, te, cv2.TM_CCOEFF_NORMED).max()))
            except cv2.error: pass
    return best

def passes_verification(image_path, label):
    tau = VERIFY_TAU.get(label)
    if tau is None: return True  # no gate for this class (e.g. text_tpdne)
    img = cv2.imread(image_path)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return verify_score(img, label) >= tau


# ── Setup & Detect ──

def setup(train_set: list[dict], templates: dict[str, str]):
    global TRAIN_SET, TEMPLATES, MODEL, BINARY_MODEL, TEMPLATE_INFO
    import time as _t; t0=_t.time()
    TRAIN_SET = train_set
    TEMPLATES = templates
    TEMPLATE_INFO = preprocess_templates(templates)

    # Load models
    load_watermark_cnn()
    load_yolo()

    # Real template crops for the verification gate (exp12)
    global DALLE_MASTER
    dp = templates.get("dalle_watermark")
    if dp:
        dm = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if dm is not None:
            DALLE_MASTER = dm[:, :, :3] if dm.ndim == 3 and dm.shape[2] == 4 else dm
    extract_real_templates(train_set)
    print("  real templates:", {k: len(v) for k, v in REAL_TEMPLATES.items()}, flush=True)

    # Train GBT fallback (for when YOLO doesn't detect)
    X, y = [], []
    for i, sample in enumerate(train_set):
        try:
            gray,edges,hsv,h,w = load_image(sample["path"])
            ts = get_template_scores(edges,gray,h,w)
            feats = extract_features(gray,edges,hsv,h,w,ts)
            feats.append(get_wm_cnn_prob(sample["path"]))
            X.append(feats); y.append(sample["label"])
        except: continue
        if (i+1)%300==0: print(f"  GBT features: {i+1}/{len(train_set)} ({_t.time()-t0:.1f}s)", flush=True)

    from collections import Counter
    counts=Counter(y); total=len(y); n_classes=len(counts)
    weight_map={l:(total/(n_classes*c))**1.2 for l,c in counts.items()}
    sw=np.array([weight_map[l] for l in y])
    X_arr=np.array(X)

    y_bin=["clean" if l=="clean" else "watermarked" for l in y]
    bc=Counter(y_bin); bw=np.array([(total/(2*bc[l]))**1.3 for l in y_bin])
    BINARY_MODEL=GradientBoostingClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,random_state=42,subsample=0.8)
    BINARY_MODEL.fit(X_arr,y_bin,sample_weight=bw)
    MODEL=GradientBoostingClassifier(n_estimators=500,max_depth=6,learning_rate=0.08,random_state=42,subsample=0.8,min_samples_leaf=3)
    MODEL.fit(X_arr,y,sample_weight=sw)
    print(f"  setup complete in {_t.time()-t0:.1f}s", flush=True)


def detect(image_path: str) -> dict:
    """Detect and localize watermarks. Returns label + bbox."""
    try:
        # Primary: YOLO localization
        yolo_result = yolo_detect(image_path)

        # Fallback: GBT classifier
        gray,edges,hsv,h,w = load_image(image_path)
        ts = get_template_scores(edges,gray,h,w)
        feats = extract_features(gray,edges,hsv,h,w,ts)
        feats.append(get_wm_cnn_prob(image_path))

        bp = BINARY_MODEL.predict_proba([feats])[0]
        wm_idx = list(BINARY_MODEL.classes_).index("watermarked")
        wm_prob = bp[wm_idx]

        pred = MODEL.predict([feats])[0]
        proba = MODEL.predict_proba([feats])[0]
        confidence = float(proba.max())

        # Binary override
        if pred == "clean" and wm_prob > 0.35:
            classes = MODEL.classes_
            clean_idx = list(classes).index("clean")
            p = proba.copy(); p[clean_idx] = 0
            if p.max() > 0.005:
                pred = classes[int(np.argmax(p))]
                confidence = float(p.max())

        # Fusion: YOLO detection overrides GBT if confident
        if yolo_result and yolo_result["confidence"] > 0.3:
            pred = yolo_result["label"]
            confidence = yolo_result["confidence"]
        elif yolo_result and pred != "clean":
            # Both agree there's a watermark — boost confidence
            confidence = max(confidence, yolo_result["confidence"])

        # Verification gate (exp12): non-clean predictions without YOLO support
        # must show localized template evidence, else revert to clean.
        yolo_backed = yolo_result is not None and yolo_result["confidence"] > 0.3
        if pred != "clean" and not yolo_backed and not passes_verification(image_path, pred):
            pred = "clean"
            confidence = 1.0 - confidence

        binary = "clean" if pred == "clean" else "watermarked"
        result = {"binary": binary, "label": pred, "confidence": confidence}
        if yolo_result:
            result["bbox"] = yolo_result["bbox"]
        return result
    except Exception:
        return {"binary": "clean", "label": "clean", "confidence": 0.0}
