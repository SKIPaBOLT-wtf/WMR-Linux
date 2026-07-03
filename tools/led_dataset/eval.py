#!/usr/bin/env python3
"""Evaluate LED-DETECTION algorithms against the agent-annotated ground truth (independent of the matcher).
GT LED positions per frame = candidates[led_ids].(cx,cy) + missed[].(x,y) from <tag>.gt.json.
A detector is any function frame_gray -> list[(x,y)]. We match detections to GT LEDs (optimal assignment
within `eps` px) and report precision / recall / F1, aggregate and stratified by GT difficulty.

  ~/miniconda3/envs/g2vr/bin/python eval.py dataset/xv1 --frames <frames_dir> --detector local_contrast
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import cv2
from prep import detect_candidates, cluster_and_flag
try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---- detectors (frame_gray -> Nx2 array of (x,y)) ----------------------------------------------
def det_local_contrast(img, cluster_only=True):
    c = detect_candidates(img)
    kept, _ = cluster_and_flag(c)
    pts = [(b["cx"], b["cy"]) for b in kept if (b.get("in_cluster") or not cluster_only)]
    return np.array(pts, float).reshape(-1, 2)

def det_local_contrast_all(img):
    return det_local_contrast(img, cluster_only=False)

DETECTORS = {"local_cluster": det_local_contrast, "local_all": det_local_contrast_all}

# Wire in the CV-research agent's detectors (research/detectors.py, detect(img)->[(x,y,score)])
try:
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent / "research"))
    import detectors as _cvd
    for _name in ("matched_filter", "dog", "blobwatch_py", "blobwatch_v2", "local_contrast", "tophat", "simpleblob"):
        if hasattr(_cvd, _name):
            def _mk(fn):
                return lambda img: np.array([(d.x, d.y) for d in fn(img)], float).reshape(-1, 2)
            DETECTORS[f"cv_{_name}"] = _mk(getattr(_cvd, _name))
except Exception as _e:
    print("(cv detectors unavailable:", _e, ")")

# ---- GT loading ---------------------------------------------------------------------------------
def load_gt(ds_dir: Path):
    gt = {}
    for gf in glob.glob(str(ds_dir / "*.gt.json")):
        g = json.load(open(gf))
        tag = g["tag"]
        cf = ds_dir / f"{tag}.candidates.json"
        if not cf.exists():
            continue
        cands = {c["id"]: c for c in json.load(open(cf))["candidates"]}
        leds = [(cands[i]["cx"], cands[i]["cy"]) for i in g.get("led_ids", []) if i in cands]
        leds += [(m["x"], m["y"]) for m in g.get("missed", [])]
        gt[tag] = dict(led_xy=np.array(leds, float).reshape(-1, 2),
                       degenerate=bool(g.get("degenerate", False)),
                       n_leds=len(leds), confidence=g.get("confidence", "?"),
                       cam=int(tag.split("_")[0][3:]), ts=int(tag.split("_")[1]))
    return gt

def match(gt_xy, det_xy, eps=4.0):
    if len(gt_xy) == 0 and len(det_xy) == 0:
        return 0, 0, 0
    if len(gt_xy) == 0:
        return 0, len(det_xy), 0
    if len(det_xy) == 0:
        return 0, 0, len(gt_xy)
    D = np.hypot(gt_xy[:, None, 0] - det_xy[None, :, 0], gt_xy[:, None, 1] - det_xy[None, :, 1])
    tp = 0
    if _HAVE_SCIPY:
        ri, ci = linear_sum_assignment(D)
        tp = int(sum(D[r, c] <= eps for r, c in zip(ri, ci)))
    else:
        used = set()
        for r in range(len(gt_xy)):
            order = np.argsort(D[r])
            for c in order:
                if c in used:
                    continue
                if D[r, c] <= eps:
                    tp += 1; used.add(c)
                break
    fp = len(det_xy) - tp
    fn = len(gt_xy) - tp
    return tp, fp, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--detector", default="local_cluster", choices=list(DETECTORS))
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--det-json", help="external detections {tag:[[x,y],...]} instead of a built-in")
    args = ap.parse_args()
    ds = Path(args.dataset); frames = Path(args.frames)
    gt = load_gt(ds)
    if not gt:
        print("No GT (*.gt.json) found yet in", ds); return
    from prep import frame_index
    idx = frame_index(frames)
    ext = json.load(open(args.det_json)) if args.det_json else None
    det_fn = DETECTORS[args.detector]
    strata = {"clean": [0, 0, 0], "degenerate": [0, 0, 0], "sparse(<=4)": [0, 0, 0], "rich(>=9)": [0, 0, 0]}
    tot = [0, 0, 0]
    for tag, g in sorted(gt.items()):
        if ext is not None:
            det = np.array(ext.get(tag, []), float).reshape(-1, 2)
        else:
            arr = idx.get(g["cam"], []); ts = g["ts"]
            fp = min(arr, key=lambda a: abs(a[0] - ts))[1] if arr else None
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE) if fp else None
            det = det_fn(img) if img is not None else np.zeros((0, 2))
        tp, fpp, fn = match(g["led_xy"], det, args.eps)
        tot[0] += tp; tot[1] += fpp; tot[2] += fn
        key = "degenerate" if g["degenerate"] else ("sparse(<=4)" if g["n_leds"] <= 4 else ("rich(>=9)" if g["n_leds"] >= 9 else "clean"))
        for kk in ([key] if g["degenerate"] else [key, "clean"] if key != "clean" else [key]):
            if kk in strata:
                strata[kk][0] += tp; strata[kk][1] += fpp; strata[kk][2] += fn
    def pr(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        return p, r, f
    print(f"=== detector '{args.detector if not ext else args.det_json}' vs GT ({len(gt)} frames, eps={args.eps}px) ===")
    p, r, f = pr(*tot)
    print(f"OVERALL  P={p:.3f} R={r:.3f} F1={f:.3f}  (TP={tot[0]} FP={tot[1]} FN={tot[2]})")
    for k, v in strata.items():
        if sum(v):
            p, r, f = pr(*v); print(f"  {k:14} P={p:.3f} R={r:.3f} F1={f:.3f}  (TP={v[0]} FP={v[1]} FN={v[2]})")

if __name__ == "__main__":
    main()
