#!/usr/bin/env python3
"""Score the REAL C blobwatch detector (via the `blobdump` harness) against the annotated LED GT.

Maps each GT tag (cam<c>_<ts>) to its nearest-timestamp source PGM, runs blobdump to dump the real C
blob centroids per frame, then reuses eval.py's matching + stratified P/R. This is the authoritative,
decoupled detection eval (the C detector itself, not a Python proxy, not the matcher).

  python run_blobdump_eval.py <dataset_dir> --frames <pgm_dir> --blobdump <path> [--pix 8]
"""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

# import the matcher + GT loader from the (already-importable) eval.py
sys.path.insert(0, str(Path(__file__).parent))
from prep import frame_index  # noqa: E402
import numpy as np  # noqa: E402
import eval as ev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--blobdump", required=True)
    ap.add_argument("--pix", type=int, default=8)
    ap.add_argument("--eps", type=float, default=4.0)
    ap.add_argument("--out-json", default="/tmp/blobdump_det.json")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    ds = Path(args.dataset)
    idx = frame_index(Path(args.frames))
    gt = ev.load_gt(ds)
    if not gt:
        print("No GT in", ds); return

    # map each GT tag -> nearest source PGM path
    tag_to_pgm = {}
    for tag, g in gt.items():
        arr = idx.get(g["cam"], []); ts = g["ts"]
        if not arr:
            continue
        ts_arr = np.array([a[0] for a in arr])
        j = int(np.argmin(np.abs(ts_arr - ts)))
        tag_to_pgm[tag] = arr[j][1]
    if not tag_to_pgm:
        print("No source PGMs matched the GT tags under", args.frames); return

    # blobdump keys output by the PGM filename stem; build stem->gt_tag so we can remap
    stem_to_tag = {Path(p).stem: t for t, p in tag_to_pgm.items()}
    pgms = sorted(set(tag_to_pgm.values()))

    # run blobdump on the mapped PGMs (chunk argv to stay under ARG_MAX)
    raw = args.out_json
    merged = {}
    CHUNK = 800
    for i in range(0, len(pgms), CHUNK):
        chunk = pgms[i:i + CHUNK]
        part = f"{raw}.part{i}"
        cmd = [args.blobdump, str(args.pix), part] + chunk
        subprocess.run(cmd, check=True)
        merged.update(json.load(open(part)))
    # remap stem -> gt tag, take only (x,y)
    det = {}
    for stem, blobs in merged.items():
        t = stem_to_tag.get(stem)
        if t is None:
            continue
        det[t] = [[b[0], b[1]] for b in blobs]
    json.dump(det, open(raw, "w"))

    # score via eval.match, stratified the same way eval.py does
    strata = {"clean_all": [0, 0, 0], "clean": [0, 0, 0], "degenerate": [0, 0, 0],
              "sparse(<=4)": [0, 0, 0], "rich(>=9)": [0, 0, 0]}
    tot = [0, 0, 0]
    for tag, g in sorted(gt.items()):
        d = np.array(det.get(tag, []), float).reshape(-1, 2)
        tp, fpp, fn = ev.match(g["led_xy"], d, args.eps)
        tot[0] += tp; tot[1] += fpp; tot[2] += fn
        key = ("degenerate" if g["degenerate"] else
               ("sparse(<=4)" if g["n_leds"] <= 4 else ("rich(>=9)" if g["n_leds"] >= 9 else "clean")))
        strata[key][0] += tp; strata[key][1] += fpp; strata[key][2] += fn
        if not g["degenerate"]:
            strata["clean_all"][0] += tp; strata["clean_all"][1] += fpp; strata["clean_all"][2] += fn

    def pr(tp, fp, fn):
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        return p, r, (2 * p * r / (p + r) if p + r else 0)
    label = args.tag or f"C blobwatch pix={args.pix}"
    p, r, f = pr(*tot)
    print(f"=== {label} vs GT ({len(gt)} frames, eps={args.eps}px) ===")
    print(f"OVERALL    P={p:.3f} R={r:.3f} F1={f:.3f}  (TP={tot[0]} FP={tot[1]} FN={tot[2]})")
    for k in ("clean_all", "clean", "sparse(<=4)", "rich(>=9)", "degenerate"):
        v = strata[k]
        if sum(v):
            p, r, f = pr(*v)
            print(f"  {k:12} P={p:.3f} R={r:.3f} F1={f:.3f}  (TP={v[0]} FP={v[1]} FN={v[2]})")


if __name__ == "__main__":
    main()
