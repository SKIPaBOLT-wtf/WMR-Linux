#!/usr/bin/env python3
"""Select a stratified set of controller frames for GT annotation — DETECTOR-based, never the matcher.
Sweep every frame, run the local-contrast candidate detector, score difficulty features
(candidate count, brightness, cluster tightness), and sample evenly across difficulty bins so the
HARD cases (few/dim LEDs, scattered clutter) are well represented, not just the easy ones.

  ~/miniconda3/envs/g2vr/bin/python select.py <capture_dir> <n_per_cam> [--stride 1] > frames.txt
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path
import numpy as np
import cv2
from prep import detect_candidates, frame_index

def features(img):
    c = detect_candidates(img)
    real = [b for b in c if not b["big"]]  # ignore huge regions (walls/monitor) for the cluster stat
    n = len(real)
    if n == 0:
        return dict(n=0, peak=0, tight=999.0, contrast=0.0)
    xs = np.array([b["cx"] for b in real]); ys = np.array([b["cy"] for b in real])
    tight = float(np.hypot(xs.std(), ys.std()))  # small => clustered (a controller), large => scattered
    return dict(n=n, peak=max(b["peak"] for b in real),
                tight=round(tight, 1), contrast=round(max(b["contrast"] for b in real), 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("n_total", type=int, help="approx total frames to select for this capture")
    ap.add_argument("--stride", type=int, default=2, help="scan every Nth frame (speed)")
    ap.add_argument("--cams", type=int, default=4)
    args = ap.parse_args()
    frames = Path(args.capture) / "frames"
    idx = frame_index(frames)
    rows = []
    for cam in range(args.cams):
        arr = idx.get(cam, [])
        for ts, fp in arr[::args.stride]:
            img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            f = features(img)
            if f["n"] >= 2 and f["tight"] < 120:  # controller plausibly in this cam's view
                rows.append((cam, ts, f["n"], f["peak"], f["tight"], f["contrast"]))
        print(f"  cam{cam}: scanned {len(arr[::args.stride])} -> {sum(1 for r in rows if r[0]==cam)} in-view", file=sys.stderr)
    # difficulty bins by candidate count: hard(2-3), med(4-6), easy(7-10), rich(11+)
    def dbin(n): return 0 if n <= 3 else 1 if n <= 6 else 2 if n <= 10 else 3
    by = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        by[dbin(r[2])].append(r)
    per = max(1, args.n_total // 4)
    sel = []
    for b, lst in by.items():
        lst.sort(key=lambda r: r[1])  # by ts (temporal spread)
        if not lst:
            continue
        step = max(1, len(lst) // per)
        sel += lst[::step][:per]
    sel.sort(key=lambda r: (r[0], r[1]))
    cap = Path(args.capture).name
    print(f"# {cap}: selected {len(sel)} frames (bins hard/med/easy/rich = "
          f"{sum(dbin(r[2])==0 for r in sel)}/{sum(dbin(r[2])==1 for r in sel)}/"
          f"{sum(dbin(r[2])==2 for r in sel)}/{sum(dbin(r[2])==3 for r in sel)})", file=sys.stderr)
    for cam, ts, n, peak, tight, contrast in sel:
        print(f"{cam} {ts}  # n={n} peak={peak} tight={tight} contrast={contrast}")

if __name__ == "__main__":
    main()
