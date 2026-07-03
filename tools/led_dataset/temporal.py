#!/usr/bin/env python3
"""Temporal-context montage for LED GT annotation. Renders a target frame plus its temporal
neighbours (same cam) side-by-side, contrast-stretched, with a SHARED fixed zoom window centred on a
chosen point. Lets the annotator judge whether a bright cluster MOVES smoothly frame-to-frame (a real
controller LED arc) or stays STATIC/continuous (scene clutter: bars, blind-slats, reflections).
Independent of any tracker output.

  ~/miniconda3/envs/g2vr/bin/python temporal.py <frames_dir> --cam 3 --ts 150699744155900 \
      --cx 615 --cy 85 --half 80 --n 4 --out /tmp/t.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import cv2
from prep import frame_index, nearest, stretch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--ts", type=int, required=True)
    ap.add_argument("--cx", type=float, help="zoom-window centre x (default: full frame)")
    ap.add_argument("--cy", type=float, help="zoom-window centre y")
    ap.add_argument("--half", type=int, default=90, help="half-size of square zoom window (px)")
    ap.add_argument("--n", type=int, default=3, help="neighbours each side")
    ap.add_argument("--zoom", type=int, default=4)
    ap.add_argument("--out", default="/tmp/temporal.png")
    args = ap.parse_args()
    frames_dir = Path(args.frames_dir)
    idx = frame_index(frames_dir)
    arr = idx.get(args.cam, [])
    if not arr:
        print(f"no cam{args.cam} frames"); return
    ts_arr = np.array([t for t, _ in arr])
    j = int(np.argmin(np.abs(ts_arr - args.ts)))
    lo, hi = max(0, j - args.n), min(len(arr), j + args.n + 1)
    panels = []
    for k in range(lo, hi):
        ts, fp = arr[k]
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if args.cx is not None and args.cy is not None:
            x0 = max(0, int(args.cx - args.half)); x1 = min(img.shape[1], int(args.cx + args.half))
            y0 = max(0, int(args.cy - args.half)); y1 = min(img.shape[0], int(args.cy + args.half))
            crop = img[y0:y1, x0:x1]
        else:
            crop = img; x0 = y0 = 0
        vis = cv2.cvtColor(stretch(crop), cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
        rel = ts - args.ts
        col = (0, 255, 255) if k == j else (180, 180, 180)
        tag = f"{'>>TARGET<<' if k == j else f'{rel/1e6:+.1f}ms'}  peak={int(crop.max())}"
        cv2.putText(vis, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        if k == j:
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 255), 2)
        panels.append(vis)
    if not panels:
        print("no panels"); return
    h = max(p.shape[0] for p in panels)
    row = np.zeros((h, sum(p.shape[1] for p in panels) + 4 * len(panels), 3), np.uint8)
    x = 0
    for p in panels:
        row[:p.shape[0], x:x + p.shape[1]] = p
        x += p.shape[1] + 4
    cv2.imwrite(args.out, row)
    print(f"wrote {args.out}  ({len(panels)} panels, centre frame peak={int(arr[j][1] if False else 0)})")


if __name__ == "__main__":
    main()
