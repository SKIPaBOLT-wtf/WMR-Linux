#!/usr/bin/env python3
"""LED-annotation dataset prep. Over-detect candidate bright spots per controller frame (precise
intensity-weighted centroids, local-contrast gate so dim real LEDs aren't missed) and render a
contrast-stretched, numbered image for an annotator to CLASSIFY each candidate as LED-on-controller
vs clutter. Independent of the matcher — the GT is what a careful viewer marks, not what the tracker says.

Run with the g2vr conda python (has cv2):
  ~/miniconda3/envs/g2vr/bin/python prep.py <frames_dir> <out_dir> --cam 2 --ts 150775303680817
  ~/miniconda3/envs/g2vr/bin/python prep.py <frames_dir> <out_dir> --list frames.txt
"""
from __future__ import annotations
import argparse, json, glob, re
from pathlib import Path
import numpy as np
import cv2

# ---- candidate detection (local-contrast, over-detect) -----------------------------------------
def detect_candidates(img: np.ndarray, abs_floor: int = 10, contrast: float = 5.0,
                      bg_sigma: float = 21.0, max_area: int = 600):
    """Return candidate bright spots. A candidate pixel must exceed BOTH an absolute floor and its
    local background by `contrast` (mirrors blobwatch's adaptive gate, but deliberately permissive so
    dim real LEDs survive for the annotator to judge). Centroid = intensity-weighted."""
    f = img.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), bg_sigma)
    mask = ((f > abs_floor) & (f - bg > contrast)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        comp = (lab == i)
        vals = f[comp]
        peak = float(vals.max())
        ys, xs = np.nonzero(comp)
        wsum = vals.sum()
        cx = float((xs * f[comp]).sum() / wsum); cy = float((ys * f[comp]).sum() / wsum)
        local_bg = float(bg[int(round(cy)), int(round(cx))])
        aspect = max(w, h) / max(1, min(w, h))
        fill = area / max(1, w * h)
        out.append(dict(cx=round(cx, 2), cy=round(cy, 2), peak=int(peak), area=area, w=w, h=h,
                        aspect=round(aspect, 2), fill=round(fill, 2),
                        contrast=round(peak - local_bg, 1), big=bool(area > max_area)))
    out.sort(key=lambda c: -c["contrast"])
    for i, c in enumerate(out):
        c["id"] = i
    return out

def cluster_and_flag(cands, cap=40, radius=45.0, min_density=2):
    """Cap to the brightest `cap` candidates (the dim tail is rarely an LED), then mark each by local
    DENSITY (# neighbours within `radius`). The controller's LEDs form a dense cluster; clutter is
    isolated. Returns (rendered_candidates, frame_flags). Flags 'degenerate' frames (motion-blur /
    clutter-storm: too many raw candidates or no dense cluster) so annotation can treat them honestly."""
    n_raw = len(cands)
    small = [c for c in cands if not c["big"]]
    kept = small[:cap]
    import numpy as _np
    if kept:
        P = _np.array([[c["cx"], c["cy"]] for c in kept])
        for i, c in enumerate(kept):
            d = _np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
            c["density"] = int((d < radius).sum() - 1)
            c["in_cluster"] = c["density"] >= min_density
    n_cluster = sum(c.get("in_cluster") for c in kept)
    degenerate = n_raw > 70 or n_cluster == 0
    flags = dict(n_raw=n_raw, n_rendered=len(kept), n_cluster=n_cluster, degenerate=bool(degenerate))
    return kept, flags

def stretch(img: np.ndarray, hi_pct: float = 99.9) -> np.ndarray:
    hi = max(24.0, np.percentile(img, hi_pct))
    return np.clip(img.astype(np.float32) / hi * 255.0, 0, 255).astype(np.uint8)

def _draw_markers(vis, cands, scale, ox=0, oy=0):
    for c in cands:
        x, y = int(c["cx"] * scale) - ox, int(c["cy"] * scale) - oy
        if x < 0 or y < 0 or x >= vis.shape[1] or y >= vis.shape[0]:
            continue
        col = (0, 220, 0) if c.get("in_cluster") else (0, 165, 255)  # green=in dense cluster, orange=isolated
        cv2.circle(vis, (x, y), max(7, int(0.7 * max(c["w"], c["h"]) * scale)), col, 1)
        cv2.putText(vis, str(c["id"]), (x + 7, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

def render(img: np.ndarray, cands, out_path: Path, title: str, scale: int = 2, overlay=None):
    """Combined image: full-frame context (left) + zoom on the candidate cluster (right) so the
    annotator sees both where the controller is and each LED clearly."""
    base = cv2.cvtColor(stretch(img), cv2.COLOR_GRAY2BGR)
    full = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    _draw_markers(full, cands, scale)
    if overlay is not None:
        for (ux, uy) in overlay:
            cv2.drawMarker(full, (int(ux * scale), int(uy * scale)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(full, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    # zoom on the candidate cluster bbox (padded), upscaled to ~full height
    if cands:
        xs = [c["cx"] for c in cands]; ys = [c["cy"] for c in cands]
        x0, x1 = max(0, int(min(xs)) - 30), min(img.shape[1], int(max(xs)) + 30)
        y0, y1 = max(0, int(min(ys)) - 30), min(img.shape[0], int(max(ys)) + 30)
        crop = base[y0:y1, x0:x1]
        if crop.size:
            zs = max(2, int(full.shape[0] / max(1, crop.shape[0])))
            zs = min(zs, 14)
            zoom = cv2.resize(crop, None, fx=zs, fy=zs, interpolation=cv2.INTER_NEAREST)
            _draw_markers(zoom, cands, zs, ox=x0 * zs, oy=y0 * zs)
            cv2.putText(zoom, f"ZOOM x{zs} of [{x0},{y0}]-[{x1},{y1}]", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            h = max(full.shape[0], zoom.shape[0])
            canvas = np.zeros((h, full.shape[1] + zoom.shape[1] + 8, 3), np.uint8)
            canvas[:full.shape[0], :full.shape[1]] = full
            canvas[:zoom.shape[0], full.shape[1] + 8:full.shape[1] + 8 + zoom.shape[1]] = zoom
            full = canvas
    cv2.imwrite(str(out_path), full)

# ---- frame lookup -------------------------------------------------------------------------------
def frame_index(frames_dir: Path):
    idx = {}
    for p in glob.glob(str(frames_dir / "cam*_*.pgm")):
        m = re.match(r"cam(\d+)_t0*(\d+)_", Path(p).name)
        if m:
            idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
    for c in idx:
        idx[c].sort()
    return idx

def nearest(idx, cam, ts):
    arr = idx.get(cam, [])
    if not arr:
        return None
    ts_arr = np.array([t for t, _ in arr])
    j = int(np.argmin(np.abs(ts_arr - ts)))
    return arr[j]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--cam", type=int)
    ap.add_argument("--ts", type=int)
    ap.add_argument("--list", help="file of 'cam ts' lines")
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()
    frames_dir = Path(args.frames_dir); out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    idx = frame_index(frames_dir)
    jobs = []
    if args.list:
        for ln in open(args.list):
            ln = ln.split("#")[0].split()
            if len(ln) >= 2:
                jobs.append((int(ln[0]), int(ln[1])))
    elif args.cam is not None and args.ts is not None:
        jobs.append((args.cam, args.ts))
    manifest = []
    for cam, ts in jobs:
        hit = nearest(idx, cam, ts)
        if not hit:
            print(f"cam{cam} ts{ts}: no frame"); continue
        fts, fp = hit
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        cands = detect_candidates(img)
        kept, flags = cluster_and_flag(cands)
        tag = f"cam{cam}_{fts}"
        ttl = f"{tag}  rendered {flags['n_rendered']}/{flags['n_raw']} cand, cluster={flags['n_cluster']}" + \
              ("  [DEGENERATE/blur]" if flags["degenerate"] else "")
        render(img, kept, out / f"{tag}.png", ttl, args.scale)
        json.dump({"frame": Path(fp).name, "cam": cam, "ts": fts, "flags": flags, "candidates": kept},
                  open(out / f"{tag}.candidates.json", "w"), indent=1)
        manifest.append(tag)
        print(f"{tag}: {flags['n_rendered']}/{flags['n_raw']} cand, cluster={flags['n_cluster']}"
              + (" DEGEN" if flags['degenerate'] else ""))
    json.dump(manifest, open(out / "manifest.json", "w"), indent=1)

if __name__ == "__main__":
    main()
