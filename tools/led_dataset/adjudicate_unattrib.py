#!/usr/bin/env python3
"""Adjudicate UNATTRIB matcher-failure frames at the POSITION level.

UNATTRIB = a device committed near the GT frame-time but its pose does not LED-explain this
view (or projects <4 visible model LEDs here). LED-level matching across a wide camera
baseline is fragile even for a correct pose, so adjudicate by geometry that survives the
transform: project EVERY committed device's pose center into the GT camera and measure the
distance to the annotated LED-cluster centroid.

  TRACKED_AT_GT      some committed device's center lands on the annotated cluster
                     (within max(--radius, 1.5x cluster radius)) -> GOOD-equivalent;
                     the tracker held this controller, only the LED-level credit failed.
  TRACKED_ELSEWHERE  every committed pose projects far away / behind the camera ->
                     the controller the annotator saw is NOT held by any commit
                     (missed-equivalent; or only the partner was committed).

  ~/miniconda3/envs/g2vr/bin/python adjudicate_unattrib.py --root results/<run>/handgt \
      --json /tmp/curfix_catalog2.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "research"))
import g2cam
sys.path.insert(0, str(Path(__file__).parent / "../telemetry"))
from matcher_failure import CAMS_JSON
from manifest import Manifest
import g2_geom as G
from dump_frames import xform
from matcher_failure import CTRL, MATCH_NS, load_gt, nearest_idx, split_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True,
                    help="replay battery root containing xv1/, clean2/, and headpose/")
    ap.add_argument("--json", required=True,
                    help="classified-frame catalog from `matcher_failure.py --telemetry-root ROOT --json PATH`")
    ap.add_argument("--radius", type=float, default=40.0, help="base accept radius, px")
    ap.add_argument("--out", default=None, help="write per-frame verdicts here")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.json)) if r["verdict"] == "UNATTRIB"]
    cams = g2cam.load_cams(CAMS_JSON)
    models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}

    splits = split_paths(args.root)
    sels = {}
    for split, cap in splits.items():
        tel = Path(cap) / "telemetry"
        m = Manifest.load(tel)
        cand = G.load_stream(tel, m, "candidate")
        sel = cand[cand["selected"] == 1]
        sels[split] = {d: sel[sel["device_id"] == d] for d in (1, 2)}

    gts = {split: load_gt(Path("dataset/pool") / split) for split in splits}

    out_rows = []
    counts = {"TRACKED_AT_GT": 0, "TRACKED_ELSEWHERE": 0, "NO_NEAR_COMMIT": 0}
    for r in rows:
        split, tag = r["split"], r["tag"]
        g = gts[split][tag]
        cam_id, ts = g["cam"], g["ts"]
        centroid = g["led_xy"].mean(axis=0)
        cluster_r = float(np.hypot(*(g["led_xy"] - centroid).T).max()) if len(g["led_xy"]) > 1 else 0.0
        accept_r = max(args.radius, 1.5 * cluster_r)

        best = None  # (dist, dev, commit_cam, n_vis)
        for d in (1, 2):
            s = sels[split][d]
            if not len(s):
                continue
            sts = np.sort(s["t_mono_ns"].astype(np.int64))
            order = np.argsort(s["t_mono_ns"].astype(np.int64))
            s_sorted = s[order]
            k = nearest_idx(sts, ts, MATCH_NS)
            if k < 0:
                continue
            row = s_sorted[k]
            R = g2cam._quat_to_R(np.array([row["qx"], row["qy"], row["qz"], row["qw"]], float))
            t = np.array([row["px"], row["py"], row["pz"]], float)
            commit_cam = int(row["cam_id"])
            if commit_cam != cam_id:
                R, t = xform(R, t, cams[commit_cam], cams[cam_id])
            if t[2] <= 0.0:
                continue  # behind the GT camera
            pm = g2cam.project_model(cams[cam_id], R, t, models[d])
            uv = pm["uv"]
            if not len(uv):
                continue
            center = uv.mean(axis=0)
            dist = float(np.hypot(*(center - centroid)))
            n_vis = int(pm["visible"].sum())
            if best is None or dist < best[0]:
                best = (dist, d, commit_cam, n_vis)

        if best is None:
            verdict = "NO_NEAR_COMMIT"
            dist, dev, ccam, n_vis = None, None, None, None
        else:
            dist, dev, ccam, n_vis = best
            verdict = "TRACKED_AT_GT" if dist <= accept_r else "TRACKED_ELSEWHERE"
        counts[verdict] += 1
        out_rows.append(dict(split=split, tag=tag, verdict=verdict, dist_px=dist,
                             accept_r=round(accept_r, 1), dev=dev, commit_cam=ccam,
                             n_proj_vis=n_vis, n_true=g["n_true"], n_ctrl=g["n_ctrl"]))

    n = len(out_rows)
    print(f"UNATTRIB adjudication: {n} frames")
    for k, v in counts.items():
        print(f"  {k:18s} {v:4d}  ({100.0*v/max(n,1):.0f}%)")
    far = [o for o in out_rows if o["verdict"] == "TRACKED_ELSEWHERE"]
    far.sort(key=lambda o: -(o["dist_px"] or 0))
    if far:
        print("\nTRACKED_ELSEWHERE frames (controller seen by annotator, no commit on it):")
        for o in far:
            print(f"  {o['split']}/{o['tag']} dev={o['dev']} dist={o['dist_px']:.0f}px "
                  f"n_true={o['n_true']} n_ctrl={o['n_ctrl']} n_proj_vis={o['n_proj_vis']}")
    if args.out:
        json.dump(out_rows, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
