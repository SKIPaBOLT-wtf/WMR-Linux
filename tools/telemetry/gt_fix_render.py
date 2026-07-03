#!/usr/bin/env python3
"""gt_fix_render.py -- visual review panels for the GT blob-fix (Step 2 of GT-FIX-V2).

For a stratified sample of cleaned-GT frames (CORRUPT / GOOD / ABSTAIN), render a per-co-visible-
camera panel of the short-exposure LED frame with:
  cyan o   = DETECTED LED blob (the raw observation, pose-independent)
  green X  = cleaned-GT pose projected into this camera (the OLD reference)
  red +    = the INDEPENDENT blob-refined corrected pose (CORRUPT frames only; what the blobs chose)
  magenta . = the fixed-witness candidate pose projected (context)
plus the paired long-exposure frame (the controller body is visible to the eye). The title carries
the verdict + the corrected-vs-GT delta + per-pose blob-explain stats so a reviewer can SEE whether
the cleaned-GT (green) floats off the blobs while the corrected pose (red) sits on them (= true
GT corruption) or whether both miss / agree (= genuine sparse-depth ambiguity, left ABSTAIN).

Run with the BASE conda (cv2): ~/miniconda3/bin/python gt_fix_render.py <capture> --dev 1 \
    --witness /tmp/wtfix_<cap>/out --out docs/sota-research/gt_fix_v2_renders
"""
from __future__ import annotations
import argparse
import glob
import re
from pathlib import Path

import numpy as np

import detection_f1 as DF
import blob_explain as BE
import gt_blob_fix as GF
from headpose_anchor import load_head_pose
from smooth_ref import build_reference
from manifest import DEVICE_NAMES


def _long_index(capdir: Path):
    """cam -> sorted [(ts, path)] of long-exposure (e300) frames, for the human-visible panel."""
    idx = {}
    for p in glob.glob(str(capdir / "frames" / "cam*_e300_*.pgm")) + \
             glob.glob(str(capdir / "frames-session2" / "cam*_e300_*.pgm")):
        m = re.match(r"cam(\d+)_t0*(\d+)_", Path(p).name)
        if m:
            idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
    for c in idx:
        idx[c].sort()
    return idx


def _nearest(arr, ts):
    if not arr:
        return None
    return min(arr, key=lambda a: abs(a[0] - ts))


def _stretch(img):
    hi = max(24.0, np.percentile(img, 99.9))
    return np.clip(img.astype(np.float32) / hi * 255.0, 0, 255).astype(np.uint8)


def _draw_pose_panel(cv2, img, cam, g2cam, g2cams, mdl, blobs, q_gt, p_gt, q_fix, p_fix,
                     q_wit, p_wit, Rh, hpos, scale=2):
    vis = cv2.resize(cv2.cvtColor(_stretch(img), cv2.COLOR_GRAY2BGR), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    for (x, y) in blobs:
        cv2.circle(vis, (int(x * scale), int(y * scale)), 7, (0, 200, 200), 1)

    def project(q, p):
        Rcm, tcm = BE._world_to_cam(DF.quat_to_R(q), p, Rh, hpos, cam)
        pm = g2cam.project_model(g2cams[cam.id], Rcm, tcm, mdl)
        return pm["uv"][pm["visible"]]

    if q_wit is not None:
        for (x, y) in project(q_wit, p_wit):
            cv2.circle(vis, (int(x * scale), int(y * scale)), 2, (200, 0, 200), -1)
    for (x, y) in project(q_gt, p_gt):
        cv2.drawMarker(vis, (int(x * scale), int(y * scale)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 13, 2)
    if q_fix is not None:
        for (x, y) in project(q_fix, p_fix):
            cv2.drawMarker(vis, (int(x * scale), int(y * scale)), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--dev", type=int, choices=(1, 2), required=True)
    ap.add_argument("--witness", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/sota-research/gt_fix_v2_renders"))
    ap.add_argument("--per-class", type=int, default=10, help="frames per verdict class")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import cv2

    capdir = args.capture
    cr = GF.load_corrupt_mask(capdir, args.dev)
    if cr is None:
        raise SystemExit(f"no gt_blobfix cache for {capdir} dev{args.dev} -- run gt_blob_fix.py first")
    ref = build_reference(capdir / "telemetry", args.dev)
    hp = load_head_pose(capdir / "telemetry")
    g2cam, _ = BE._lazy_imports()
    cams = DF.load_cameras(g2cam.HMD_CAMERAS.as_posix())
    g2cams = g2cam.load_cams()
    mdl = g2cam.load_led_model(g2cam.CTRL_LEFT if args.dev == 1 else g2cam.CTRL_RIGHT)
    cache = BE.BlobCache(GF._frames_dir(capdir),
                         disk_cache=Path("/tmp/gtfix") / f"blobcache_{capdir.name}.npz")
    longi = _long_index(capdir)
    wit_t = wit_p = wit_q = wit_v = None
    if args.witness:
        wit_t, wit_p, wit_q, wit_v = GF._load_witness_csv(args.witness / f"dev{args.dev}.csv")

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    classes = {"CORRUPT": GF.V_CORRUPT, "GOOD": GF.V_GOOD, "ABSTAIN": GF.V_ABSTAIN}
    manifest = []
    for cname, cval in classes.items():
        idx = np.where(cr.verdict == cval)[0]
        if idx.size == 0:
            continue
        pick = rng.choice(idx, min(args.per_class, idx.size), replace=False)
        for i in sorted(pick):
            ts = int(cr.t_ns[i])
            hpos, hq = GF._head_interp(hp, ts)
            if hpos is None:
                continue
            Rh = DF.quat_to_R(hq)
            q_gt, p_gt = ref.quat[i], ref.pos[i]
            q_fix = cr.fix_quat[i] if np.isfinite(cr.fix_quat[i, 0]) else None
            p_fix = cr.fix_pos[i] if q_fix is not None else None
            q_wit, p_wit = GF._witness_seed(wit_t, wit_p, wit_q, wit_v, ts)
            # per co-visible camera short-exp panels
            panels = []
            for cid, cam in enumerate(cams):
                Rcm, tcm = BE._world_to_cam(DF.quat_to_R(q_gt), p_gt, Rh, hpos, cam)
                pm = g2cam.project_model(g2cams[cid], Rcm, tcm, mdl)
                covis_fix = False
                if q_fix is not None:
                    Rf, tf = BE._world_to_cam(DF.quat_to_R(q_fix), p_fix, Rh, hpos, cam)
                    covis_fix = int(g2cam.project_model(g2cams[cid], Rf, tf, mdl)["visible"].sum()) >= BE.MIN_VIS_LEDS
                if int(pm["visible"].sum()) < BE.MIN_VIS_LEDS and not covis_fix:
                    continue
                path, _dt = cache.nearest_frame(cid, ts)
                if path is None:
                    continue
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                blobs, _c = cache.blobs(path)
                pan = _draw_pose_panel(cv2, img, cam, g2cam, g2cams, mdl, blobs,
                                       q_gt, p_gt, q_fix, p_fix, q_wit, p_wit, Rh, hpos)
                cv2.putText(pan, f"cam{cid} blobs={len(blobs)}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                panels.append(pan)
            if not panels:
                continue
            h = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
            row = np.hstack([np.hstack([p, np.full((h, 5, 3), 60, np.uint8)]) for p in panels])
            # long-exp panel (human-visible controller)
            lh = _nearest(longi.get(0, []) or sum(longi.values(), []), ts) if longi else None
            dq = cr.fix_dq_deg[i]
            dp = cr.fix_dp_cm[i]
            hdr = (f"{capdir.name} dev{args.dev} {cname} ts={ts} | gt_reproj="
                   f"{cr.gt_reproj_px[i]:.1f}px fix_reproj={cr.fix_reproj_px[i]:.1f}px "
                   f"dq={dq:.0f}deg dp={dp:.1f}cm anchor_flag={bool(cr.anchor_flag[i])} "
                   f"covis={cr.n_covis[i]}")
            banner = np.full((44, row.shape[1], 3), 30, np.uint8)
            cv2.putText(banner, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(banner, "cyan=blob  greenX=cleaned-GT  red+=blob-refined-FIX  magenta=witness",
                        (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
            full = np.vstack([banner, row])
            name = f"{cname}_{capdir.name}_dev{args.dev}_{ts}.png"
            cv2.imwrite(str(out / name), full)
            manifest.append(dict(file=name, cls=cname, ts=ts, dq_deg=float(dq), dp_cm=float(dp),
                                 gt_reproj=float(cr.gt_reproj_px[i]), fix_reproj=float(cr.fix_reproj_px[i]),
                                 anchor_flag=bool(cr.anchor_flag[i]), n_covis=int(cr.n_covis[i])))
    import json
    mpath = out / f"manifest_{capdir.name}_dev{args.dev}.json"
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"rendered {len(manifest)} panels -> {out}  (manifest {mpath.name})")


if __name__ == "__main__":
    main()
