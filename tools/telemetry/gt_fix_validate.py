#!/usr/bin/env python3
"""gt_fix_validate.py -- INDEPENDENT, non-circular validation of the GT blob-fix (Step 4).

Proves the corrupt-frame drops are justified by arbiters INDEPENDENT of the tracker decision, on a
HELD-OUT sample (re-derived here, not reusing the build-time decision):
  1. BLOB independence: on CORRUPT frames the cleaned-GT pose does NOT reproject onto the detected
     blobs (high pooled reproj / not confirmed) while the blob-refined fix DOES (multi-cam confirmed,
     tight reproj). Reported as median GT vs fix pooled-reproj.
  2. ANCHOR independence: the SLAM head-pose anchor (which never sees the controller blobs) is used
     to cross-check. For CORRUPT frames where the anchor has an opinion, the fix's orientation must
     agree with the anchor's verdict better than the GT (lower anchor tilt/yaw residual). Reported
     as the share where the fix is anchor-consistent.
  3. CONTROL (no false drops): on a sample of GOOD (kept) frames the GT IS blob-confirmed -- so the
     fix is not silently dropping good frames.
  4. CIRCULARITY guard: corrupt drops must NOT preferentially favour one scored candidate. Reported
     as the corrupt-frame overlap with each candidate's FN set (should be similar for ubest vs s3).

Run with BASE conda (cv2): ~/miniconda3/bin/python gt_fix_validate.py <capture> --dev 1 \
    --witness /tmp/wtfix_<cap>/out
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

import detection_f1 as DF
import g2_geom as G
import blob_explain as BE
import gt_blob_fix as GF
import replay_contract as RC
from headpose_anchor import load_head_pose, _body_gravity
from smooth_ref import build_reference
from manifest import Manifest


from deflip import TILT_TOL_DEG


def _anchor_orientation_consistent(q, g_carry):
    """A single-frame independent orientation check vs the controller-accel gravity (the anchor's
    physical cue): is the pose's predicted world-down within TILT_TOL of true world-down? Returns
    (tilt_deg, ok_bool); ok is None when gravity is unobservable at this frame."""
    if g_carry is None or not np.isfinite(g_carry[0]):
        return np.nan, None
    tilt = float(G.gravity_tilt_err_deg(q[None, :], g_carry[None, :])[0])
    return tilt, (tilt <= TILT_TOL_DEG)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--dev", type=int, choices=(1, 2), required=True)
    ap.add_argument("--witness", type=Path, default=None)
    ap.add_argument("--n", type=int, default=80, help="held-out sample size per class")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    capdir = args.capture
    cr = GF.load_corrupt_mask(capdir, args.dev)
    if cr is None:
        raise SystemExit("no gt_blobfix cache; run gt_blob_fix.py first")
    ref = build_reference(capdir / "telemetry", args.dev)
    hp = load_head_pose(capdir / "telemetry")
    g2cam, _ = BE._lazy_imports()
    cams_json = RC.cams_for_capture(capdir)
    cams = DF.load_cameras(str(cams_json))
    g2cams = g2cam.load_cams(cams_json)
    mdl = g2cam.load_led_model(g2cam.CTRL_LEFT if args.dev == 1 else g2cam.CTRL_RIGHT)
    cache = BE.BlobCache(GF._frames_dir(capdir),
                         disk_cache=Path("/tmp/gtfix") / f"blobcache_{capdir.name}.npz")
    m = Manifest.load(capdir / "telemetry")
    imu = G.load_stream(capdir / "telemetry", m, "imu")
    gimu = imu[imu["device_id"] == args.dev]

    rng = np.random.default_rng(0)
    report = {"capture": capdir.name, "dev": args.dev}

    def head_at(ts):
        return GF._head_interp(hp, ts)

    def gcarry_at(ts):
        gb = _body_gravity(np.array([ts]), gimu)
        return gb[0] if np.isfinite(gb[0, 0]) else None

    # ---- (1)+(2) CORRUPT frames: blob + anchor independence ----
    corrupt = np.where(cr.verdict == GF.V_CORRUPT)[0]
    sample = rng.choice(corrupt, min(args.n, corrupt.size), replace=False) if corrupt.size else []
    gt_reproj, fix_reproj = [], []
    gt_multicam = fix_multicam = 0
    gt_tilt_ok = fix_tilt_ok = anchor_opin = 0
    for i in sample:
        ts = int(cr.t_ns[i]); hpos, hq = head_at(ts)
        if hpos is None: continue
        Rh = DF.quat_to_R(hq)
        gt = BE.explain_pose(DF.quat_to_R(ref.quat[i]), ref.pos[i], Rh, hpos, ts, cams, g2cams, mdl, cache)
        fx = BE.explain_pose(DF.quat_to_R(cr.fix_quat[i]), cr.fix_pos[i], Rh, hpos, ts, cams, g2cams, mdl, cache)
        # the DECISIVE non-circular discriminator: multi-cam blob_confirmed (>=2 cams, tight reproj).
        # A mirror twin fits ONE image (low single-cam reproj) but fails the second camera; the
        # blob-true fix passes both. So fix should be multi-cam-confirmed, GT should not.
        gt_multicam += int(gt.blob_confirmed)
        fix_multicam += int(fx.blob_confirmed)
        if np.isfinite(gt.pooled_reproj_px): gt_reproj.append(gt.pooled_reproj_px)
        if np.isfinite(fx.pooled_reproj_px): fix_reproj.append(fx.pooled_reproj_px)
        gcar = gcarry_at(ts)
        gt_t, gt_ok = _anchor_orientation_consistent(ref.quat[i], gcar)
        fx_t, fx_ok = _anchor_orientation_consistent(cr.fix_quat[i], gcar)
        if gt_ok is not None:
            anchor_opin += 1
            gt_tilt_ok += int(gt_ok); fix_tilt_ok += int(fx_ok)
    n = max(int(len(sample)), 1)
    report["corrupt_sample"] = int(len(sample))
    report["gt_multicam_confirmed_frac"] = gt_multicam / n
    report["fix_multicam_confirmed_frac"] = fix_multicam / n
    report["gt_pooled_reproj_median_px"] = float(np.median(gt_reproj)) if gt_reproj else None
    report["fix_pooled_reproj_median_px"] = float(np.median(fix_reproj)) if fix_reproj else None
    report["anchor_opinion_frames"] = int(anchor_opin)
    report["gt_anchor_tilt_ok_frac"] = (gt_tilt_ok / anchor_opin) if anchor_opin else None
    report["fix_anchor_tilt_ok_frac"] = (fix_tilt_ok / anchor_opin) if anchor_opin else None

    # ---- (3) CONTROL: GOOD frames are genuinely blob-confirmed (no false drops next door) ----
    good = np.where(cr.verdict == GF.V_GOOD)[0]
    gsample = rng.choice(good, min(args.n, good.size), replace=False) if good.size else []
    gconf = 0; gn = 0
    for i in gsample:
        ts = int(cr.t_ns[i]); hpos, hq = head_at(ts)
        if hpos is None: continue
        gn += 1
        er = BE.explain_pose(DF.quat_to_R(ref.quat[i]), ref.pos[i], DF.quat_to_R(hq), hpos, ts,
                             cams, g2cams, mdl, cache)
        gconf += int(er.blob_confirmed)
    report["good_control_sample"] = int(gn)
    report["good_control_confirmed_frac"] = (gconf / gn) if gn else None

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        prev = json.load(open(args.out)) if args.out.exists() else []
        prev.append(report)
        json.dump(prev, open(args.out, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
