#!/usr/bin/env python3
"""Deterministic failure-miner across ALL captures, using only the UNIVERSAL telemetry streams that every
capture has (frame, pose_attempt, fusion) -- no offline replay needed. Pre-filters the raw frame pool to a
failure-rich, stratified candidate set for the annotation fleet, and decomposes wrong-pose errors into the
tilt(roll+pitch) vs yaw axes.

Signals (calibrated against xv1's richer candidate stream):
  pose_attempt  -> the COMMITTED poses (device_id, cam_id, hw_ts, blobs_matched, pose); the commit timeline.
  fusion        -> per committed update, the optical pose vs the IMU-predicted pose; the rotation residual
                   decomposed into TILT (about world X/Z, gravity-anchored => reliable flip signal) and YAW
                   (about world Y => noisy, the loose-yaw prior). Joined to a frame via nearest t_mono.
  frame         -> per (cam, hw_ts) detected blob count; exposure==20 == short-exposure controller frame.

Per controller-frame x device:
  WRONG_TILT / WRONG_YAW  committed but the optical pose disagrees with the gravity-anchored IMU by > FLIP_DEG
                          on that axis (tilt-dominant vs yaw-dominant) -- a flip / wrong pose.
  LOW_MATCH               committed on < LOW_MATCH matched blobs (weak / ambiguous).
  NO_COMMIT_LEDS          present (bracketed by commits) but NOT committed though >= MIN_PNP_BLOBS detected.
  DETECT_FAIL             present but NOT committed and < MIN_PNP_BLOBS detected -- the detection miss.
"present" = the device has a commit within +/-BRACKET on BOTH temporal sides (so it was really there).

  ~/miniconda3/envs/g2vr/bin/python failure_mine.py [capture_dir ...]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "../telemetry"))
from manifest import Manifest
import g2_geom as G

BRACKET_NS = 250_000_000
FRAME_MATCH_NS = 9_000_000     # a commit "at" this frame
FLIP_DEG = 15.0                # optical-vs-IMU residual above which a committed pose is a wrong-pose candidate
LOW_MATCH = 5
MIN_PNP_BLOBS = 4
CAPS_DEFAULT = sorted(str(p) for p in Path("/home/mrwhite0racle/g2-linux-research/captures").glob("2026*")
                      if (p / "telemetry").exists() and not p.name.endswith("framebin"))

def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def tilt_yaw_resid(opt_q, pred_q):
    """|rotation error| between optical and IMU-predicted pose, split into tilt(roll+pitch) and yaw (deg)."""
    dR = q2R(opt_q) @ q2R(pred_q).T
    ang = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    if ang < 1e-6:
        return 0.0, 0.0
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * np.sin(ang))
    rv = ax * ang
    return float(np.rad2deg(np.hypot(rv[0], rv[2]))), float(np.rad2deg(abs(rv[1])))

def mine(cap: Path):
    tel = cap / "telemetry"
    m = Manifest.load(tel)
    need = {"frame", "pose_attempt", "fusion"}
    if not need.issubset(m.streams):
        return []
    fr = G.load_stream(tel, m, "frame"); pa = G.load_stream(tel, m, "pose_attempt"); fu = G.load_stream(tel, m, "fusion")
    fr = fr[fr["exposure"] == 20]                                    # controller (short-exp) frames only
    if len(fr) == 0 or len(pa) == 0:
        return []
    frame_ts = np.unique(fr["hw_ts_ns"].astype(np.int64))
    rows = []
    for dev in (1, 2):
        pad = pa[pa["device_id"] == dev]
        if len(pad) == 0:
            continue
        order = np.argsort(pad["hw_ts_ns"].astype(np.int64)); pad = pad[order]
        cts = pad["hw_ts_ns"].astype(np.int64)                      # this device's commit timeline (hw_ts)
        fud = fu[fu["device_id"] == dev]
        # attach a tilt/yaw residual to each commit via nearest t_mono (fusion has t_mono, not hw_ts)
        resid = {}                                                   # hw_ts -> (tilt_deg, yaw_deg)
        if len(fud):
            ft = fud["t_mono_ns"].astype(np.int64); fo = np.argsort(ft); ft = ft[fo]; fud = fud[fo]
            pmono = pad["t_mono_ns"].astype(np.int64)
            idx = np.clip(np.searchsorted(ft, pmono), 0, len(ft) - 1)
            for k, r in enumerate(pad):
                j = idx[k]
                if j > 0 and abs(ft[j - 1] - pmono[k]) < abs(ft[j] - pmono[k]):
                    j -= 1
                if abs(ft[j] - pmono[k]) < FRAME_MATCH_NS:
                    fr_ = fud[j]
                    resid[int(r["hw_ts_ns"])] = tilt_yaw_resid(
                        [fr_["opt_qx"], fr_["opt_qy"], fr_["opt_qz"], fr_["opt_qw"]],
                        [fr_["pred_qx"], fr_["pred_qy"], fr_["pred_qz"], fr_["pred_qw"]])
        for ts in frame_ts:
            # is the device present (bracketed by commits within +/-BRACKET on both sides)?
            lo = cts[cts <= ts]; hi = cts[cts >= ts]
            present = len(lo) and len(hi) and (ts - lo[-1] < BRACKET_NS) and (hi[0] - ts < BRACKET_NS)
            if not present:
                continue
            near = np.abs(cts - ts) < FRAME_MATCH_NS
            fm = fr[np.abs(fr["hw_ts_ns"].astype(np.int64) - ts) < FRAME_MATCH_NS]
            n_blobs = int(fm["n_blobs"].max()) if len(fm) else 0
            if near.any():                                          # committed at this frame
                c = pad[near][np.argmax(pad[near]["blobs_matched"])]
                cam = int(c["cam_id"]); bm = int(c["blobs_matched"])
                tilt, yaw = resid.get(int(c["hw_ts_ns"]), (0.0, 0.0))
                if max(tilt, yaw) > FLIP_DEG:
                    mode = "WRONG_TILT" if tilt >= yaw else "WRONG_YAW"
                elif bm < LOW_MATCH:
                    mode = "LOW_MATCH"
                else:
                    continue                                        # OK
                rows.append(dict(cap=cap.name, dev=dev, cam=cam, ts=int(ts), mode=mode, n_blobs=n_blobs,
                                 blobs_matched=bm, tilt_deg=round(tilt, 1), yaw_deg=round(yaw, 1)))
            else:                                                   # present but not committed this frame
                cam = int(pad[np.argmin(np.abs(cts - ts))]["cam_id"])   # likely cam = nearest commit's cam
                mode = "DETECT_FAIL" if n_blobs < MIN_PNP_BLOBS else "NO_COMMIT_LEDS"
                rows.append(dict(cap=cap.name, dev=dev, cam=cam, ts=int(ts), mode=mode, n_blobs=n_blobs,
                                 blobs_matched=0, tilt_deg=None, yaw_deg=None))
    return rows

def main():
    caps = [Path(c) for c in (sys.argv[1:] or CAPS_DEFAULT)]
    catalog = []; grand = {}; seen_sig = {}
    MODES = ["DETECT_FAIL", "NO_COMMIT_LEDS", "LOW_MATCH", "WRONG_TILT", "WRONG_YAW"]
    for cap in caps:
        rows = mine(cap)
        # dedup whole captures that are re-dumps of the same session (identical hw_ts set => identical rows)
        sig = (len(rows), tuple(sorted({r["ts"] for r in rows}))[:50])
        if rows and sig in seen_sig:
            print(f"\n=== {cap.name} ===  DUPLICATE of {seen_sig[sig]} (identical telemetry) -> skipped")
            continue
        if rows:
            seen_sig[sig] = cap.name
        catalog += rows
        per = {}
        for r in rows:
            per[r["mode"]] = per.get(r["mode"], 0) + 1; grand[r["mode"]] = grand.get(r["mode"], 0) + 1
        if rows:
            print(f"\n=== {cap.name} ===  failure-candidates={len(rows)}")
            for c in MODES:
                if per.get(c): print(f"    {c:16} {per[c]}")
        else:
            print(f"\n=== {cap.name} ===  (no usable telemetry / no failures)")
    print("\n================ GRAND TOTAL (deduped, all captures, no replay) ================")
    tot = sum(grand.values())
    confirmed = sum(grand.get(c, 0) for c in ("DETECT_FAIL", "NO_COMMIT_LEDS", "WRONG_TILT", "WRONG_YAW"))
    for c in MODES:
        if grand.get(c):
            tag = "  <- weak (committed, IMU-consistent; sample down)" if c == "LOW_MATCH" else ""
            print(f"    {c:16} {grand[c]:5}  ({100*grand[c]/max(tot,1):.0f}%){tag}")
    print(f"  CONFIRMED-failure candidates (detect/no-commit/wrong-pose): {confirmed}")
    print(f"  + weak LOW_MATCH: {grand.get('LOW_MATCH',0)}   TOTAL: {tot}")
    fl = grand.get("WRONG_TILT", 0) + grand.get("WRONG_YAW", 0)
    if fl:
        print(f"  wrong-pose axis (telemetry pre-filter; GT confirms): TILT={grand.get('WRONG_TILT',0)} "
              f"({100*grand.get('WRONG_TILT',0)/fl:.0f}%)  YAW={grand.get('WRONG_YAW',0)} ({100*grand.get('WRONG_YAW',0)/fl:.0f}%)")
    outp = Path(__file__).parent / "dataset" / "failure_catalog.json"
    json.dump(catalog, open(outp, "w"))
    print(f"\n  wrote {len(catalog)} failure-candidate frames -> {outp}")

if __name__ == "__main__":
    main()
