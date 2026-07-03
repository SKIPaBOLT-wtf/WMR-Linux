#!/usr/bin/env python3
"""Flip-commit regime, v2 — independent raw-accel ground truth (Step 1).

The cleaned-GT reference (RTS-smoothed deflipped optical) is self-referential and
is itself corrupted on flip-heavy frames, so it cannot define "is this commit a
tilt flip". The control (rawaccel_regime control run) proved that when accel is
gravity-clean (|‖a‖−g| < 1 m/s²) the raw-accel gravity direction matches a GOOD
committed pose to ~2-4 deg median. So raw-accel-clean IS an independent absolute
tilt truth. Here we use it as the denominator:

  A frame is a CLEAN-GRAVITY frame when |‖a‖−g| < ACCEL_BAND.
  On clean-gravity frames, the committed optical pose is a TILT-FLIP-COMMIT when
  its body-gravity disagrees with the raw-accel gravity by >= FLIP_TILT_DEG.

For those flip-commits we measure, vs the SAME raw-accel truth:
  held_tilt      held ESKF prior (pred quat) body-gravity vs raw-accel gravity
                 (is the held prior already flipped at the moment the flip commits?)
  gyroprop_tilt  gyro propagated from the last clean-gravity-confirmed orientation,
                 vs raw-accel gravity (does dead-reckoned gyro stay un-flipped?)

This is the measurement that tells us whether feeding raw-accel-clean gravity to
the matcher's tilt clamp (instead of the held, possibly-flipped prior) would have
rejected the flipped twin at commit time.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_geom as G  # noqa: E402
from manifest import Manifest  # noqa: E402

G_MS2 = 9.80665
DOWN = np.array([0.0, -1.0, 0.0])


def load_imu(telem, dev):
    m = Manifest.load(telem)
    imu = G.load_stream(telem, m, "imu")
    imu = imu[imu["device_id"] == dev]
    t = imu["t_mono_ns"].astype(np.int64)
    a = np.stack([imu["ax"], imu["ay"], imu["az"]], 1).astype(np.float64)
    g = np.stack([imu["gx"], imu["gy"], imu["gz"]], 1).astype(np.float64)
    o = np.argsort(t)
    return t[o], a[o], g[o]


def load_dev_csv(path):
    rows = list(csv.DictReader(path.open(newline="")))
    t = np.array([int(r["t_ns"]) for r in rows], dtype=np.int64)
    ov = np.array([int(float(r["opt_valid"])) != 0 for r in rows])
    oq = np.array([[float(r["opt_qx"]), float(r["opt_qy"]), float(r["opt_qz"]), float(r["opt_qw"])] for r in rows])
    pq = np.array([[float(r["pred_qx"]), float(r["pred_qy"]), float(r["pred_qz"]), float(r["pred_qw"])] for r in rows])
    pt = np.array([int(float(r["pred_tracked"])) != 0 for r in rows])
    return t, ov, oq, pq, pt


def gravity_tilt_deg_q_vs_accel(q_world_body, accel_unit):
    """tilt (deg) between a pose's body-gravity-down and the raw-accel gravity-down.
    body-down for the pose is q^-1 applied to world-down... but raw accel is in body
    frame directly: body-up = accel_unit, body-down = -accel_unit. The pose's claim of
    body-down is q^-1 . world_down. Compare in body frame."""
    pose_body_down = G.quat_rotate_inv(q_world_body, DOWN)
    pose_body_down = pose_body_down / (np.linalg.norm(pose_body_down) + 1e-12)
    accel_body_down = -accel_unit
    return float(np.degrees(np.arccos(np.clip(np.dot(pose_body_down, accel_body_down), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("replay_out", type=Path)
    ap.add_argument("--tag", default="cap")
    ap.add_argument("--accel-band", type=float, default=1.0)
    ap.add_argument("--flip-tilt-deg", type=float, default=15.0)
    ap.add_argument("--confident-tilt-deg", type=float, default=6.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    telem = args.capture / "telemetry"
    out_dev = {}
    for dev in (1, 2):
        it, ia, ig = load_imu(telem, dev)
        t, ov, oq, pq, pt = load_dev_csv(args.replay_out / f"dev{dev}.csv")

        last_conf_t = None
        last_conf_q = None  # last clean-gravity-confirmed body orientation (committed opt)
        recs = []
        n_clean = 0
        n_commit_on_clean = 0
        for i in range(len(t)):
            j = int(np.searchsorted(it, int(t[i])))
            best = -1
            for c in (j - 1, j):
                if 0 <= c < len(it) and abs(int(it[c]) - int(t[i])) <= 20e6:
                    if best < 0 or abs(int(it[c]) - int(t[i])) < abs(int(it[best]) - int(t[i])):
                        best = c
            if best < 0:
                continue
            a = ia[best]
            anorm = float(np.linalg.norm(a))
            excess = abs(anorm - G_MS2)
            if excess >= args.accel_band:
                continue  # not a clean-gravity frame; raw accel is contaminated -> skip
            n_clean += 1
            a_unit = a / (anorm + 1e-12)

            committed = None
            if ov[i] and np.isfinite(oq[i]).all() and np.linalg.norm(oq[i]) > 0.5:
                committed = G.quat_normalize(oq[i][None])[0]
                n_commit_on_clean += 1
                commit_tilt = gravity_tilt_deg_q_vs_accel(committed, a_unit)

                held = np.nan
                if pt[i] and np.isfinite(pq[i]).all() and np.linalg.norm(pq[i]) > 0.5:
                    held = gravity_tilt_deg_q_vs_accel(G.quat_normalize(pq[i][None])[0], a_unit)

                gyroprop = np.nan
                coast_ms = np.nan
                if last_conf_t is not None and last_conf_q is not None:
                    seg = (it >= last_conf_t) & (it <= int(t[i]))
                    if seg.sum() >= 1:
                        qp = last_conf_q.copy()
                        ts = it[seg]
                        gs = ig[seg]
                        tp = last_conf_t
                        for k in range(len(ts)):
                            dt = (int(ts[k]) - int(tp)) / 1e9
                            tp = int(ts[k])
                            if dt <= 0 or dt > 0.1:
                                continue
                            w = gs[k]
                            ang = float(np.linalg.norm(w)) * dt
                            if ang > 1e-9:
                                ax = w / np.linalg.norm(w)
                                dq = np.array([ax[0] * np.sin(ang / 2), ax[1] * np.sin(ang / 2),
                                               ax[2] * np.sin(ang / 2), np.cos(ang / 2)])
                                qp = G.quat_normalize(G.quat_mul(qp, dq)[None])[0]
                        gyroprop = gravity_tilt_deg_q_vs_accel(qp, a_unit)
                        coast_ms = (int(t[i]) - last_conf_t) / 1e6

                if commit_tilt >= args.flip_tilt_deg:
                    recs.append({"commit_tilt": commit_tilt, "held": held, "gyroprop": gyroprop,
                                 "coast_ms": coast_ms, "excess": excess})

                # update last-confident from a clean-gravity-AGREEING committed pose
                if commit_tilt <= args.confident_tilt_deg:
                    last_conf_t = int(t[i])
                    last_conf_q = committed.copy()

        def stat(key, mask=None):
            v = np.array([r[key] for r in recs], dtype=float)
            if mask is not None:
                v = v[mask]
            v = v[np.isfinite(v)]
            if v.size == 0:
                return None
            return {"n": int(v.size), "median": round(float(np.median(v)), 2),
                    "p25": round(float(np.percentile(v, 25)), 2), "p75": round(float(np.percentile(v, 75)), 2),
                    "p95": round(float(np.percentile(v, 95)), 2),
                    "within10_frac": round(float(np.mean(v < 10.0)), 3),
                    "within15_frac": round(float(np.mean(v < 15.0)), 3)}

        n = len(recs)
        coast = np.array([r["coast_ms"] for r in recs], dtype=float)
        short = np.isfinite(coast) & (coast < 200)
        out_dev[f"dev{dev}"] = {
            "n_clean_gravity_frames": n_clean,
            "n_commit_on_clean": n_commit_on_clean,
            "n_tilt_flip_commits_on_clean": n,
            "flip_commit_rate_on_clean": round(n / max(n_commit_on_clean, 1), 4),
            "commit_tilt_deg": stat("commit_tilt"),
            "held_tilt_deg": stat("held"),
            "gyroprop_tilt_deg": stat("gyroprop"),
            "gyroprop_tilt_deg_shortcoast_lt200ms": stat("gyroprop", short),
            "coast_ms": stat("coast_ms"),
        }
    res = {"tag": args.tag, "accel_band": args.accel_band, "flip_tilt_deg": args.flip_tilt_deg, "devices": out_dev}
    print(json.dumps(res, indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
