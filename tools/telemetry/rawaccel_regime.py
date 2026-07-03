#!/usr/bin/env python3
"""Measure the flip-COMMIT regime for the absolute-tilt-reference fix (Step 1).

At every frame where the committed optical pose is a TILT-flip (its body-frame
gravity direction disagrees with the cleaned-GT reference by a large tilt angle,
distinct from a pure-yaw flip), quantify three things, per controller:

  motion        |‖accel‖ − g| at the commit instant (is the controller moving /
                is the raw accel gravity-clean?)
  held_tilt     tilt angle between the HELD ESKF prior orientation's body-gravity
                and the cleaned-GT reference body-gravity (is the held prior
                ALREADY in the flipped basin when the flip commits?)
  rawaccel_tilt tilt angle between the raw-accel gravity direction (world-up via
                head pose, projected to body) and the reference body-gravity.
  gyroprop_tilt tilt of the gyro propagated forward from the last CONFIDENT
                (un-flipped) optical commit vs the reference body-gravity.

This decides which absolute reference to feed the matcher's tilt clamp: raw-accel
gravity (trustworthy only when |‖accel‖−g| small) or gyro-propagated-from-last-
confident-commit (trustworthy through motion, uncorrupted by the flipped fold).

No source change; reads the captures' telemetry + the current-best replay CSVs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g2_geom as G  # noqa: E402
from manifest import Manifest  # noqa: E402
from smooth_ref import build_reference  # noqa: E402
from headpose_anchor import load_head_pose  # noqa: E402

G_MS2 = 9.80665
DOWN = np.array([0.0, -1.0, 0.0])  # OpenXR world-down


def _interp_quat_pos(t_src, q_src, p_src, t_q):
    # nearest-bracket slerp/lerp, returns (q, p) with nan outside the bracket
    out_q = np.full((t_q.shape[0], 4), np.nan)
    out_p = np.full((t_q.shape[0], 3), np.nan)
    for k, t in enumerate(t_q):
        hi = int(np.searchsorted(t_src, t))
        if hi <= 0 or hi >= t_src.shape[0]:
            continue
        lo = hi - 1
        span = float(t_src[hi] - t_src[lo])
        if span <= 0:
            continue
        f = (t - t_src[lo]) / span
        out_p[k] = (1 - f) * p_src[lo] + f * p_src[hi]
        out_q[k] = G.quat_slerp(q_src[lo], q_src[hi], float(f))
    return out_q, out_p


def load_dev_csv(path: Path):
    import csv
    rows = list(csv.DictReader(path.open(newline="")))
    t = np.array([int(r["t_ns"]) for r in rows], dtype=np.int64)
    opt_valid = np.array([int(float(r["opt_valid"])) != 0 for r in rows])
    opt_q = np.array([[float(r["opt_qx"]), float(r["opt_qy"]), float(r["opt_qz"]), float(r["opt_qw"])] for r in rows])
    opt_p = np.array([[float(r["opt_px"]), float(r["opt_py"]), float(r["opt_pz"])] for r in rows])
    pred_q = np.array([[float(r["pred_qx"]), float(r["pred_qy"]), float(r["pred_qz"]), float(r["pred_qw"])] for r in rows])
    pred_tr = np.array([int(float(r["pred_tracked"])) != 0 for r in rows])
    return t, opt_valid, opt_q, opt_p, pred_q, pred_tr


def load_imu(telem: Path, dev: int):
    m = Manifest.load(telem)
    imu = G.load_stream(telem, m, "imu")
    imu = imu[imu["device_id"] == dev]
    t = imu["t_mono_ns"].astype(np.int64)
    a = np.stack([imu["ax"], imu["ay"], imu["az"]], axis=1).astype(np.float64)
    g = np.stack([imu["gx"], imu["gy"], imu["gz"]], axis=1).astype(np.float64)
    order = np.argsort(t)
    return t[order], a[order], g[order]


def nearest(t_src, t, max_dt):
    j = int(np.searchsorted(t_src, t))
    best = -1
    for c in (j - 1, j):
        if 0 <= c < t_src.shape[0] and abs(int(t_src[c]) - int(t)) <= max_dt:
            if best < 0 or abs(int(t_src[c]) - t) < abs(int(t_src[best]) - t):
                best = c
    return best


def tilt_angle(q_a, q_b):
    """Tilt (deg) between two R_world_body orientations: angle between the
    world-down directions each maps from body-down. Independent of yaw."""
    da = G.quat_rotate(q_a, DOWN)
    db = G.quat_rotate(q_b, DOWN)
    da = da / (np.linalg.norm(da) + 1e-12)
    db = db / (np.linalg.norm(db) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(da, db), -1, 1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("replay_out", type=Path, help="dir with dev1.csv/dev2.csv from current-best replay")
    ap.add_argument("--tag", default="cap")
    ap.add_argument("--flip-tilt-deg", type=float, default=15.0, help="min tilt-err vs ref to call a TILT flip")
    ap.add_argument("--confident-tilt-deg", type=float, default=8.0, help="max tilt-err for a CONFIDENT commit")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    telem = args.capture / "telemetry"
    hp = load_head_pose(telem)
    out_all = {}
    for dev in (1, 2):
        ref = build_reference(telem, dev)
        if ref is None:
            out_all[f"dev{dev}"] = {"error": "no reference"}
            continue
        # cleaned reference (RTS-smoothed + deflipped) as R_world_body in OpenXR
        ref_t = ref.t_ns.astype(np.int64)
        ref_q = ref.quat
        ref_valid = ref.valid

        t, opt_valid, opt_q, opt_p, pred_q, pred_tr = load_dev_csv(args.replay_out / f"dev{dev}.csv")
        imu_t, imu_a, imu_g = load_imu(telem, dev)

        # reference at each replay frame
        ref_q_at, _ = _interp_quat_pos(ref_t[ref_valid], ref_q[ref_valid], ref.pos[ref_valid], t)
        ref_ok = np.isfinite(ref_q_at).all(axis=1)

        records = []
        last_confident_idx = -1  # index into the replay-frame array
        last_confident_t = None
        last_confident_pred_q = None
        for i in range(len(t)):
            if not ref_ok[i]:
                continue
            rq = ref_q_at[i]
            # classify the COMMITTED optical pose this frame
            if opt_valid[i] and np.isfinite(opt_q[i]).all() and np.linalg.norm(opt_q[i]) > 0.5:
                committed = G.quat_normalize(opt_q[i][None])[0]
                commit_tilt = tilt_angle(committed, rq)
                commit_geo = float(G.quat_geodesic_deg(committed, rq))
                commit_yaw = abs(float(G.yaw_diff_deg(G.world_yaw_deg(committed), G.world_yaw_deg(rq))))
                is_flip = commit_tilt >= args.flip_tilt_deg
                is_confident = commit_tilt <= args.confident_tilt_deg and commit_geo <= 20.0
            else:
                committed = None
                is_flip = False
                is_confident = False

            # raw accel at this frame
            ai = nearest(imu_t, int(t[i]), int(20e6))
            if ai >= 0:
                a = imu_a[ai]
                accel_norm = float(np.linalg.norm(a))
                # body-up = a_unit; body-down = -a_unit; rawaccel orientation tilt vs ref
                a_unit = a / (accel_norm + 1e-12)
                # tilt of raw-accel gravity: angle between ref-body-down and -a_unit (both body frame)
                ref_body_down = G.quat_rotate_inv(rq, DOWN)
                ref_body_down = ref_body_down / (np.linalg.norm(ref_body_down) + 1e-12)
                rawaccel_tilt = float(np.degrees(np.arccos(np.clip(np.dot(ref_body_down, -a_unit), -1, 1))))
                accel_excess = abs(accel_norm - G_MS2)
            else:
                accel_norm = np.nan
                rawaccel_tilt = np.nan
                accel_excess = np.nan

            # held ESKF prior (the pred orientation = honest filter belief)
            held_tilt = np.nan
            if pred_tr[i] and np.isfinite(pred_q[i]).all() and np.linalg.norm(pred_q[i]) > 0.5:
                pq = G.quat_normalize(pred_q[i][None])[0]
                held_tilt = tilt_angle(pq, rq)

            # gyro propagated from last confident commit
            gyroprop_tilt = np.nan
            coast_gap_ms = np.nan
            if last_confident_t is not None and last_confident_pred_q is not None:
                # integrate gyro from last_confident_t to t[i]
                seg = (imu_t >= last_confident_t) & (imu_t <= int(t[i]))
                if seg.sum() >= 1:
                    q_prop = last_confident_pred_q.copy()
                    ts = imu_t[seg]
                    gs = imu_g[seg]
                    tprev = last_confident_t
                    for k in range(len(ts)):
                        dt = (int(ts[k]) - int(tprev)) / 1e9
                        tprev = int(ts[k])
                        if dt <= 0 or dt > 0.1:
                            continue
                        w = gs[k]  # rad/s body
                        ang = np.linalg.norm(w) * dt
                        if ang > 1e-9:
                            ax = w / np.linalg.norm(w)
                            dq = np.array([ax[0] * np.sin(ang / 2), ax[1] * np.sin(ang / 2),
                                           ax[2] * np.sin(ang / 2), np.cos(ang / 2)])
                            # q_world_body update: world-frame integration q = q (x) dq_body
                            q_prop = G.quat_mul(q_prop, dq)
                            q_prop = G.quat_normalize(q_prop[None])[0]
                    gyroprop_tilt = tilt_angle(q_prop, rq)
                    coast_gap_ms = (int(t[i]) - last_confident_t) / 1e6

            if is_flip:
                records.append({
                    "t_ns": int(t[i]),
                    "commit_tilt_deg": commit_tilt,
                    "commit_yaw_deg": commit_yaw,
                    "commit_geo_deg": commit_geo,
                    "accel_norm": accel_norm,
                    "accel_excess": accel_excess,
                    "rawaccel_tilt_deg": rawaccel_tilt,
                    "held_tilt_deg": held_tilt,
                    "gyroprop_tilt_deg": gyroprop_tilt,
                    "coast_gap_ms": coast_gap_ms,
                })

            # advance last-confident anchor (use the committed opt pose as the gyro seed,
            # since that IS the last trustworthy absolute orientation)
            if is_confident and committed is not None:
                last_confident_t = int(t[i])
                last_confident_pred_q = committed.copy()

        # summarize
        def stat(key, mask=None):
            vals = np.array([r[key] for r in records], dtype=float)
            if mask is not None:
                vals = vals[mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return None
            return {"n": int(vals.size), "median": float(np.median(vals)),
                    "p25": float(np.percentile(vals, 25)), "p75": float(np.percentile(vals, 75)),
                    "p95": float(np.percentile(vals, 95))}

        n = len(records)
        excess = np.array([r["accel_excess"] for r in records], dtype=float)
        clean_mask = np.isfinite(excess) & (excess < 1.0)   # |‖a‖−g| < 1 m/s²: gravity-clean
        moving_mask = np.isfinite(excess) & (excess >= 1.0)
        out_all[f"dev{dev}"] = {
            "n_tilt_flip_commits": n,
            "accel_excess_ms2": stat("accel_excess"),
            "frac_accel_clean_lt1": float(clean_mask.sum()) / max(n, 1),
            "held_tilt_deg": stat("held_tilt_deg"),
            "held_tilt_ge15_frac": float(np.sum(np.array([r["held_tilt_deg"] for r in records]) >= 15.0)) / max(n, 1),
            "rawaccel_tilt_deg_all": stat("rawaccel_tilt_deg"),
            "rawaccel_tilt_deg_when_clean": stat("rawaccel_tilt_deg", clean_mask),
            "rawaccel_tilt_deg_when_moving": stat("rawaccel_tilt_deg", moving_mask),
            "gyroprop_tilt_deg": stat("gyroprop_tilt_deg"),
            "coast_gap_ms": stat("coast_gap_ms"),
        }
        # how often is each reference within 10deg of true (= usable to clamp)?
        for refname in ("held_tilt_deg", "rawaccel_tilt_deg", "gyroprop_tilt_deg"):
            v = np.array([r[refname] for r in records], dtype=float)
            v = v[np.isfinite(v)]
            out_all[f"dev{dev}"][f"{refname}_within10_frac"] = (float(np.sum(v < 10.0)) / max(v.size, 1)) if v.size else None
        out_all[f"dev{dev}"]["rawaccel_within10_when_clean_frac"] = None
        rv = np.array([r["rawaccel_tilt_deg"] for r in records], dtype=float)
        if clean_mask.any():
            rvc = rv[clean_mask]
            rvc = rvc[np.isfinite(rvc)]
            out_all[f"dev{dev}"]["rawaccel_within10_when_clean_frac"] = float(np.sum(rvc < 10.0)) / max(rvc.size, 1)

    out = {"tag": args.tag, "capture": str(args.capture), "devices": out_all}
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
