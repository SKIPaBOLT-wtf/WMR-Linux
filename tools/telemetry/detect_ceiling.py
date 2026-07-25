#!/usr/bin/env python3
"""detect_ceiling.py -- detection fraction vs head-relative elevation + the top-edge regime.

Promoted from the 2026-07-03 C2 top-edge analysis (results/session-analysis-20260703/
c2-top-edge/{ep_extract,coverage,detect_ceiling}.py; findings.md). The over-forehead
complaint splits into a HARD camera ceiling (position-coverage boundary, ~+25..+31 deg
frontal from the session's own hmd-cameras.json) and a DETECTION DEAD BAND starting
~+15 deg where LEDs still project in-image but the detector finds no blobs
(emission-cone rolloff). Both must be standing numbers.

Method (matches the C2 scripts exactly):
  * head-relative series per device from getpose + head_pose (getpose world -> tracker
    world via a fitted translation + grip lever/quat offset against fusion pred rows);
    elevation/azimuth of the felt controller position in the XR head frame.
  * detection sampling at 10 Hz: per instant, projected-visible LED count (best cam) and
    detected ctrl blobs (blob.bin within GATE_PX of this device's LEDs and not closer to
    the other device's), accept within +/-60 ms (fusion outcome==1), POSITION_TRACKED.
  * position-coverage boundary per azimuth from the rt8 camera model at r=0.45 m.

Emits the elevation-binned curve plus two scalars:
  * shoulder50_deg   -- elevation where frac(>=1 ctrl blob) crosses 0.5 going up
  * ceiling dwell    -- untracked dwell fraction for felt positions AT/BEYOND the
                        coverage boundary (the pure-physics regime)

BASELINE (capture 20260703-202811-capture-diversity-postflip; reproduces c2-top-edge/
findings.md + detect_ceiling.csv + dwell_rate.csv):
  frac(>=1 ctrl blob) by elevation band: +0..+5 0.95, +15..+20 0.53, +20..+25 0.29,
    +25..+30 0.16, above +30 0.00 (geometric frac(>=4 LEDs) ~1.0 until +30)
  shoulder50 ~ +18.2 deg (inside the +15-20 deg band)
  boundary over |az|<=60: min +25 / median +30 / max +31 deg
  untracked dwell frac +15..+20: dev1 0.48, dev2 0.24; +25..+30: dev1 0.72, dev2 0.80

Usage: detect_ceiling.py <capture_dir> [--out DIR] [--step-ms 100] [--gate-px 12]
                         [--radius 0.45] [--cams JSON] [--ctrl-left JSON] [--ctrl-right JSON]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest  # noqa: E402
import g2_geom as G  # noqa: E402
import detection_f1 as DF  # noqa: E402
import replay_contract as RC  # noqa: E402

POS_VALID = 1 << 1
POS_TRACKED = 1 << 5
XR_CV_FLIP = np.diag([1.0, -1.0, -1.0])
ELEV_BINS = np.arange(-90, 91, 5)
CURVE_MIN_N = 5


# ---------------------------------------------------------------- head-relative series

def head_interp(hp: np.ndarray, t_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear pos + nlerp quat interpolation of the head_pose stream at t_query (ns)."""
    t = hp["t_mono_ns"].astype(np.int64)
    good = t > 9.0e12  # drop pre-clock-sync warm-up rows
    t = t[good]
    pos = np.stack([hp["px"][good], hp["py"][good], hp["pz"][good]], axis=-1).astype(float)
    quat = np.stack([hp["qx"][good], hp["qy"][good], hp["qz"][good], hp["qw"][good]], axis=-1).astype(float)
    for i in range(1, len(quat)):
        if np.dot(quat[i], quat[i - 1]) < 0:
            quat[i] = -quat[i]
    i = np.clip(np.searchsorted(t, t_query), 1, len(t) - 1)
    t0, t1 = t[i - 1], t[i]
    w = np.clip((t_query - t0) / np.maximum(t1 - t0, 1), 0.0, 1.0)[:, None]
    p = pos[i - 1] * (1 - w) + pos[i] * w
    q = G.quat_normalize(quat[i - 1] * (1 - w) + quat[i] * w)
    return p, q


def fit_world_offset(fu: np.ndarray, gp: np.ndarray, dev: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """getpose world = tracker world + translation; plus the constant body-frame grip
    lever arm and grip quat offset (q_getpose = q_model * q_off), fit against fusion
    pred rows (C2's measured rotation delta was 0.56 deg -- treated as pure translation)."""
    f = fu[fu["device_id"] == dev]
    g = gp[gp["device_id"] == dev]
    tf = f["t_mono_ns"].astype(np.int64)
    tg = g["t_mono_ns"].astype(np.int64)
    idx = np.clip(np.searchsorted(tg, tf), 1, len(tg) - 1)
    pick = np.where(np.abs(tg[idx] - tf) < np.abs(tg[idx - 1] - tf), idx, idx - 1)
    ok = np.abs(tg[pick] - tf) / 1e6 < 3
    FP = np.stack([f["pred_px"], f["pred_py"], f["pred_pz"]], -1)[ok].astype(float)
    GPp = np.stack([g["px"], g["py"], g["pz"]], -1)[pick][ok].astype(float)
    QF = G.quat_normalize(np.stack([f["pred_qx"], f["pred_qy"], f["pred_qz"], f["pred_qw"]], -1)[ok].astype(float))
    QG = G.quat_normalize(np.stack([g["qx"], g["qy"], g["qz"], g["qw"]], -1)[pick][ok].astype(float))
    t_world = np.median(GPp - FP, axis=0)
    res_body = G.quat_rotate_inv(QG, GPp - t_world - FP)
    lever = np.median(res_body, axis=0)
    qoff = G.quat_mul(G.quat_conj(QF), QG)
    qoff = G.quat_canonical_sign(qoff, qoff[np.argmax(np.abs(qoff[:, 3]))])
    q_off = G.quat_normalize(np.median(qoff, axis=0))
    return t_world, lever, q_off


def headrel_series(gp: np.ndarray, hp: np.ndarray, fu: np.ndarray, dev: int) -> dict[str, np.ndarray]:
    """Per-getpose-pull head-relative geometry of the felt (tracker-world model) pose."""
    t_world, lever, q_off = fit_world_offset(fu, gp, dev)
    g = gp[gp["device_id"] == dev]
    t = g["t_mono_ns"].astype(np.int64)
    order = np.argsort(t)
    g, t = g[order], t[order]
    flags = g["relation_flags"].astype(np.uint32)
    q_g = G.quat_normalize(np.stack([g["qx"], g["qy"], g["qz"], g["qw"]], axis=-1).astype(float))
    p = (np.stack([g["px"], g["py"], g["pz"]], axis=-1).astype(float)
         - t_world - G.quat_rotate(q_g, lever))
    q_model = G.quat_mul(q_g, G.quat_conj(q_off))
    first_valid = int(np.argmax((flags & POS_VALID) != 0))
    t, flags, p, q_model = t[first_valid:], flags[first_valid:], p[first_valid:], q_model[first_valid:]
    hpos, hq = head_interp(hp, t)
    rel = G.quat_rotate_inv(hq, p - hpos)  # XR head frame: X right, Y up, -Z forward
    return dict(
        t=t, p=p, q_model=q_model, hpos=hpos, hq=hq, rel=rel,
        dist=np.linalg.norm(rel, axis=-1),
        elev=np.degrees(np.arctan2(rel[:, 1], np.hypot(rel[:, 0], rel[:, 2]))),
        az=np.degrees(np.arctan2(rel[:, 0], -rel[:, 2])),
        tracked=(flags & POS_TRACKED) != 0,
    )


# ---------------------------------------------------------------- coverage boundary

def top_boundary(cams: list[DF.Camera], radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-azimuth top elevation still position-covered by ANY camera (rt8 frustum at
    the given head-frame radius). LED facing/exposure can only shrink this."""
    az = np.arange(-180.0, 180.1, 1.0)
    el = np.arange(-90.0, 90.1, 1.0)
    azg, elg = np.meshgrid(az, el)
    azr, elr = np.radians(azg), np.radians(elg)
    p_xr = np.stack([np.cos(elr) * np.sin(azr), np.sin(elr), -np.cos(elr) * np.cos(azr)], axis=-1) * radius
    p_cv = p_xr @ XR_CV_FLIP.T
    covered = np.zeros(p_cv.shape[:2], dtype=bool)
    for c in cams:
        R_cam_imu, t_cam_imu = DF.pose_inv(c.P_imu_cam_R, c.P_imu_cam_t)
        pc = p_cv @ R_cam_imu.T + t_cam_imu
        u, v, ok = DF.rt8_project(pc, c)
        covered |= ok & (pc[..., 2] > 0.05) & (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)
    top = np.full(len(az), np.nan)
    for i in range(len(az)):
        idx = np.flatnonzero(covered[:, i])
        if idx.size:
            top[i] = el[idx.max()]
    return az, top


# ---------------------------------------------------------------- detection sampling

def sample_detection(telem: Path, series: dict[int, dict[str, np.ndarray]],
                     cams: list[DF.Camera], led: dict[int, tuple[np.ndarray, np.ndarray]],
                     step_ns: int, gate_px: float) -> np.ndarray:
    """10 Hz per-device census: (dev, elev, az, nvis_max, nblob_best, accepted, tracked)."""
    m = Manifest.load(telem)
    bl = G.load_stream(telem, m, "blob")
    fr = G.load_stream(telem, m, "frame")
    fu = G.load_stream(telem, m, "fusion")

    fidx = {}
    for c in range(4):
        s = fr[fr["cam_id"] == c]
        o = np.argsort(s["t_mono_ns"])
        fidx[c] = (s["t_mono_ns"][o].astype(np.int64), s["hw_ts_ns"][o].astype(np.int64))
    bl_key: dict[tuple[int, int], list[int]] = {}
    for i in range(len(bl)):
        bl_key.setdefault((int(bl["cam_id"][i]), int(bl["hw_ts_ns"][i])), []).append(i)

    recs = []
    for dev in (1, 2):
        odev = 2 if dev == 1 else 1
        S, SO = series[dev], series[odev]
        lp, ln = led[dev]
        olp, oln = led[odev]
        acc_t = np.sort(fu[(fu["device_id"] == dev) & (fu["outcome"] == 1)]["t_mono_ns"].astype(np.int64))

        for ts in np.arange(S["t"][0], S["t"][-1], step_ns):
            gi = np.clip(np.searchsorted(S["t"], ts), 1, len(S["t"]) - 1)
            p, q, hp_, hq_ = S["p"][gi], S["q_model"][gi], S["hpos"][gi], S["hq"][gi]
            oi = np.clip(np.searchsorted(SO["t"], ts), 1, len(SO["t"]) - 1)
            po, qo = SO["p"][oi], SO["q_model"][oi]
            Rd, Rh, Ro = DF.quat_to_R(q), DF.quat_to_R(hq_), DF.quat_to_R(qo)
            nvis_max, nblob_best = 0, 0
            for c in cams:
                u, v = DF.project_device_leds(lp, ln, Rd, p, Rh, hp_, c)
                if len(u) == 0:
                    continue
                nvis_max = max(nvis_max, len(u))
                tm, hw = fidx[c.id]
                j = np.clip(np.searchsorted(tm, ts), 1, len(tm) - 1)
                j = j if abs(tm[j] - ts) < abs(tm[j - 1] - ts) else j - 1
                if abs(int(tm[j]) - ts) > 40e6:
                    continue
                bidx = bl_key.get((c.id, int(hw[j])), [])
                if not bidx:
                    continue
                B = bl[bidx]
                bx = np.stack([B["x"], B["y"]], -1).astype(float)
                uv = np.stack([u, v], -1)
                d_self = np.min(np.linalg.norm(bx[:, None] - uv[None], axis=-1), axis=1)
                ou, ov = DF.project_device_leds(olp, oln, Ro, po, Rh, hp_, c)
                if len(ou):
                    ouv = np.stack([ou, ov], -1)
                    d_oth = np.min(np.linalg.norm(bx[:, None] - ouv[None], axis=-1), axis=1)
                else:
                    d_oth = np.full(len(B), 1e9)
                nblob_best = max(nblob_best, int(((d_self <= gate_px) & (d_self <= d_oth)).sum()))
            k = np.searchsorted(acc_t, ts)
            near_acc = any(0 <= kk < len(acc_t) and abs(int(acc_t[kk]) - int(ts)) < 60e6
                           for kk in (k - 1, k))
            recs.append((dev, float(S["elev"][gi]), float(S["az"][gi]), nvis_max, nblob_best,
                         int(near_acc), int(S["tracked"][gi])))
    return np.array(recs)


# ---------------------------------------------------------------- aggregation

def curve_rows(R: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lo in ELEV_BINS[:-1]:
        sel = (R[:, 1] >= lo) & (R[:, 1] < lo + 5)
        if sel.sum() < CURVE_MIN_N:
            continue
        r = R[sel]
        rows.append(dict(elev_lo=int(lo), n=int(sel.sum()),
                         med_leds=float(np.median(r[:, 3])),
                         frac_leds4=float((r[:, 3] >= 4).mean()),
                         frac_blob1=float((r[:, 4] >= 1).mean()),
                         frac_blob3=float((r[:, 4] >= 3).mean()),
                         accept_rate=float(r[:, 5].mean()),
                         tracked=float(r[:, 6].mean())))
    return rows


def shoulder50(rows: list[dict[str, Any]]) -> float | None:
    """Elevation of the first downward 0.5-crossing of frac(>=1 ctrl blob) above 0 deg
    (linear interpolation between 5-deg band centers)."""
    pts = [(r["elev_lo"] + 2.5, r["frac_blob1"]) for r in rows if r["elev_lo"] >= 0]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if y0 >= 0.5 > y1:
            return round(x0 + (x1 - x0) * (y0 - 0.5) / (y0 - y1), 1)
    return None


def dwell_rows(series: dict[int, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for dev in (1, 2):
        S = series[dev]
        dt = np.clip(np.diff(S["t"]) / 1e9, 0, 0.05)
        eb = S["elev"][:-1]
        dwell, _ = np.histogram(eb, bins=ELEV_BINS, weights=dt)
        untracked, _ = np.histogram(eb[~S["tracked"][:-1]], bins=ELEV_BINS,
                                    weights=dt[~S["tracked"][:-1]])
        for i in range(len(ELEV_BINS) - 1):
            if dwell[i] <= 0:
                continue
            rows.append(dict(dev=dev, elev_lo=int(ELEV_BINS[i]), elev_hi=int(ELEV_BINS[i + 1]),
                             dwell_s=round(float(dwell[i]), 2),
                             untracked_s=round(float(untracked[i]), 2),
                             untracked_frac=round(float(untracked[i] / dwell[i]), 3)))
    return rows


def ceiling_dwell(series: dict[int, dict[str, np.ndarray]],
                  bnd_az: np.ndarray, bnd_top: np.ndarray) -> dict[str, Any]:
    """Untracked dwell fraction while the felt position sits AT/BEYOND the coverage
    boundary at its azimuth (the pure-physics over-the-top regime)."""
    fin = np.isfinite(bnd_top)
    out: dict[str, Any] = {}
    for dev in (1, 2):
        S = series[dev]
        dt = np.clip(np.diff(S["t"]) / 1e9, 0, 0.05)
        top_at = np.interp(S["az"][:-1], bnd_az[fin], bnd_top[fin], left=np.nan, right=np.nan)
        beyond = np.isfinite(top_at) & (S["elev"][:-1] >= top_at)
        dwell = float(dt[beyond].sum())
        untr = float(dt[beyond & ~S["tracked"][:-1]].sum())
        out[str(dev)] = dict(dwell_s=round(dwell, 2), untracked_s=round(untr, 2),
                             untracked_frac=round(untr / dwell, 3) if dwell > 0 else None)
    return out


# ---------------------------------------------------------------- main

def _default_provenance(capture: Path, pattern: str, fallback: str) -> str:
    hits = sorted((capture / "provenance").glob(pattern))
    return str(hits[0]) if hits else fallback


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="capture dir (telemetry/ + provenance/)")
    ap.add_argument("--out", type=Path, help="write detect_ceiling.{csv,png}, dwell CSV + JSON here")
    ap.add_argument("--step-ms", type=float, default=100.0, help="census sampling period")
    ap.add_argument("--gate-px", type=float, default=12.0, help="blob-to-LED association gate")
    ap.add_argument("--radius", type=float, default=0.45, help="boundary test radius (m, head frame)")
    ap.add_argument("--cams", help="hmd-cameras.json (default: capture provenance snapshot)")
    ap.add_argument("--ctrl-left", help="left controller LED model JSON")
    ap.add_argument("--ctrl-right", help="right controller LED model JSON")
    args = ap.parse_args()

    capture = args.capture
    telem = capture / "telemetry" if (capture / "telemetry" / "manifest.json").is_file() else capture
    cams_json = args.cams or str(RC.cams_for_capture(capture))
    ctrl_l = args.ctrl_left or _default_provenance(capture, "controller_*L.json", DF.DEFAULT_CTRL_LEFT)
    ctrl_r = args.ctrl_right or _default_provenance(capture, "controller_*R.json", DF.DEFAULT_CTRL_RIGHT)
    cams = DF.load_cameras(cams_json)
    led = {1: DF.load_led_model(ctrl_l), 2: DF.load_led_model(ctrl_r)}
    print(f"# detect_ceiling: {capture}\ncams: {cams_json}\nctrl: {ctrl_l} | {ctrl_r}")

    m = Manifest.load(telem)
    gp = G.load_stream(telem, m, "getpose")
    hp = G.load_stream(telem, m, "head_pose")
    fu = G.load_stream(telem, m, "fusion")
    series = {dev: headrel_series(gp, hp, fu, dev) for dev in (1, 2)}

    bnd_az, bnd_top = top_boundary(cams, args.radius)
    core = np.abs(bnd_az) <= 60
    print(f"coverage boundary (any cam, r={args.radius} m), az [-60,60]: "
          f"top elev min={np.nanmin(bnd_top[core]):.0f} median={np.nanmedian(bnd_top[core]):.0f} "
          f"max={np.nanmax(bnd_top[core]):.0f} deg")

    R = sample_detection(telem, series, cams, led, int(args.step_ms * 1e6), args.gate_px)
    rows = curve_rows(R)
    print("\nelev_bin    n   medLEDs  frac_leds>=4  frac_blob>=1  frac_blob>=3  accept  tracked")
    for r in rows:
        print(f"{r['elev_lo']:+4d}..{r['elev_lo'] + 5:+4d} {r['n']:4d}  {r['med_leds']:5.1f}   "
              f"{r['frac_leds4']:10.2f}  {r['frac_blob1']:12.2f}  {r['frac_blob3']:12.2f}  "
              f"{r['accept_rate']:6.2f}  {r['tracked']:7.2f}")

    sh50 = shoulder50(rows)
    cdwell = ceiling_dwell(series, bnd_az, bnd_top)
    drows = dwell_rows(series)
    print(f"\nshoulder50 (frac_blob>=1 crosses 0.5): {sh50} deg" if sh50 is not None
          else "\nshoulder50: no 0.5 crossing above 0 deg")
    for dev, c in cdwell.items():
        print(f"hard-ceiling dwell dev{dev}: {c['dwell_s']} s at/beyond boundary, "
              f"untracked_frac={c['untracked_frac']}")
    print("\nuntracked dwell by elevation band (>=+10 deg):")
    for r in drows:
        if r["elev_lo"] >= 10:
            print(f"  dev{r['dev']} {r['elev_lo']:+3d}..{r['elev_hi']:+3d}: dwell={r['dwell_s']:5.2f}s "
                  f"untracked_frac={r['untracked_frac']:.3f}")

    if args.out:
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
        with (out / "detect_ceiling.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with (out / "dwell_by_elevation.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
            w.writeheader()
            w.writerows(drows)
        (out / "detect_ceiling.json").write_text(json.dumps(dict(
            capture=str(capture), radius_m=args.radius, gate_px=args.gate_px,
            step_ms=args.step_ms,
            boundary_core=dict(min=float(np.nanmin(bnd_top[core])),
                               median=float(np.nanmedian(bnd_top[core])),
                               max=float(np.nanmax(bnd_top[core]))),
            shoulder50_deg=sh50, ceiling_dwell=cdwell, curve=rows, dwell=drows,
        ), indent=1) + "\n")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ctr = [r["elev_lo"] + 2.5 for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.plot(ctr, [r["frac_blob3"] for r in rows], "-o", color="#30638e", label=">=3 ctrl blobs detected")
            ax.plot(ctr, [r["frac_blob1"] for r in rows], "-s", color="#6ca6c1", label=">=1 ctrl blob detected")
            ax.plot(ctr, [r["accept_rate"] for r in rows], "-^", color="#d1495b", label="optical accept within 60ms")
            ax.plot(ctr, [r["tracked"] for r in rows], "--", color="#888", label="POSITION_TRACKED")
            ax.axvline(float(np.nanmedian(bnd_top[core])), color="#1a1a2e", ls="--", lw=1.2)
            if sh50 is not None:
                ax.axvline(sh50, color="#e8a13a", ls=":", lw=1.2)
                ax.text(sh50 + 0.5, 0.75, f"shoulder50\n{sh50:+.1f}", fontsize=8, color="#a06a10")
            ax.text(float(np.nanmedian(bnd_top[core])) + 0.5, 0.9, "geometric top\nboundary", fontsize=8)
            ax.set_xlabel("head-relative elevation (deg)")
            ax.set_ylabel("fraction of samples")
            ax.set_title("Detection / accept / tracked rate vs elevation (both devices pooled)")
            ax.legend(loc="center left")
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(out / "detect_ceiling.png", dpi=130)
        except ImportError:
            print("matplotlib unavailable -- skipped detect_ceiling.png")
        print(f"\nwrote {out}/detect_ceiling.{{csv,json,png}} + dwell_by_elevation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
