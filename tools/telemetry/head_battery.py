#!/usr/bin/env python3
"""head_battery.py -- N8 head-pose quality battery for a G2 capture.

The controller battery had ZERO head coverage until the 2026-07-03 session analysis
(results/session-analysis-20260703/SYNTHESIS.md section 3.1). This tool productizes the
C4 head analysis (c4-head/head_quality.py + event_context.py detectors) and the B1 skew
estimator (b1-slam-skew-20260703/estimator_ci.py, adopted here with its audited
bucketed-envelope clock base, synthetic-validated estimator A and block-bootstrap CIs).

Inputs: <capture>/telemetry (head_pose + imu dev0) and the capture's EuRoC recording
(euroc_*/mav0 with the raw Basalt gt trajectory + imu0), auto-located.

Metrics:
  1. STILL-DRIFT: relaxed instrument-still segments (smoothed |gyro| < 0.10 rad/s,
     >= 1.5 s); per segment linear position wander (mm/s), net drift, rotation rate.
  2. MOVING JITTER (gyro-unexplained, clock-corrected): per-frame |orientation step| minus
     |integrated dev0 gyro| over the same (shift-corrected) interval; percentiles + duty
     over moving frames (warm-up + episodic >=3 deg snaps excluded).
  3. RESNAP CENSUS: IMU-gated pos/ori step discontinuity detectors on head_pose AND
     euroc gt, merged within 60 ms; total + major (>=3 deg or >=20 mm) rates; worst event.
  4. SLAM INPUT/OUTPUT GAP CENSUS: euroc cam0..3 (>40 ms), euroc imu0 (>20 ms), gt output
     (>40 ms), head_pose stream (>50 ms), telemetry dev0 arrival batches (>20 ms).
  5. STAMP-SKEW: estimator A (residual-scan) shift of gt vs euroc imu0 with per-cell
     (brightness x speed) block-bootstrap CIs, plus gt / head_pose vs telemetry dev0 on
     the bucketed mono base.
  6. SLAM CONTROLLER-MASK CENSUS (B5 repair): from the mask telemetry stream — coverage
     duty, per-device masked-area fraction vs the cap, cap/disable rates, sources
     (prediction/last-seen), last-optical age behind the pushed rects. Steal/leak vs
     detected corners live in results/b5-ledmask-20260704/emulate_repair.py (needs the
     euroc corner extraction). Absent stream = pre-B5-repair driver.

BASELINE (capture 20260703-202811-capture-diversity-postflip; reproduces the C4/B1 leg
reports -- results/session-analysis-20260703/c4-head/findings.md, results/b1-slam-skew-20260703/):
  still-drift: one 2.7 s segment (t_rel 126.5-129.2), |v| 3.03 (head_pose) / 2.87 (gt) mm/s,
    net 12.8 mm. C4 published 3.04 / 3.0; on the naive clock base this code reproduces those
    exactly -- the gt delta is one 30 Hz sample at the segment edge shifted by the 15.5 ms
    base correction.
  moving jitter (gt vs telemetry dev0, at measured shift): p50 0.044 / p90 0.167 /
    p99 0.757 deg/frame, duty>0.1deg 20.4% (n=2552 moving frames).
    NOTE: C4's findings.md published p90 0.209 / p99 0.959 on "77% of frames" from a
    script that was not preserved. Reproduced root cause: those numbers come from the
    moving mask WITHOUT the >=3 deg episodic-snap exclusion (and without warm-up trim),
    on the naive clock base at shift -20.75 ms -- that variant gives n=2949 (76%),
    p50 0.049 / p90 0.211 / p99 1.01, duty>0.1 0.249. I.e. the published p90/p99
    double-count episodic snaps that this battery reports separately in the resnap
    census; this tool pins the snap-excluded (continuous-only) numbers, matching B1's
    audited machinery (skew_results.json "c4_reproduction" recorded the same delta).
  resnaps (t_rel > 2 s): 39 merged events (18.1/min), 9 major (4.2/min);
    worst 819 mm + 59.4 deg at t_rel 104.77 s coinciding with the only gt output gap (66.8 ms).
    NOTE: C4's findings.md census (46 / 21.3 per min, 12 major) was computed on the NAIVE
    clock base its own audit later flagged as settling-corrupted (-17.7 ms); rerunning this
    detector on that base reproduces the three extra driver-only fast-swing "majors"
    (34.04 / 103.95 / 121.09 s) -- they are gyro-gate misalignment artifacts, not resnaps.
    All SLAM-level majors (the felt events) are identical between the two bases.
  gaps: euroc cams max dt 35.4 ms (none > 40 ms); imu0 one 184.7 ms start-up gap;
    gt one 66.8 ms gap; telemetry dev0 one 208 ms start-up batch gap.
  skew: gt vs euroc-imu0 FULL -3.9 ms (boot ci95 [-6.3, -2.1]); cells dark x 6-30 dps
    -6.9, dark x 30-120 -4.7, bright x 6-30 -0.0, bright x 30-120 -4.3 ms;
    gt vs telemetry-dev0 -6.2 ms; head_pose vs telemetry-dev0 -1.7 ms.
    (C4's first-pass values were head_pose -3.5 / gt -5.5 ms -- same "SLAM stamped late"
    verdict, superseded by the B1 estimator adopted here.)

Usage: head_battery.py <capture_dir> [--euroc MAV0] [--json OUT] [--boot N] [--seed S]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest  # noqa: E402
import g2_geom as G  # noqa: E402
from headpose_anchor import load_head_pose  # noqa: E402

SHIFTS_MS = np.arange(-30.0, 30.001, 0.25)
BOOT_SHIFTS_MS = np.arange(-15.0, 15.001, 0.5)
MOVING_TH = 0.10       # rad/s smoothed |gyro| -> "moving" frame
STILL_TH = 0.10        # rad/s smoothed |gyro| -> relaxed instrument-still
SNAP_EXCL_DEG = 3.0    # exclude episodic snaps from the continuous-jitter metric
WARMUP_S = 2.0
MERGE_S = 0.06         # resnap merge window across streams
MAJOR_ORI_DEG = 3.0
MAJOR_POS_MM = 20.0
# brightness x speed cells (estimator_ci.py operating points)
DARK_LT = 12.0
BRIGHT_GE = 18.0
SPEED_EDGES_DPS = (6.0, 30.0, 120.0)
MIN_CELL_FRAMES = 120


# ---------------------------------------------------------------- loading

def find_euroc(capture: Path) -> Path | None:
    """Locate the session's EuRoC mav0 dir. G2_RECORD is a path PREFIX (upstream appends
    _YYYYMMDDHHMMSS per SLAM-tracker start); multiple dirs = multiple launches, keep the
    one with the most gt rows (adjudicates a discarded first launch)."""
    best, best_rows = None, -1
    for euroc in sorted(capture.glob("euroc*")):
        gt = euroc / "mav0" / "gt" / "data.csv"
        if not gt.is_file():
            continue
        rows = sum(1 for _ in gt.open()) - 1
        if rows > best_rows:
            best, best_rows = euroc / "mav0", rows
    return best


def load_pose_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    t = d[:, 0] / 1e9
    p = d[:, 1:4]
    q = d[:, [5, 6, 7, 4]]  # csv is w,x,y,z -> x,y,z,w
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], p[keep], q[keep]


def load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    t = d[:, 0] / 1e9
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], d[keep, 1:4]  # gyro rad/s


def dev0_mono(imu: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Telemetry dev0 gyro on the mono clock via the BUCKETED min envelope (per-10s-bucket
    minima of t_mono - hw, settled buckets only). The naive global min is corrupted by a
    ~-17.7 ms settling transient in the first bucket (C4 clock-alignment audit)."""
    i0 = imu[imu["device_id"] == 0]
    i0 = i0[np.argsort(i0["hw_ts_ns"], kind="stable")]
    diff = i0["t_mono_ns"].astype(np.float64) - i0["hw_ts_ns"].astype(np.float64)
    tb = i0["t_mono_ns"].astype(np.float64) / 1e9
    edges = np.arange(tb[0], tb[-1] + 10.0, 10.0)
    mins = [diff[(tb >= a) & (tb < b)].min()
            for a, b in zip(edges[:-1], edges[1:]) if ((tb >= a) & (tb < b)).sum() > 100]
    med = np.median(mins)
    off = float(np.median([v for v in mins if abs(v - med) < 2e6]))
    t = i0["hw_ts_ns"].astype(np.float64) / 1e9 + off / 1e9
    gyr = np.stack([i0["gx"], i0["gy"], i0["gz"]], 1).astype(np.float64)
    return t, gyr, off / 1e9


def brightness_series(mav0: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-frame mean pixel of euroc cam0 (2x-downsampled, matching flow_skew.py), at
    frame-interval midpoints. Needs Pillow; returns None (speed-only cells) without it."""
    try:
        from PIL import Image
    except ImportError:
        return None
    csv = mav0 / "cam0" / "data.csv"
    if not csv.is_file():
        return None
    rows = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=str)
    t = rows[:, 0].astype(np.int64) / 1e9
    bright = np.empty(len(rows))
    for i, name in enumerate(rows[:, 1]):
        bright[i] = np.asarray(Image.open(mav0 / "cam0" / "data" / name), dtype=np.float32)[::2, ::2].mean()
    # value at interval midpoint i is the SECOND frame's mean (flow_skew.py convention,
    # kept so the estimator_ci.py cell memberships reproduce exactly)
    mid = (t[:-1] + t[1:]) / 2
    return mid, bright[1:]


# ---------------------------------------------------------------- core math (B1 estimator)

def smooth_gyro_norm(t_i: np.ndarray, gyr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gn = np.linalg.norm(gyr, axis=1)
    fs = 1.0 / np.median(np.diff(t_i))
    win = max(1, int(0.25 * fs))
    return np.convolve(gn, np.ones(win) / win, mode="same"), gn


def pose_steps(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-interval orientation step magnitude (deg), dt (s), midpoints (s)."""
    dang = G.quat_geodesic_deg(q[1:], q[:-1])
    dt = np.diff(t)
    mid = (t[:-1] + t[1:]) / 2
    return dang, dt, mid


def frame_mask(dang: np.ndarray, dt: np.ndarray, mid: np.ndarray,
               gn_s_at: Callable[[np.ndarray], np.ndarray], t0: float) -> np.ndarray:
    ok = (dt > 1e-4) & (dt < 0.3)
    ok &= mid - t0 > WARMUP_S
    ok &= dang < SNAP_EXCL_DEG
    return ok & (gn_s_at(mid) >= MOVING_TH)


def gyro_cumangle(t_i: np.ndarray, gn: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(0.5 * (gn[1:] + gn[:-1]) * np.diff(t_i))])
    return c * 180.0 / np.pi


def gint_windows(t_i: np.ndarray, cum: np.ndarray, ta: np.ndarray, tb: np.ndarray) -> np.ndarray:
    return np.interp(tb, t_i, cum) - np.interp(ta, t_i, cum)


def residual_scan(t: np.ndarray, dang: np.ndarray, mask: np.ndarray,
                  t_i: np.ndarray, cum: np.ndarray, shifts_ms: np.ndarray = SHIFTS_MS) -> np.ndarray:
    ta, tb, da = t[:-1][mask], t[1:][mask], dang[mask]
    curve = np.empty(len(shifts_ms))
    for k, s in enumerate(shifts_ms):
        curve[k] = np.median(np.abs(da - gint_windows(t_i, cum, ta + s / 1e3, tb + s / 1e3)))
    return curve


def parabola_min(shifts_ms: np.ndarray, curve: np.ndarray, halfwin: int = 8) -> tuple[float, float]:
    k = int(np.argmin(curve))
    a, b = max(0, k - halfwin), min(len(curve), k + halfwin + 1)
    c = np.polyfit(shifts_ms[a:b], curve[a:b], 2)
    if c[0] <= 0:
        return float(shifts_ms[k]), float(curve[k])
    s = -c[1] / (2 * c[0])
    return float(s), float(np.polyval(c, s))


def boot_ci(t: np.ndarray, dang: np.ndarray, mid: np.ndarray, mask: np.ndarray,
            t_i: np.ndarray, cum: np.ndarray, n: int, block_s: float, seed: int) -> dict[str, Any]:
    """Per-cell 2 s block bootstrap of estimator A (estimator_ci.py parameters)."""
    rng = np.random.default_rng(seed)
    tm = mid[mask]
    blocks = np.floor((tm - tm[0]) / block_s).astype(int)
    uniq = np.unique(blocks)
    idx_all = np.arange(mask.sum())
    ta_m, tb_m, da_m = t[:-1][mask], t[1:][mask], dang[mask]
    ests = []
    for _ in range(n):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_all[blocks == b] for b in chosen])
        curve = np.empty(len(BOOT_SHIFTS_MS))
        for k, s in enumerate(BOOT_SHIFTS_MS):
            g = gint_windows(t_i, cum, ta_m[sel] + s / 1e3, tb_m[sel] + s / 1e3)
            curve[k] = np.median(np.abs(da_m[sel] - g))
        ests.append(parabola_min(BOOT_SHIFTS_MS, curve, halfwin=6)[0])
    e = np.array(ests)
    return dict(mean=round(float(e.mean()), 3), se=round(float(e.std()), 3),
                ci95=[round(float(np.percentile(e, 2.5)), 3),
                      round(float(np.percentile(e, 97.5)), 3)])


def jitter_stats(t: np.ndarray, dang: np.ndarray, mask: np.ndarray,
                 t_i: np.ndarray, cum: np.ndarray, shift_ms: float) -> dict[str, Any]:
    ta, tb = t[:-1][mask], t[1:][mask]
    resd = np.abs(dang[mask] - gint_windows(t_i, cum, ta + shift_ms / 1e3, tb + shift_ms / 1e3))
    return dict(
        percentiles_deg={str(p): round(float(np.percentile(resd, p)), 4) for p in (50, 75, 90, 95, 99)},
        duty={str(th): round(float((resd > th).mean()), 4) for th in (0.1, 0.2, 0.3)},
        n=int(mask.sum()),
    )


# ---------------------------------------------------------------- still segments + drift

def still_segments(t_i: np.ndarray, gn_s: np.ndarray, th: float = STILL_TH,
                   min_s: float = 1.5, trim_s: float = 0.2) -> list[tuple[float, float]]:
    raw = gn_s < th
    idx = np.flatnonzero(np.diff(np.concatenate([[0], raw.view(np.int8), [0]])))
    segs = []
    for a, b in zip(idx[::2], idx[1::2]):
        ta, tb = t_i[a] + trim_s, t_i[min(b, len(t_i) - 1)] - trim_s
        if tb - ta >= min_s:
            segs.append((float(ta), float(tb)))
    return segs


def _sign_fix(q: np.ndarray) -> np.ndarray:
    qs = q.copy()
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[i - 1]) < 0:
            qs[i] = -qs[i]
    return qs


def still_drift(segs: list[tuple[float, float]], t0: float,
                streams: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for k, (a, b) in enumerate(segs):
        for name, (t, p, q) in streams.items():
            msk = (t >= a) & (t <= b)
            if msk.sum() < 15:
                continue
            ts, ps = t[msk], p[msk]
            qs = _sign_fix(q[msk])
            rv = np.degrees(G.quat_log(qs))
            tt = ts - ts[0]
            vel = np.polyfit(tt, ps, 1)[0] * 1000.0        # mm/s per axis
            rrate = np.polyfit(tt, rv, 1)[0]               # deg/s per axis
            det = ps - (ps[0] + np.outer(tt, np.polyfit(tt, ps, 1)[0]))
            rows.append(dict(
                seg=k, t0_rel=round(a - t0, 1), t1_rel=round(b - t0, 1), dur_s=round(b - a, 1),
                stream=name,
                vel_mm_s=round(float(np.linalg.norm(vel)), 2),
                net_mm=round(float(np.linalg.norm(ps[-1] - ps[0]) * 1000.0), 1),
                rot_deg_s=round(float(np.linalg.norm(rrate)), 3),
                net_deg=round(float(G.quat_geodesic_deg(qs[0], qs[-1])), 2),
                ptp_mm=round(float(np.linalg.norm(det.max(0) - det.min(0)) * 1000.0), 1),
            ))
    return rows


# ---------------------------------------------------------------- resnap census

def detect_resnaps(t: np.ndarray, p: np.ndarray, q: np.ndarray,
                   t_i: np.ndarray, gn_s: np.ndarray, cum: np.ndarray,
                   pos_floor_mm: float, ori_floor_deg: float) -> list[dict[str, Any]]:
    """C4's IMU-gated step detectors: an event is an orientation step exceeding
    1.5x the integrated gyro + floor, or a position step exceeding IMU-plausible motion
    (2 mm while gyro-still, else 2.5 m/s * dt + 5 mm)."""
    dang = G.quat_geodesic_deg(_sign_fix(q)[1:], _sign_fix(q)[:-1])
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1) * 1000.0
    dt = np.diff(t)
    g_ang = gint_windows(t_i, cum, t[:-1], t[1:])
    still = np.interp(t[:-1], t_i, gn_s) < 0.06
    events = []
    for i in range(len(dt)):
        if dt[i] <= 0 or dt[i] > 0.3:
            continue
        pos_allow = 2.0 if still[i] else 2500.0 * dt[i] + 5.0
        ori_excess = dang[i] - (g_ang[i] * 1.5 + ori_floor_deg)
        pos_excess = dp[i] - max(pos_allow, pos_floor_mm)
        if ori_excess > 0 or pos_excess > 0:
            events.append(dict(t=float(t[i + 1]), pos_step_mm=float(dp[i]),
                               ori_step_deg=float(dang[i]), gyro_deg=float(g_ang[i])))
    return events


def merge_events(per_stream: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    allev = sorted(((s, e) for s, evs in per_stream.items() for e in evs), key=lambda x: x[1]["t"])
    merged: list[dict[str, Any]] = []
    for stream, e in allev:
        if merged and e["t"] - merged[-1]["t"] < MERGE_S:
            m = merged[-1]
            m["streams"].add(stream)
            m["pos_step_mm"] = max(m["pos_step_mm"], e["pos_step_mm"])
            m["ori_step_deg"] = max(m["ori_step_deg"], e["ori_step_deg"])
        else:
            merged.append(dict(t=e["t"], streams={stream}, pos_step_mm=e["pos_step_mm"],
                               ori_step_deg=e["ori_step_deg"], gyro_deg=e["gyro_deg"]))
    return merged


# ---------------------------------------------------------------- gap census

def _gaps(t_s: np.ndarray, th_ms: float) -> dict[str, Any]:
    d = np.diff(np.sort(t_s)) * 1000.0
    idx = np.flatnonzero(d > th_ms)
    return dict(threshold_ms=th_ms, n=int(idx.size),
                gaps=[dict(t_s=round(float(t_s[i + 1]), 3), gap_ms=round(float(d[i]), 1)) for i in idx[:20]],
                max_dt_ms=round(float(d.max()), 1) if d.size else None)


# ---------------------------------------------------------------- battery

def run_battery(capture: Path, euroc: Path | None, boot_n: int, seed: int) -> dict[str, Any]:
    telem = capture / "telemetry" if (capture / "telemetry" / "manifest.json").is_file() else capture
    m = Manifest.load(telem)
    imu = G.load_stream(telem, m, "imu")
    hp = load_head_pose(telem, m)
    if hp is None:
        raise SystemExit("no usable head_pose stream")
    t_hp = hp.t_ns.astype(np.float64) / 1e9
    keep = np.concatenate([[True], np.diff(t_hp) > 0])
    t_hp, p_hp, q_hp = t_hp[keep], hp.pos[keep], hp.quat[keep]

    mav0 = euroc if euroc is not None else find_euroc(capture)
    if mav0 is None:
        raise SystemExit(f"no euroc_*/mav0 gt under {capture} (pass --euroc)")
    t_gt, p_gt, q_gt = load_pose_csv(mav0 / "gt" / "data.csv")
    t_i0, gyr0 = load_imu_csv(mav0 / "imu0" / "data.csv")

    t_d0, gyr_d0, d0_off = dev0_mono(imu)
    gn_s_d0, gn_d0 = smooth_gyro_norm(t_d0, gyr_d0)
    cum_d0 = gyro_cumangle(t_d0, gn_d0)
    gn_at_d0 = lambda tt: np.interp(tt, t_d0, gn_s_d0)  # noqa: E731

    T0 = float(t_hp[0])
    span_s = float(t_gt[-1] - t_gt[0])
    report: dict[str, Any] = dict(
        capture=str(capture), euroc=str(mav0), t0_mono_s=round(T0, 3),
        span_s=round(span_s, 1), dev0_mono_offset_s=round(d0_off, 6),
        head_pose_rows=int(t_hp.size), gt_rows=int(t_gt.size),
    )

    # 1. still drift
    segs = still_segments(t_d0, gn_s_d0)
    report["still_drift"] = dict(
        segments=[dict(t0_rel=round(a - T0, 1), t1_rel=round(b - T0, 1), dur_s=round(b - a, 1))
                  for a, b in segs],
        total_still_s=round(sum(b - a for a, b in segs), 1),
        rows=still_drift(segs, T0, {"head_pose": (t_hp, p_hp, q_hp), "euroc_gt": (t_gt, p_gt, q_gt)}),
    )

    # 2+5. moving jitter + stamp skew vs telemetry dev0 (both pose streams)
    jitter: dict[str, Any] = {}
    for name, (t_p, q_p) in (("euroc_gt", (t_gt, q_gt)), ("head_pose", (t_hp, q_hp))):
        dang, dt, mid = pose_steps(t_p, _sign_fix(q_p))
        mask = frame_mask(dang, dt, mid, gn_at_d0, float(t_gt[0]))
        curve = residual_scan(t_p, dang, mask, t_d0, cum_d0)
        shift, min_med = parabola_min(SHIFTS_MS, curve)
        moving_frac = float(mask.sum() / max(1, len(dang)))
        jitter[name] = dict(
            shift_vs_telemetry_dev0_ms=round(shift, 3),
            min_med_deg=round(min_med, 4),
            moving_frame_frac=round(moving_frac, 3),
            at_zero_shift=jitter_stats(t_p, dang, mask, t_d0, cum_d0, 0.0),
            at_measured_shift=jitter_stats(t_p, dang, mask, t_d0, cum_d0, shift),
        )
    report["moving_jitter"] = jitter

    # 3. resnap census
    per_stream = {
        "head_pose": detect_resnaps(t_hp, p_hp, q_hp, t_d0, gn_s_d0, cum_d0, 8.0, 0.35),
        "euroc_gt": detect_resnaps(t_gt, p_gt, q_gt, t_d0, gn_s_d0, cum_d0, 8.0, 0.5),
    }
    merged = [e for e in merge_events(per_stream) if e["t"] - T0 > WARMUP_S]
    major = [e for e in merged if e["ori_step_deg"] >= MAJOR_ORI_DEG or e["pos_step_mm"] >= MAJOR_POS_MM]
    worst = max(merged, key=lambda e: e["pos_step_mm"], default=None)
    report["resnaps"] = dict(
        raw={k: len(v) for k, v in per_stream.items()},
        merged=len(merged), merged_per_min=round(len(merged) / span_s * 60.0, 1),
        major=len(major), major_per_min=round(len(major) / span_s * 60.0, 1),
        major_events=[dict(t_rel_s=round(e["t"] - T0, 2), level=("SLAM" if "euroc_gt" in e["streams"] else "driver-only"),
                           pos_step_mm=round(e["pos_step_mm"], 1), ori_step_deg=round(e["ori_step_deg"], 2),
                           gyro_integrated_deg=round(e["gyro_deg"], 2)) for e in major],
        worst=(dict(t_rel_s=round(worst["t"] - T0, 2), pos_step_mm=round(worst["pos_step_mm"], 1),
                    ori_step_deg=round(worst["ori_step_deg"], 2)) if worst else None),
    )

    # 4. SLAM input/output gap census
    gaps: dict[str, Any] = {}
    for c in range(4):
        csv = mav0 / f"cam{c}" / "data.csv"
        if csv.is_file():
            tc = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=0) / 1e9
            gaps[f"euroc_cam{c}"] = _gaps(tc, 40.0)
    gaps["euroc_imu0"] = _gaps(t_i0, 20.0)
    gaps["euroc_gt_output"] = _gaps(t_gt, 40.0)
    gaps["head_pose_stream"] = _gaps(t_hp, 50.0)
    i0 = imu[imu["device_id"] == 0]
    gaps["telemetry_dev0_batches"] = _gaps(np.unique(i0["t_mono_ns"]).astype(np.float64) / 1e9, 20.0)
    report["input_gaps"] = gaps

    # 5. per-cell skew estimate vs euroc imu0 (Basalt-internal seam) with bootstrap CIs
    gn_s_i0, gn_i0 = smooth_gyro_norm(t_i0, gyr0)
    cum_i0 = gyro_cumangle(t_i0, gn_i0)
    gn_at_i0 = lambda tt: np.interp(tt, t_i0, gn_s_i0)  # noqa: E731
    dang, dt, mid = pose_steps(t_gt, _sign_fix(q_gt))
    mask = frame_mask(dang, dt, mid, gn_at_i0, float(t_gt[0]))
    spd = np.degrees(gn_at_i0(mid))
    lo, mi, hi = SPEED_EDGES_DPS
    cells: dict[str, np.ndarray] = {"FULL": np.ones(len(mid), bool)}
    bright = brightness_series(mav0)
    if bright is not None:
        br = np.interp(mid, bright[0], bright[1])
        cells.update({
            f"dark x {lo:.0f}-{mi:.0f}dps": (br < DARK_LT) & (spd < mi),
            f"dark x {mi:.0f}-{hi:.0f}dps": (br < DARK_LT) & (spd >= mi) & (spd < hi),
            f"bright x {lo:.0f}-{mi:.0f}dps": (br >= BRIGHT_GE) & (spd < mi),
            f"bright x {mi:.0f}-{hi:.0f}dps": (br >= BRIGHT_GE) & (spd >= mi) & (spd < hi),
        })
    else:
        cells.update({f"spd {lo:.0f}-{mi:.0f}dps": spd < mi,
                      f"spd {mi:.0f}-{hi:.0f}dps": (spd >= mi) & (spd < hi)})
    skew: dict[str, Any] = {}
    for name, cm in cells.items():
        sel = mask & cm
        if sel.sum() < MIN_CELL_FRAMES:
            skew[name] = dict(n=int(sel.sum()), shift_ms=None)
            continue
        # estimator_ci.py's exact scan grid, so the point estimates reproduce alongside the CIs
        curve = residual_scan(t_gt, dang, sel, t_i0, cum_i0, shifts_ms=BOOT_SHIFTS_MS)
        s, v = parabola_min(BOOT_SHIFTS_MS, curve, halfwin=6)
        skew[name] = dict(n=int(sel.sum()), shift_ms=round(s, 3), min_med_deg=round(v, 4),
                          boot=boot_ci(t_gt, dang, mid, sel, t_i0, cum_i0, boot_n, 2.0, seed))
    report["skew_gt_vs_euroc_imu0"] = skew

    # 6. SLAM controller-mask census (B5 repair observability; mask stream ships with the
    # frame-cadence push). Rect staleness is <= one controller-frame period by construction;
    # the standing regression signals here are coverage duty, per-device area vs the cap,
    # cap/disable rates, and the last-optical age distribution behind the pushed rects.
    # Steal/leak against detected corners stay with the b5-ledmask emulation pipeline
    # (results/b5-ledmask-20260704/emulate_repair.py) — they need the euroc corner extraction.
    if "mask" in m.streams:
        mk = G.load_stream(telem, m, "mask")
        if mk.size:
            ENABLED, PRED, LAST, CAPPED = 1, 2, 4, 8
            en = (mk["flags"] & ENABLED) != 0
            cam01 = mk["cam_id"] < 2
            per_dev: dict[str, Any] = {}
            for dev in np.unique(mk["device_id"]):
                r = mk[mk["device_id"] == dev]
                ren = (r["flags"] & ENABLED) != 0
                area = np.where(ren, (r["x1"] - r["x0"]) * (r["y1"] - r["y0"]) / (640.0 * 480.0), 0.0)
                age = r["optical_age_ms"][r["optical_age_ms"] >= 0]
                per_dev[f"dev{dev}"] = dict(
                    rows=int(len(r)),
                    enabled_duty=round(float(ren.mean()), 4),
                    capped_frac=round(float(((r["flags"] & CAPPED) != 0).mean()), 5),
                    pred_source_frac=round(float(((r["flags"] & PRED) != 0).mean()), 4),
                    last_seen_source_frac=round(float(((r["flags"] & LAST) != 0).mean()), 4),
                    area_frac_p50_p99_max=[round(float(np.percentile(area[ren], q)), 4)
                                           for q in (50, 99, 100)] if ren.any() else None,
                    optical_age_ms_p50_p90_max=[round(float(np.percentile(age, q)), 1)
                                                for q in (50, 90, 100)] if age.size else None,
                )
            report["controller_mask"] = dict(
                rows=int(len(mk)), enabled_duty_cam01=round(float(en[cam01].mean()), 4),
                per_device=per_dev)
        else:
            report["controller_mask"] = dict(rows=0, note="mask stream present but empty")
    else:
        report["controller_mask"] = dict(note="mask stream absent (pre-B5-repair driver)")
    return report


# ---------------------------------------------------------------- output

def print_summary(r: dict[str, Any]) -> None:
    print(f"# head battery: {r['capture']}")
    print(f"euroc: {r['euroc']}  span={r['span_s']}s  head_pose={r['head_pose_rows']} rows  "
          f"gt={r['gt_rows']} rows  dev0 mono base={r['dev0_mono_offset_s'] * 1e3:.3f} ms")

    sd = r["still_drift"]
    print(f"\n[1] STILL-DRIFT  (relaxed still: smoothed |gyro|<{STILL_TH} rad/s, >=1.5 s; "
          f"{len(sd['segments'])} segments, {sd['total_still_s']} s total)")
    for row in sd["rows"]:
        print(f"  seg{row['seg']} [{row['t0_rel']:7.1f}-{row['t1_rel']:7.1f}s {row['dur_s']:4.1f}s] "
              f"{row['stream']:10s} |v|={row['vel_mm_s']:6.2f} mm/s net={row['net_mm']:6.1f} mm "
              f"|w|={row['rot_deg_s']:6.3f} deg/s net={row['net_deg']:5.2f} deg ptp={row['ptp_mm']:5.1f} mm")

    print("\n[2] MOVING GYRO-UNEXPLAINED JITTER (deg/frame; warm-up + >=3deg snaps excluded; "
          "at the measured stamp shift)")
    for name, j in r["moving_jitter"].items():
        pc = j["at_measured_shift"]["percentiles_deg"]
        duty = j["at_measured_shift"]["duty"]
        print(f"  {name:10s} shift={j['shift_vs_telemetry_dev0_ms']:+7.3f} ms  "
              f"moving={j['moving_frame_frac'] * 100:.0f}% of frames (n={j['at_measured_shift']['n']})  "
              f"p50={pc['50']:.3f} p90={pc['90']:.3f} p99={pc['99']:.3f}  "
              f"duty>0.1/0.2/0.3deg={duty['0.1']:.3f}/{duty['0.2']:.3f}/{duty['0.3']:.3f}")

    rs = r["resnaps"]
    print(f"\n[3] RESNAPS (post-warm-up, streams merged within {int(MERGE_S * 1000)} ms): "
          f"{rs['merged']} events = {rs['merged_per_min']}/min; "
          f"{rs['major']} MAJOR (>={MAJOR_ORI_DEG:.0f}deg or >={MAJOR_POS_MM:.0f}mm) = {rs['major_per_min']}/min")
    for e in rs["major_events"]:
        print(f"  t+{e['t_rel_s']:7.2f}s {e['level']:<12s} pos={e['pos_step_mm']:7.1f} mm "
              f"ori={e['ori_step_deg']:6.2f} deg (gyro integral {e['gyro_integrated_deg']:.2f} deg)")
    if rs["worst"]:
        w = rs["worst"]
        print(f"  worst: {w['pos_step_mm']} mm + {w['ori_step_deg']} deg at t+{w['t_rel_s']}s")

    print("\n[4] SLAM INPUT/OUTPUT GAPS")
    for name, g in r["input_gaps"].items():
        detail = "; ".join(f"t={x['t_s']:.1f}s {x['gap_ms']}ms" for x in g["gaps"]) or "none"
        print(f"  {name:22s} >{g['threshold_ms']:.0f}ms: {g['n']:3d}  (max dt {g['max_dt_ms']} ms)  {detail}")

    print("\n[5] SLAM STAMP SKEW, gt vs euroc imu0 (estimator A + block-bootstrap CI); "
          "negative = pose stamped LATE vs IMU")
    for name, c in r["skew_gt_vs_euroc_imu0"].items():
        if c["shift_ms"] is None:
            print(f"  {name:22s} n={c['n']:4d}  (too few frames)")
            continue
        b = c["boot"]
        print(f"  {name:22s} n={c['n']:4d}  shift={c['shift_ms']:+7.3f} ms  "
              f"boot mean={b['mean']:+.3f} se={b['se']:.3f} ci95=[{b['ci95'][0]:+.2f}, {b['ci95'][1]:+.2f}]")
    for name in ("euroc_gt", "head_pose"):
        j = r["moving_jitter"][name]
        print(f"  {name} vs telemetry dev0 (bucketed base): shift={j['shift_vs_telemetry_dev0_ms']:+.3f} ms")

    cm = r.get("controller_mask", {})
    print("\n[6] SLAM CONTROLLER-MASK CENSUS (B5 repair: frame-cadence prediction+last-seen push)")
    if "per_device" not in cm:
        print(f"  {cm.get('note', 'no data')}")
    else:
        print(f"  rows={cm['rows']}  enabled duty cam0/1={cm['enabled_duty_cam01']:.3f}")
        for dev, d in cm["per_device"].items():
            print(f"  {dev}: duty={d['enabled_duty']:.3f} capped={d['capped_frac']:.4f} "
                  f"src pred/last={d['pred_source_frac']:.3f}/{d['last_seen_source_frac']:.3f} "
                  f"area p50/p99/max={d['area_frac_p50_p99_max']} "
                  f"optical age ms p50/p90/max={d['optical_age_ms_p50_p90_max']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path, help="capture dir (with telemetry/ and euroc_*/)")
    ap.add_argument("--euroc", type=Path, help="explicit euroc mav0 dir (default: auto-locate)")
    ap.add_argument("--json", type=Path, help="write the full JSON report")
    ap.add_argument("--boot", type=int, default=120, help="bootstrap replicates per skew cell")
    ap.add_argument("--seed", type=int, default=11, help="bootstrap RNG seed")
    args = ap.parse_args()

    report = run_battery(args.capture, args.euroc, args.boot, args.seed)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1) + "\n")
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
