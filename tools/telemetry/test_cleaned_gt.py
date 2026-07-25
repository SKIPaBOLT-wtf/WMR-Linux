#!/usr/bin/env python3
"""test_cleaned_gt.py -- decoupled + adversarial tests for the cleaned-GT / MSE tooling.

Tests assert BEHAVIOUR / CONTRACTS, not internals, and deliberately exercise degenerate
and adversarial inputs (synthetic flips, gravity-blind runs, speed fly-aways, ambiguous
yaw flips) -- not just the ideal path. Run with the conda env:
    ~/miniconda3/envs/g2vr/bin/python test_cleaned_gt.py
Exits non-zero on any failure. No production code, no telemetry files required.
"""
from __future__ import annotations

import sys

import numpy as np

import g2_geom as G
from deflip import deflip_device, Q_GOOD, Q_OK, Q_CORRECTED, Q_UNCERTAIN, TILT_TOL_DEG

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


def synth_track(n, dt_ms=22.0, yaw_rate_deg=10.0):
    """A clean slowly-yawing trajectory, upright (gravity-consistent). Returns
    t_ns, pos, quat, g_body (body gravity == down rotated into body)."""
    t = (np.arange(n) * dt_ms * 1e6).astype(np.int64)
    pos = np.stack([0.01 * np.arange(n), np.zeros(n), np.zeros(n)], axis=1)
    quat = np.zeros((n, 4))
    g_body = np.zeros((n, 3))
    for i in range(n):
        yaw = np.deg2rad(yaw_rate_deg) * i * dt_ms / 1000.0
        q = G.quat_exp([0, yaw, 0])  # yaw about world-up
        quat[i] = q
        # body gravity = R^-1 * down (driftless cue ground truth)
        g_body[i] = G.quat_rotate_inv(q, G.DOWN)
    return t, pos, quat, g_body


def mirror_tilt_flip(q):
    """Apply a 180-deg tilt flip (about a horizontal body axis) -> gravity-inconsistent."""
    return G.quat_mul(q, G.quat_exp([np.pi, 0, 0]))


def consec_flip_rate(t, quat, flip_deg=90.0, gap_ms=120.0):
    n = f = 0
    for i in range(1, len(t)):
        if (t[i] - t[i - 1]) / 1e6 <= gap_ms:
            n += 1
            if G.quat_geodesic_deg(quat[i], quat[i - 1]) >= flip_deg:
                f += 1
    return f, n


def main():
    np.random.seed(0)
    conf_hi = lambda n: (np.full(n, 8), np.full(n, 0.8))  # many LEDs, low reproj

    # ---- 1: clean track has zero flips, all GOOD/OK, smoother is a no-op-ish ----
    print("[1] clean track -> no spurious flips")
    n = 60
    t, pos, quat, gb = synth_track(n)
    mb, rp = conf_hi(n)
    res = deflip_device(t, pos, quat, gb, mb, rp)
    check("no flips detected on clean data", res.stats["n_flip"] == 0, str(res.stats))
    check("no CORRECTED/UNCERTAIN on clean data",
          (res.quality >= Q_OK).all(), str(np.unique(res.quality, return_counts=True)))
    check("cleaned == raw on clean data",
          np.allclose(G.quat_geodesic_deg(res.quat, res.quat_raw), 0, atol=1e-6))

    # ---- 2: single isolated TILT flip is detected, corrected, gravity-confirmed ----
    print("[2] single gravity-inconsistent tilt flip -> CORRECTED")
    t, pos, quat, gb = synth_track(n)
    k = 30
    quat[k] = mirror_tilt_flip(quat[k])  # g_body[k] still the TRUE body gravity (driftless)
    mb, rp = conf_hi(n)
    res = deflip_device(t, pos, quat, gb, mb, rp)
    check("the flip index is flagged", k in res.flip_idx.tolist(), str(res.flip_idx))
    check("the flip is CORRECTED (gravity-confirmed)", res.quality[k] == Q_CORRECTED,
          f"quality={res.quality[k]}")
    # corrected orientation should be gravity-consistent again
    tc = G.gravity_tilt_err_deg(res.quat[k][None, :], gb[k][None, :])[0]
    check("corrected pose is gravity-consistent (tilt <= TILT_TOL)", tc <= TILT_TOL_DEG + 5,
          f"tilt={tc:.1f}")
    fr_raw = consec_flip_rate(t, quat)[0]
    fr_new = consec_flip_rate(t, res.quat)[0]
    check("residual consecutive flips reduced", fr_new < fr_raw, f"{fr_raw}->{fr_new}")

    # ---- 3: a RUN of tilt flips (sustained wrong branch) -> all corrected ----
    print("[3] sustained tilt-flip run -> filled inward from both ends")
    t, pos, quat, gb = synth_track(n)
    for k in range(25, 33):
        quat[k] = mirror_tilt_flip(quat[k])
    mb, rp = conf_hi(n)
    res = deflip_device(t, pos, quat, gb, mb, rp)
    fr_new, ncons = consec_flip_rate(t, res.quat)
    check("run of flips mostly removed (<=1 residual)", fr_new <= 1, f"residual={fr_new}/{ncons}")
    tc = G.gravity_tilt_err_deg(res.quat[25:33], gb[25:33])
    check("run corrected to gravity-consistent (median tilt small)",
          np.median(tc) <= TILT_TOL_DEG + 5, f"median tilt={np.median(tc):.1f}")

    # ---- 4: a PURE-YAW flip with gravity blind -> UNCERTAIN, not silently accepted ----
    print("[4] gravity-blind yaw flip -> UNCERTAIN (honest abstention)")
    t, pos, quat, gb = synth_track(n)
    k = 30
    quat[k] = G.quat_mul(quat[k], G.quat_exp([0, np.pi, 0]))  # 180 yaw: gravity CANNOT see it
    gb_blind = gb.copy()
    gb_blind[k] = np.nan  # controller accelerating -> gravity unobservable at this frame
    mb, rp = conf_hi(n)
    res = deflip_device(t, pos, quat, gb_blind, mb, rp)
    check("yaw flip is detected as a flip", k in res.flip_idx.tolist(), str(res.flip_idx))
    check("yaw flip with no gravity is UNCERTAIN (not CORRECTED-confident)",
          res.quality[k] == Q_UNCERTAIN, f"quality={res.quality[k]}")

    # ---- 5: no gravity available ANYWHERE -> all UNCERTAIN, never crash ----
    print("[5] gravity-blind everywhere -> degrades gracefully")
    t, pos, quat, gb = synth_track(n)
    gb_all_nan = np.full_like(gb, np.nan)
    res = deflip_device(t, pos, quat, gb_all_nan, *conf_hi(n))
    check("no crash, returns N quality flags", res.quality.shape[0] == n)
    check("all flagged UNCERTAIN when no gravity reference",
          (res.quality == Q_UNCERTAIN).all() or res.stats["n_grav_consistent"] == 0,
          str(np.unique(res.quality, return_counts=True)))

    # ---- 6: speed gate rejects a physically-impossible position fly-away ----
    print("[6] speed gate rejects a fly-away (smoother)")
    from smooth_ref import _speed_gate
    t, pos, quat, gb = synth_track(n)
    pos[30] = pos[30] + np.array([50.0, 0, 0])  # 50 m jump in 22 ms -> ~2270 m/s
    keep = _speed_gate(t, pos, quat, np.full(n, Q_OK))
    check("the fly-away pose is rejected", not keep[30])
    check("neighbours are kept", keep[29] and keep[31])

    # ---- 7: backwards / duplicate timestamps don't crash the speed gate ----
    print("[7] backwards + duplicate timestamps handled")
    t, pos, quat, gb = synth_track(10)
    t[5] = t[4]            # duplicate
    t[7] = t[6] - 1000     # backwards
    keep = _speed_gate(t, pos, quat, np.full(10, Q_OK))
    check("duplicate timestamp dropped", not keep[5])
    check("backwards timestamp dropped", not keep[7])

    # ---- 8: tiny input (N<3) builds nothing rather than crashing ----
    print("[8] degenerate tiny input")
    t2 = np.array([0, 22_000_000], np.int64)
    res = deflip_device(t2, np.zeros((2, 3)), np.tile([0, 0, 0, 1.0], (2, 1)),
                        np.tile(G.DOWN, (2, 1)), np.full(2, 8), np.full(2, 0.8))
    check("2-pose input returns 2 quality flags (no crash)", res.quality.shape[0] == 2)

    # ===================================================================
    # C1: the INDEPENDENT head-pose orientation anchor (headpose_anchor.py)
    # ===================================================================
    import headpose_anchor as HA

    def head_track(n, dt_ms=22.0):
        """A non-drifting, gravity-aligned head world orientation (upright, slowly yawing the
        OTHER way from the controller so head-relative yaw is informative). Returns quats."""
        t = (np.arange(n) * dt_ms * 1e6).astype(np.int64)
        hq = np.zeros((n, 4))
        for i in range(n):
            yaw = np.deg2rad(-4.0) * i * dt_ms / 1000.0
            hq[i] = G.quat_exp([0, yaw, 0])
        return t, hq

    # ---- 9: clean track + head pose -> all GOOD, no spurious flip calls ----
    print("[9] C1 anchor: clean track -> GOOD, no false flips")
    n = 60
    t, pos, quat, gb = synth_track(n)
    _, hq = head_track(n)
    a = HA.headpose_anchor_device(t, quat, gb, hq)
    check("no TILT_FLIP on clean data", a.stats["n_tilt_flip"] == 0, str(a.stats))
    check("no YAW_FLIP on clean data", a.stats["n_yaw_flip"] == 0, str(a.stats))
    check("clean data is mostly GOOD", a.stats["n_good"] >= n - 5, str(a.stats))

    # ---- 10: a PURE-YAW flip that gravity CANNOT see -> caught by the head yaw anchor ----
    print("[10] C1 anchor: pure-yaw flip (gravity-blind) -> A_YAW_FLIP")
    t, pos, quat, gb = synth_track(n)
    _, hq = head_track(n)
    k = 30
    quat[k] = G.quat_mul(quat[k], G.quat_exp([0, np.pi, 0]))  # 180 yaw about world-up
    # gravity is STILL observable here (static) and STILL consistent (yaw doesn't move gravity)
    a = HA.headpose_anchor_device(t, quat, gb, hq)
    tilt_k = a.tilt_err[k]
    check("the yaw flip is gravity-consistent (tilt small) -- gravity is blind",
          (not np.isnan(tilt_k)) and tilt_k < TILT_TOL_DEG, f"tilt={tilt_k}")
    check("the head-yaw anchor flags it as A_YAW_FLIP (independent of gravity)",
          a.verdict[k] == HA.A_YAW_FLIP, f"verdict={a.verdict[k]} yaw_resid={a.yaw_resid[k]}")

    # ---- 11: a TILT flip -> A_TILT_FLIP (the strong physical cue still works) ----
    print("[11] C1 anchor: tilt flip -> A_TILT_FLIP")
    t, pos, quat, gb = synth_track(n)
    _, hq = head_track(n)
    quat[k] = mirror_tilt_flip(quat[k])
    a = HA.headpose_anchor_device(t, quat, gb, hq)
    check("tilt flip flagged A_TILT_FLIP", a.verdict[k] == HA.A_TILT_FLIP,
          f"verdict={a.verdict[k]} tilt={a.tilt_err[k]}")

    # ---- 12: no head pose anywhere -> yaw cue silent, never a false YAW_FLIP ----
    print("[12] C1 anchor: no head pose -> abstains on yaw, no false yaw calls")
    t, pos, quat, gb = synth_track(n)
    quat[k] = G.quat_mul(quat[k], G.quat_exp([0, np.pi, 0]))  # yaw flip, but...
    hq_nan = np.full((n, 4), np.nan)                          # ...no head reference to see it
    a = HA.headpose_anchor_device(t, quat, gb, hq_nan)
    check("no YAW_FLIP calls without a head reference (honest abstention)",
          a.stats["n_yaw_flip"] == 0, str(a.stats))
    check("the gravity-blind yaw flip frame is ABSTAIN or GOOD, not a false tilt flip",
          a.verdict[k] in (HA.A_ABSTAIN, HA.A_GOOD), f"verdict={a.verdict[k]}")

    # ---- 13: cues DISAGREE (gravity OK but head says yaw flip on a noisy head frame) -> not a false GOOD ----
    print("[13] C1 anchor: tilt-ok + head-uncertain -> GOOD only when yaw track is confident")
    t, pos, quat, gb = synth_track(n)
    _, hq = head_track(n)
    # make the head yaw track UNCERTAIN at k by NaNing the head around it (too few inliers)
    for j in range(k - HA.YAW_TRACK_WIN, k + HA.YAW_TRACK_WIN + 1):
        if 0 <= j < n and j != k:
            hq[j] = np.nan
    a = HA.headpose_anchor_device(t, quat, gb, hq)
    check("with an uncertain head-yaw track, a tilt-ok frame is still GOOD (gravity carries it)",
          a.verdict[k] == HA.A_GOOD, f"verdict={a.verdict[k]} yaw_resid={a.yaw_resid[k]}")

    # ---- 14: the de-flipper miss-rate scorer runs on degenerate arrays without crashing ----
    print("[14] C1 anchor: yaw-track estimator handles sparse/NaN input")
    rel = np.full(40, np.nan)
    rel[::10] = 5.0  # < YAW_TRACK_MIN_INLIERS neighbours in any +-YAW_TRACK_WIN window
    tr, resid, inl = HA._head_relative_yaw_track(rel, (np.arange(40) * 22e6).astype(np.int64))
    check("sparse yaw input -> all-NaN track (no false confidence)", np.isnan(tr).all(),
          f"non-nan={np.sum(~np.isnan(tr))}")

    # ---- 15: static-gravity carry extends the tilt verdict over a short motion gap, bounded ----
    print("[15] C1 anchor: static body-gravity carry is bounded")
    t = (np.arange(10) * 22e6).astype(np.int64)
    gb2 = np.full((10, 3), np.nan)
    gb2[2] = G.DOWN              # one static sample
    carried = HA._static_body_gravity_carry(t, gb2, carry_ms=60.0)
    check("carry fills nearby frames within the window",
          (not np.isnan(carried[3, 0])) and (not np.isnan(carried[1, 0])))
    check("carry does NOT fill frames beyond the window",
          np.isnan(carried[9, 0]), f"carried[9]={carried[9]}")

    # ===================================================================
    # C2: per-failure-mode metrics (fly-off, jitter, yield) in mse_eval.py
    # ===================================================================
    from mse_eval import fly_off_metrics, jitter_metric, yield_metric

    # ---- 16: fly-off catches a teleport beyond arm-reach and a big consecutive jump ----
    print("[16] C2 metric: fly-off catches a teleport out of the play volume")
    n = 50
    t = (np.arange(n) * 22e6).astype(np.int64)
    pos = np.zeros((n, 3))
    pos[:, 0] = 0.01 * np.arange(n)        # a slow, in-reach hand motion
    fm_clean = fly_off_metrics(t, pos)
    check("clean motion: no frames beyond arm-reach", fm_clean["frac_beyond_reach"] == 0.0,
          str(fm_clean))
    check("clean motion: small consecutive jumps", fm_clean["max_jump_m"] < 0.05, str(fm_clean))
    pos[25] = np.array([5.0, 0.0, 0.0])    # a 5 m teleport (degenerate PnP fly-away)
    fm_fly = fly_off_metrics(t, pos)
    check("fly-off: the teleport is caught as a large jump", fm_fly["max_jump_m"] > 4.0,
          str(fm_fly))
    check("fly-off: the teleport frame is beyond arm-reach",
          fm_fly["frac_beyond_reach"] > 0.0, str(fm_fly))

    # ---- 17: jitter measures high-freq self-deviation; smooth << noisy, gaps not counted ----
    print("[17] C2 metric: jitter separates smooth motion from high-freq noise")
    t = (np.arange(n) * 22e6).astype(np.int64)
    smooth = np.zeros((n, 3)); smooth[:, 0] = 0.01 * np.arange(n)
    noisy = smooth + np.random.normal(0, 0.01, (n, 3))  # 1 cm white jitter
    js = jitter_metric(t, smooth)
    jn = jitter_metric(t, noisy)
    check("smooth motion has near-zero jitter", js["rms_cm"] < 0.2, str(js))
    check("noisy motion has clearly higher jitter", jn["rms_cm"] > js["rms_cm"] + 0.3,
          f"smooth={js} noisy={jn}")
    tg = t.copy(); tg[25:] += int(500e6)  # a 500 ms dropout gap mid-stream
    jg = jitter_metric(tg, smooth)
    check("a dropout gap is not counted as jitter (still smooth)", jg["rms_cm"] < 0.2, str(jg))

    # ---- 18: yield = fraction of rows that carried a pose; latency = sample interval ----
    print("[18] C2 metric: yield + latency")
    t = (np.arange(40) * 22e6).astype(np.int64)
    yld = yield_metric(t, n_cand_valid=30, ref=None)  # 30 of 40 carried a pose
    check("yield is fraction of rows with a pose", abs(yld["yield_pct"] - 75.0) < 1e-6, str(yld))
    check("median interval reflects the sample spacing (~22 ms)",
          abs(yld["median_interval_ms"] - 22.0) < 1.0, str(yld))

    # ===================================================================
    # C3: post-crash tail trim in load_head_pose (headpose_anchor.py)
    # ===================================================================
    print("[19] C3 trim: clean stream is not trimmed")
    # 60s of head sitting roughly still
    nh = 60 * 30  # 30Hz
    th = (np.arange(nh) * (1e9 / 30)).astype(np.int64)
    pos_clean = np.zeros((nh, 3)) + np.random.normal(0, 0.05, (nh, 3))  # 5cm wobble
    trim, drift = HA._detect_tail_drift_trim((th - th[0]) / 1e9, pos_clean)
    check("clean stream: no trim", trim == nh, f"trim={trim} drift={drift:.2f}m")

    print("[20] C3 trim: a SLAM-crash drift tail (sustained >1.5m, no return) is trimmed")
    pos_crash = np.zeros((nh, 3)) + np.random.normal(0, 0.05, (nh, 3))
    # last 5s: monotonic drift to 4m off baseline
    last5_idx = nh - 5 * 30
    drift_dist = np.linspace(0, 4.0, nh - last5_idx)
    pos_crash[last5_idx:, 0] = drift_dist + np.random.normal(0, 0.02, drift_dist.shape[0])
    trim, drift = HA._detect_tail_drift_trim((th - th[0]) / 1e9, pos_crash)
    check("crash tail: trim fires", trim < nh, f"trim={trim} drift={drift:.2f}m")
    check("crash tail: trim point is near the start of the divergence (within 1s)",
          abs(trim - last5_idx) <= 30, f"trim={trim} expected near {last5_idx}")
    check("crash tail: reported end drift is ~4m", 3.0 < drift < 5.0, f"drift={drift:.2f}m")

    print("[21] C3 trim: a legitimate roomscale walk that RETURNS is NOT trimmed")
    pos_walk = np.zeros((nh, 3))
    # walk out 1.8m then return — never sustained-divergent
    half = nh // 2
    pos_walk[:half, 0] = np.linspace(0, 1.8, half)
    pos_walk[half:, 0] = np.linspace(1.8, 0.05, nh - half)
    pos_walk += np.random.normal(0, 0.02, (nh, 3))
    trim, drift = HA._detect_tail_drift_trim((th - th[0]) / 1e9, pos_walk)
    check("walk-and-return: no trim (false-positive guard)", trim == nh,
          f"trim={trim} drift={drift:.2f}m")

    print("[22] C3 trim: a SHORT (<1s) drift at the very end is NOT trimmed (too brief to be confident)")
    pos_short = np.zeros((nh, 3)) + np.random.normal(0, 0.05, (nh, 3))
    # only the last 20 samples (~0.67s) drift far
    pos_short[-20:, 0] = np.linspace(0, 3.0, 20)
    trim, drift = HA._detect_tail_drift_trim((th - th[0]) / 1e9, pos_short)
    check("brief drift: no trim (sustain requirement)", trim == nh,
          f"trim={trim} drift={drift:.2f}m")


    # ===================================================================
    # C4: regression gate fail-safety (regression_check.check)
    # ===================================================================
    print("[23] C4 gate: a missing/NaN gated metric can never report 'ok'")
    from regression_check import GUARDED, check as gate_check

    def mk_metrics(**overrides):
        mt = {
            "yield_pct": 80.0, "pos_med_cm": 0.4, "wrong_branch_pct": 3.0,
            "flip_rate_pct": 0.02, "fly_max_jump_m": 0.3,
            "reentry_bf_err_cm_median": 4.0, "reentry_freeze_err_cm_median": 6.0,
            "reentry_bf_err_cm_mean": 5.0, "reentry_freeze_err_cm_mean": 7.0,
            "reentry_snap_m": 0.05, "fly_frac_beyond_reach": 0.001,
            "anchor_tilt_flip_pct": 0.0, "anchor_yaw_flip_pct": 0.02,
        }
        mt.update(overrides)
        return mt

    base_cols = {"pred": {m.key: mk_metrics()[m.key] for m in GUARDED
                          if "pred" in m.cols and not m.informational and m.floor_key is None}}
    baseline = {"1": base_cols}

    ok, rows = gate_check({1: {"pred": mk_metrics()}}, baseline, 2.0, 3.0)
    check("clean metrics still pass", ok, str(rows))

    ok, rows = gate_check({1: {"pred": mk_metrics(anchor_yaw_flip_pct=float("nan"))}},
                          baseline, 2.0, 3.0)
    st = {r[2]: r[6] for r in rows}
    check("NaN anchor metric FAILs the gate (was silently 'ok')", not ok, str(st))
    check("NaN anchor metric renders FAIL(n/a)", st.get("YAWflip%") == "FAIL(n/a)", str(st))

    ok, rows = gate_check({1: {"pred": mk_metrics(flip_rate_pct=None)}}, baseline, 2.0, 3.0)
    check("None flip rate (0/0 pairs) FAILs the gate", not ok,
          str({r[2]: r[6] for r in rows}))

    ok, rows = gate_check({1: {"pred": mk_metrics(reentry_bf_err_cm_median=None,
                                                  reentry_freeze_err_cm_median=None)}},
                          baseline, 2.0, 3.0)
    st = {r[2]: r[6] for r in rows}
    check("both-sided missing reentry floor renders n/a without failing",
          ok and st.get("reentryErr") == "n/a", str(st))

    ok, rows = gate_check({1: {"pred": mk_metrics(reentry_bf_err_cm_median=None)}},
                          baseline, 2.0, 3.0)
    st = {r[2]: r[6] for r in rows}
    check("one-sided missing reentry floor FAILs (measurement vanished)",
          (not ok) and st.get("reentryErr") == "FAIL(n/a)", str(st))

    print("[24] C4 render: run_ab table renders None/NaN as n/a")
    from run_ab import _fmt
    check("None -> n/a", _fmt(None, "{:.2f}") == "n/a")
    check("NaN -> n/a", _fmt(float("nan"), "{:.2f}") == "n/a")
    check("number formats normally", _fmt(1.234, "{:.2f}") == "1.23")


    print("[25] C2 sentinels: statistics that do not exist render None, never 0.00")
    t1 = np.array([0], dtype=np.int64)
    fm1 = fly_off_metrics(t1, np.zeros((1, 3)))
    check("single pose: no jump statistic (None, not 0.0)",
          fm1["max_jump_m"] is None and fm1["p99_jump_m"] is None and fm1["n_consecutive"] == 0,
          str(fm1))
    tg2 = (np.arange(3) * int(10e9)).astype(np.int64)  # all gaps >> FLY_GAP_MS
    fm2 = fly_off_metrics(tg2, np.zeros((3, 3)))
    check("all-gap trajectory: no jump statistic", fm2["max_jump_m"] is None, str(fm2))
    jm = jitter_metric(tg2, np.zeros((3, 3)))
    check("no contiguous run: jitter is None with n=0",
          jm["rms_cm"] is None and jm["p95_cm"] is None and jm["n"] == 0, str(jm))

    from mse_eval import reentry_accuracy_from_csv
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write("t_ns,drop_optical,opt_valid,opt_px,opt_py,opt_pz,pred_px,pred_py,pred_pz\n")
        for i in range(10):
            tf.write(f"{i*22000000},0,1,0.1,0.2,0.3,0.1,0.2,0.3\n")
        csv_path = tf.name
    ra = reentry_accuracy_from_csv(csv_path)
    _os.unlink(csv_path)
    check("zero re-entry events: floor medians are None with n_events=0",
          ra["bf_err_cm_median"] is None and ra["freeze_err_cm_median"] is None
          and ra["n_events"] == 0, str(ra))


    print("[26] C4 surface: run_ab forwards harness WARN/ERROR stderr even on rc=0")
    from run_ab import run_replay_dual
    import io, contextlib, stat, tempfile as _tf
    with _tf.TemporaryDirectory() as td:
        stub = _os.path.join(td, "stub_replay")
        out_dir = _os.path.join(td, "out")
        with open(stub, "w") as f:
            f.write("#!/bin/bash\n"
                    "echo 'WARN: frame 42 completion barrier timed out' >&2\n"
                    "echo 'benign chatter' >&2\n"
                    "touch \"$6/dev1.csv\" \"$6/dev2.csv\"\n")
        _os.chmod(stub, stat.S_IRWXU)
        cap = io.StringIO()
        with contextlib.redirect_stderr(cap):
            ok_run = run_replay_dual(stub, "f", "c", "t", "l", "r", out_dir, "xv1")
        err = cap.getvalue()
        check("stub replay succeeds", ok_run, err)
        check("WARN line is surfaced on rc=0", "completion barrier timed out" in err, repr(err))
        check("benign stderr chatter is NOT forwarded", "benign chatter" not in err, repr(err))


    print("[27] C4 tolerance: small-baseline pct metrics gate at max(floor, 5x baseline)")
    ok, rows = gate_check({1: {"pred": mk_metrics(flip_rate_pct=0.2)}}, baseline, 2.0, 3.0)
    check("10x flip-rate rise (0.02 -> 0.20) FAILS despite --tol-pct 2.0", not ok,
          str({r[2]: r[6] for r in rows}))
    ok, rows = gate_check({1: {"pred": mk_metrics(anchor_tilt_flip_pct=0.15)}}, baseline, 2.0, 3.0)
    check("anchor flip appearing at 0.15 pts over a 0.0 baseline FAILS", not ok,
          str({r[2]: r[6] for r in rows}))
    ok, rows = gate_check({1: {"pred": mk_metrics(flip_rate_pct=0.05)}}, baseline, 2.0, 3.0)
    check("a within-floor flip wiggle (0.02 -> 0.05) still passes", ok,
          str({r[2]: r[6] for r in rows}))
    ok, rows = gate_check({1: {"pred": mk_metrics(wrong_branch_pct=4.5)}}, baseline, 2.0, 3.0)
    check("percent-scale wrong-branch keeps the old absolute gate (3.0 -> 4.5 passes)", ok,
          str({r[2]: r[6] for r in rows}))
    ok, rows = gate_check({1: {"pred": mk_metrics(wrong_branch_pct=5.5)}}, baseline, 2.0, 3.0)
    check("wrong-branch beyond the absolute tolerance still FAILS (3.0 -> 5.5)", not ok,
          str({r[2]: r[6] for r in rows}))


    print("[28] holdsnap: the hold-then-snap episode detector")
    from holdsnap import hold_snap_episodes, HOLD_GAIN_M
    n = 200
    t_hs = (np.arange(n) * 25_000_000).astype(np.int64)  # 40 Hz optical cadence
    # Clean run: the fused report tracks the optical it is handed, both a few mm off GT.
    err_opt = np.full(n, 0.01)
    err_pred = np.full(n, 0.015)
    pos_pred = np.zeros((n, 3))
    check("clean run yields no episodes", hold_snap_episodes(t_hs, err_pred, err_opt, pos_pred) == [])
    # Inject a hold: 20 frames (500 ms) where the report is 0.8 m off while optical is 1 cm off.
    err_pred_h = err_pred.copy()
    err_pred_h[100:120] = 0.8
    pos_pred_h = pos_pred.copy()
    pos_pred_h[100:120] = [0.8, 0.0, 0.0]  # held away, then snapping back at 120
    eps = hold_snap_episodes(t_hs, err_pred_h, err_opt, pos_pred_h)
    check("injected 500 ms hold is detected as exactly one episode", len(eps) == 1, str(eps))
    check("a flat-error episode is classified held", eps and eps[0]["held"], str(eps))
    check("episode duration matches the injected span", eps and abs(eps[0]["duration_s"] - 0.475) < 1e-6,
          str(eps))
    check("episode reports the exit snap magnitude", eps and abs(eps[0]["exit_snap_m"] - 0.8) < 1e-9,
          str(eps))
    check("episode is marked resolved when frames follow", eps and eps[0]["resolved"], str(eps))
    # A hold that is real but SUB-SLACK cannot have been caused by the jump gate -> not an episode.
    err_pred_s = err_pred.copy()
    err_pred_s[100:120] = HOLD_GAIN_M * 0.9
    check("a sub-slack disagreement is not an episode",
          hold_snap_episodes(t_hs, err_pred_s, err_opt, pos_pred) == [])
    # The filter being wrong is NOT enough: when the optical it was handed is equally wrong, the
    # front-end is the culprit and the gate is blameless -- the metric must stay silent.
    check("pred and opt wrong TOGETHER is not an episode",
          hold_snap_episodes(t_hs, err_pred_h, np.where(err_pred_h > 0.1, 0.8, err_opt),
                             pos_pred_h) == [])
    # Two separated holds must not be merged into one.
    err_pred_2 = err_pred.copy()
    err_pred_2[40:50] = 0.9
    err_pred_2[100:110] = 0.9
    check("two separated holds stay two episodes",
          len(hold_snap_episodes(t_hs, err_pred_2, err_opt, pos_pred)) == 2)
    # An unresolved hold running to the end of the record is still counted (worse, not invisible).
    err_pred_e = err_pred.copy()
    err_pred_e[-10:] = 0.9
    eps_e = hold_snap_episodes(t_hs, err_pred_e, err_opt, pos_pred)
    check("a hold that never resolves is still counted", len(eps_e) == 1, str(eps_e))
    check("an unresolved hold is flagged unresolved", eps_e and not eps_e[0]["resolved"], str(eps_e))
    # A re-entry glide starts a metre out and closes the distance; it must NOT read as a hold.
    err_glide = err_opt.copy()
    err_glide[100:130] = np.linspace(1.1, 0.5, 30)
    eps_g = hold_snap_episodes(t_hs, err_glide, err_opt, pos_pred)
    check("a converging re-entry glide is detected but NOT classified held",
          len(eps_g) == 1 and not eps_g[0]["held"], str(eps_g))
    # An episode must be contiguous in TIME: a run either side of a coast is two events, not one.
    t_gap = t_hs.copy()
    t_gap[110:] += 2_000_000_000  # a 2 s dropout in the middle of the injected hold
    check("a run bridging a 2 s coast splits into two episodes",
          len(hold_snap_episodes(t_gap, err_pred_h, err_opt, pos_pred_h)) == 2)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
