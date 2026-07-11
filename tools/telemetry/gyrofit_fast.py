#!/usr/bin/env python3
"""Fast, parity-validated reimplementation of imu_calib_from_optical.gyro_intrinsic_cleaned.

WHY: the SoT fitter's compose_gyro re-integrates body gyro from the segment anchor on every step
(O(N) per step, Python quaternion loop). That is fine on a ~20s .replay fixture (its design point,
the dev2 live-0611 flow) but intractable on a full 168s session where dev1 has 10s OOV gaps -- a
gap-spanning segment composes over thousands of IMU samples per candidate endpoint. This version
precomputes the CUMULATIVE body-rotation quaternion Q(t) once over the IMU grid; any segment's gyro
rotation is then Q(t0)^-1 * Q(t1) (O(1)). Body-frame right-multiply composition is associative:
Q(t1) = Q(t0) . rel(t0->t1), so rel = Q(t0)^-1 . Q(t1) -- exact up to sub-sample endpoint handling,
which is negligible for the >=12deg segments the fit uses. Parity to the SoT fitter is ENFORCED by
the __main__ gate (nan-aware, full-matrix, segment-set-identical comparison on real capture windows;
non-zero exit on any true divergence — see _parity_main).

The algorithm (segment walk, 12deg span, 30deg consistency gate, Kabsch on rotation axes weighted by
angle, robust median scale, >=8 clean segments, rank>=2 axis-identifiability guard) is byte-for-byte
the SoT algorithm; only compose_gyro is accelerated. The only scale bound either fitter applies is
the (0.5, 1.5) ratio clip before the median; plausibility banding happens downstream in the driver's
cache gate, not in any caller of this fit.
"""
from __future__ import annotations
import numpy as np

MIN_SEG_DEG = 12.0
CONSIST_DEG = 30.0
# Verbatim mirror of imu_calib_from_optical.GYRO_AXIS_EIG_FLOOR (see the derivation note there);
# the parity gate below asserts the two fitters' identifiability verdicts stay identical.
GYRO_AXIS_EIG_FLOOR = 0.02


def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw,
                     aw*bw - ax*bx - ay*by - az*bz])


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]]) / np.dot(q, q)


def qangle(q):
    q = q / np.linalg.norm(q)
    return np.degrees(2 * np.arctan2(np.linalg.norm(q[:3]), abs(q[3])))


def expq(v):
    a = np.linalg.norm(v)
    if a < 1e-12:
        return np.array([v[0]/2, v[1]/2, v[2]/2, 1.0])
    u = v / a
    return np.array([*(np.sin(a/2)*u), np.cos(a/2)])


class CumGyro:
    """Cumulative body-rotation Q(t) = integral of body gyro from its[0] to t (trapezoid on the IMU
    grid, interpolated partial end). Segment rotation rel(t0,t1) = Q(t0)^-1 . Q(t1)."""
    def __init__(self, its, gyro):
        self.its = np.asarray(its, float)
        self.gyro = np.asarray(gyro, float)
        n = len(its)
        Q = np.empty((n, 4))
        Q[0] = [0, 0, 0, 1.0]
        for k in range(1, n):
            dt = self.its[k] - self.its[k-1]
            w = 0.5 * (self.gyro[k] + self.gyro[k-1])
            Q[k] = qmul(Q[k-1], expq(w * dt))
        self.Q = Q

    def at(self, t):
        j = int(np.searchsorted(self.its, t) - 1)
        if j < 0:
            return np.array([0, 0, 0, 1.0])
        if j >= len(self.its) - 1:
            return self.Q[-1].copy()
        # partial from its[j] to t, trapezoid with gyro interpolated at t
        wt = np.array([np.interp(t, self.its, self.gyro[:, c]) for c in range(3)])
        w = 0.5 * (self.gyro[j] + wt)
        return qmul(self.Q[j], expq(w * (t - self.its[j])))

    def rel(self, t0, t1):
        return qmul(qinv(self.at(t0)), self.at(t1))


def gyro_intrinsic_cleaned_fast(its, gyro, pt, pq, cum=None,
                                min_seg_deg=MIN_SEG_DEG, consist_deg=CONSIST_DEG):
    """Same contract as ic.gyro_intrinsic_cleaned: returns
    (M_g, ok, n_seg, scale, mis_deg, n_flips, axis_eig), including the rank>=2 axis-identifiability
    guard (ok=False, identity M_g when the accepted segments' rotation axes are ~single-axis).
    `cum` = a prebuilt CumGyro over the FULL device IMU (reused across slices for speed)."""
    if cum is None:
        cum = CumGyro(its, gyro)
    pt_s = np.asarray(pt, float) * 1e-9
    pq = np.asarray(pq, float)
    ratios, ag, ao, wts = [], [], [], []
    anchor, i, flips, n = 0, 1, 0, len(pt_s)
    while i < n:
        qg = cum.rel(pt_s[anchor], pt_s[i])
        q0 = pq[anchor]; q1 = pq[i]
        qo = qmul(qinv(q0), q1)
        thg, tho = qangle(qg), qangle(qo)
        if tho < min_seg_deg and thg < min_seg_deg:
            i += 1
            continue
        derr = qangle(qmul(qinv(qg), qo))
        if derr < consist_deg and thg > 3.0:
            ratios.append(tho / thg)
            ng = qg[:3] / (np.linalg.norm(qg[:3]) + 1e-12)
            no = qo[:3] / (np.linalg.norm(qo[:3]) + 1e-12)
            ag.append(ng); ao.append(no); wts.append(thg)
            anchor = i
        else:
            flips += 1
        i += 1
    ag = np.array(ag); ao = np.array(ao); w = np.array(wts)
    if len(ag):
        S = (ag * w[:, None]).T @ ag / w.sum()   # angle-weighted gyro-axis scatter, trace 1
        axis_eig = np.linalg.eigvalsh(S)[::-1]
    else:
        axis_eig = np.full(3, float("nan"))
    if len(ag) < 8 or axis_eig[1] < GYRO_AXIS_EIG_FLOOR:
        return np.eye(3), False, len(ratios), float("nan"), float("nan"), flips, axis_eig
    H = (ao * w[:, None]).T @ ag
    U, _, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    mis = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    rr = np.array(ratios)
    rr = rr[(rr > 0.5) & (rr < 1.5)]
    scale = float(np.median(rr)) if len(rr) else 1.0
    return R @ (scale * np.eye(3)), True, len(ratios), scale, mis, flips, axis_eig


PARITY_CAP, PARITY_DEV = "20260706-183948-s3final-gain32-main", 1
PARITY_WINDOWS = [(20, 20), (40, 30), (0, 60)]   # (start offset s, duration s)


def _parity_main():
    """Real-data parity GATE vs the SoT fitter, on recorded capture windows (needs the local
    capture + the results-dir loader). Per window, nan-aware:
      - the segment walk must be identical (nseg, nfl) and the identifiability verdicts must agree
        (ok flags, axis_eig spectra);
      - both-failed identically (ok=False from both, identity M_g) counts as PARITY-OK;
      - otherwise the full 3x3 M_g must match (Frobenius |dM| < 1e-3) and |dscale| < 3e-3.
    Exits non-zero on any true divergence — this is the regression gate for this reimplementation."""
    import importlib.util
    import os
    import sys
    import time
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.join(repo, "results", "dev1-gyro-cal-20260707"))
    from fit_dev1_mg import get_dev
    _s = importlib.util.spec_from_file_location("ic", os.path.join(here, "imu_calib_from_optical.py"))
    ic = importlib.util.module_from_spec(_s); _s.loader.exec_module(ic)
    its, gyro, pt, pq, _f = get_dev(PARITY_CAP, PARITY_DEV)
    n_fail = 0
    for (lo0, dur) in PARITY_WINDOWS:
        lo = pt[0] + int(lo0 * 1e9); hi = lo + int(dur * 1e9); m = (pt >= lo) & (pt < hi)
        t = time.time(); MA, okA, nsA, scA, misA, nflA, eigA = ic.gyro_intrinsic_cleaned(its, gyro, pt[m], pq[m]); ta = time.time() - t
        t = time.time(); MB, okB, nsB, scB, misB, nflB, eigB = gyro_intrinsic_cleaned_fast(its, gyro, pt[m], pq[m]); tb = time.time() - t
        problems = []
        if okA != okB:
            problems.append(f"ok {okA}!={okB}")
        if nsA != nsB:
            problems.append(f"nseg {nsA}!={nsB}")
        if nflA != nflB:
            problems.append(f"nfl {nflA}!={nflB}")
        if not np.allclose(eigA, eigB, atol=1e-4, equal_nan=True):
            problems.append(f"axis_eig {np.round(eigA, 6)}!={np.round(eigB, 6)}")
        dM = float(np.linalg.norm(MA - MB))
        if not okA and not okB:
            if dM != 0.0:
                problems.append(f"both failed but M_g differ |dM|={dM:.2e}")  # contract: identity
        elif okA and okB:
            dsc = abs(scA - scB)
            if not (dM < 1e-3 and dsc < 3e-3):
                problems.append(f"|dM|={dM:.2e} dscale={dsc:.2e}")
        verdict = "PARITY-OK" if not problems else "PARITY-FAIL(" + "; ".join(problems) + ")"
        n_fail += bool(problems)
        print(f"win[{lo0}+{dur}s] n={m.sum():4d}  "
              f"ic: ok={okA} mis={misA:.3f} scale={scA:.4f} nseg={nsA} eig1={eigA[1]:.3f} ({ta:.1f}s) | "
              f"fast: ok={okB} mis={misB:.3f} scale={scB:.4f} nseg={nsB} ({tb:.2f}s) | "
              f"|dM|={dM:.2e}  {verdict}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    import sys
    sys.exit(_parity_main())
