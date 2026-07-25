#!/usr/bin/env python3
"""gt_blob_fix.py -- yaw/position-branch GT correction for the cleaned-GT, driven ONLY by
independent (non-tracker) arbiters: the multi-camera blob-explain + the SLAM head-pose anchor.

THE PROBLEM (Lever 0, RECALL-PUSH-STRATEGY.md)
----------------------------------------------
The cleaned-GT is built from the controller's OWN recorded optical poses. The metric-audit fixed
the TILT branch (gravity-snap). The residual corruption is the YAW/POSITION branch: on the recorded
stream's sustained lock-loss / flip runs, the committed pose is wrong in BOTH orientation AND
position, and there is no recorded correct twin to recover it from. The gravity-snap fixes the tilt
but keeps the (wrong) raw position, so the GT still does not land on the blobs there. RTS-smoothing
through such a run averages the wrong-branch poses into the reference. Scoring a candidate against
such a frame is scoring against a broken yardstick.

THE FIX (anti-circular: drop, do not fabricate)
-----------------------------------------------
We do NOT fix the GT toward the candidate's pose (that would make the GT validate the tracker by
construction). We INVALIDATE the GT frames that an INDEPENDENT arbiter proves unreliable, so the
scorer simply does not score against them. A frame is invalidated only when ALL of:
  1. it has good MULTI-CAM co-visibility (>=2 cameras see >=4 model LEDs) -- the arbiter has the
     evidence to judge it (NOT a sparse single-cam frame, which is genuine depth ambiguity);
  2. the cleaned-GT pose is NOT blob-confirmed (it does not reproject tightly onto the DETECTED
     LED blobs across the co-visible cameras -- the raw observation, pose-independent);
  3. an INDEPENDENT corroborator agrees the GT is on a wrong branch:
       (a) the SLAM head-pose anchor flags a TILT/YAW flip at this frame, OR
       (b) the gravity/anchor-corrected orientation branch (deflip's snapped twin) blob-confirms
           where the GT does not (a provable wrong-branch where a corrected branch is right).
Frames that are merely smoothed slightly off the blobs but on the RIGHT branch (no alternative
confirms, anchor does not flag) are LEFT AS-IS (ABSTAIN) -- never guessed, never dropped. Sparse
single-cam frames are LEFT AS-IS (genuine sparse-depth ambiguity).

This is symmetric: invalidating a frame removes one reading from the recall denominator for ALL
candidates equally; it cannot make a bad tracker look good (the frame is simply not scored).

OUTPUT
------
build_corrupt_mask(capture, dev) -> CorruptResult with the per-frame verdict over the cleaned-GT
reference timeline (aligned to Reference.t_ns), cached to <capture>/telemetry/gt_blobfix_<dev>.npz.
smooth_ref.build_reference consumes the cache (when present) to set ref.valid=False on CORRUPT
frames. Absent the cache, build_reference is unchanged (so all existing tests pass untouched).

Run with the BASE conda (has cv2): ~/miniconda3/bin/python gt_blob_fix.py <capture> [--dev 1|2]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import detection_f1 as DF
import g2_geom as G
from manifest import DEVICE_NAMES
from headpose_anchor import (
    load_head_pose, anchor_from_capture, A_TILT_FLIP, A_YAW_FLIP, A_GOOD,
)
import blob_explain as BE
import replay_contract as RC

# A frame is only CORRUPT if the blob-true pose differs from the GT by MORE than the scoring
# tolerance -- i.e. the GT would WRONGLY mis-score a blob-correct candidate. If the GT is within
# tolerance of the blob-true pose it is FINE for scoring (a blob-correct candidate also matches the
# GT), so refining it a few deg/cm is NOT corruption and the frame is kept. (Scoring bar 5cm/15deg.)
FIX_MIN_DQ_DEG = 15.0
FIX_MIN_DP_CM = 5.0
GT_BLOBFIX_ALGORITHM_VERSION = "gt_blob_fix_v3_provenance"

# per-frame GT verdict
V_GOOD = 0       # GT pose blob-confirmed (lands tightly on the detected blobs, multi-cam)
V_CORRUPT = 1    # GT unreliable on an independent basis -> invalidate (drop from scoreable)
V_ABSTAIN = 2    # GT not confirmed but no independent corroborator says it is wrong -> keep as-is
V_SPARSE = 3     # <2 cam co-visible -> cannot adjudicate (sparse-depth) -> keep as-is

VERDICT_NAME = {V_GOOD: "GOOD", V_CORRUPT: "CORRUPT", V_ABSTAIN: "ABSTAIN", V_SPARSE: "SPARSE"}


@dataclass
class CorruptResult:
    device_id: int
    t_ns: np.ndarray            # (M,) cleaned-GT reference times (== Reference.t_ns)
    verdict: np.ndarray         # (M,) V_GOOD / V_CORRUPT / V_ABSTAIN / V_SPARSE
    gt_confirmed: np.ndarray    # (M,) bool: GT pose blob-confirmed
    gt_reproj_px: np.ndarray    # (M,) pooled GT reproj over confirming cams (NaN if none)
    fix_confirmed: np.ndarray   # (M,) bool: an INDEPENDENT blob-refined pose blob-confirmed
    fix_reproj_px: np.ndarray   # (M,) pooled reproj of the blob-refined pose (NaN if none)
    fix_quat: np.ndarray        # (M,4) the independently blob-identified corrected world quat (NaN)
    fix_pos: np.ndarray         # (M,3) the independently blob-identified corrected world pos (NaN)
    fix_dq_deg: np.ndarray      # (M,) refined-vs-GT orientation delta (deg, NaN)
    fix_dp_cm: np.ndarray       # (M,) refined-vs-GT position delta (cm, NaN)
    anchor_flag: np.ndarray     # (M,) bool: SLAM anchor flagged tilt/yaw flip here
    n_covis: np.ndarray         # (M,) co-visible cameras (>=4 LEDs)
    provenance: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def corrupt_mask(self) -> np.ndarray:
        return self.verdict == V_CORRUPT


def _head_interp(hp, ts_ns: int, max_gap_ms: float = 40.0):
    """Nearest (pos, quat) head pose to ts_ns within max_gap_ms, else (None, None)."""
    j = int(np.searchsorted(hp.t_ns, ts_ns))
    best = -1
    for c in (j - 1, j):
        if 0 <= c < hp.t_ns.shape[0] and abs(int(hp.t_ns[c]) - int(ts_ns)) <= max_gap_ms * 1e6:
            if best < 0 or abs(int(hp.t_ns[c]) - int(ts_ns)) < abs(int(hp.t_ns[best]) - int(ts_ns)):
                best = c
    return (hp.pos[best], hp.quat[best]) if best >= 0 else (None, None)


def _cache_path(capture: Path, dev: int) -> Path:
    return Path(capture) / "telemetry" / f"gt_blobfix_{DEVICE_NAMES[dev]}.npz"


def load_corrupt_mask(capture: Path, dev: int) -> CorruptResult | None:
    """Load a cached CorruptResult, or None if not built yet. Used by smooth_ref (numpy-only;
    does NOT import cv2/g2cam), so the scorer can consume the fix under the g2vr env."""
    p = _cache_path(Path(capture), dev)
    if not p.is_file():
        return None
    d = np.load(p, allow_pickle=True)
    stats = dict(d["stats"].item()) if "stats" in d else {}
    provenance = dict(d["provenance"].item()) if "provenance" in d else {
        "algorithm_version": "legacy_no_provenance",
        "has_witness_seed": None,
        "cache_path": str(p),
    }
    return CorruptResult(
        device_id=int(d["device_id"]),
        t_ns=d["t_ns"],
        verdict=d["verdict"],
        gt_confirmed=d["gt_confirmed"],
        gt_reproj_px=d["gt_reproj_px"],
        fix_confirmed=d["fix_confirmed"],
        fix_reproj_px=d["fix_reproj_px"],
        fix_quat=d["fix_quat"],
        fix_pos=d["fix_pos"],
        fix_dq_deg=d["fix_dq_deg"],
        fix_dp_cm=d["fix_dp_cm"],
        anchor_flag=d["anchor_flag"],
        n_covis=d["n_covis"],
        provenance=provenance,
        stats=stats,
    )


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_provenance(capture: Path, dev: int, witness_csv_dir: "Path | str | None", limit: int) -> dict:
    witness_dir = Path(witness_csv_dir) if witness_csv_dir is not None else None
    witness_csv = witness_dir / f"dev{dev}.csv" if witness_dir is not None else None
    return {
        "algorithm_version": GT_BLOBFIX_ALGORITHM_VERSION,
        "capture": str(Path(capture).resolve()),
        "cams": str(RC.cams_for_capture(capture)),
        "device_id": int(dev),
        "device_name": DEVICE_NAMES[dev],
        "has_witness_seed": witness_dir is not None,
        "witness_csv_dir": str(witness_dir.resolve()) if witness_dir is not None else "",
        "witness_csv_sha256": _sha256_file(witness_csv) if witness_csv is not None else None,
        "limit": int(limit),
        "fix_min_dq_deg": float(FIX_MIN_DQ_DEG),
        "fix_min_dp_cm": float(FIX_MIN_DP_CM),
    }


def build_corrupt_mask(capture: Path, dev: int,
                       cams=None, g2cams=None, led_model=None,
                       cache: "BE.BlobCache | None" = None,
                       write: bool = True, limit: int = 0,
                       witness_csv_dir: "Path | str | None" = None) -> CorruptResult | None:
    """Compute the per-frame GT-corruption verdict over the cleaned-GT reference timeline.

    The blob-PnP refine (the independent corrector) is seeded from the mirror-twin branches of the
    GT AND, when `witness_csv_dir` is given, from a FIXED witness binary's per-frame pose (its
    dev{dev}.csv opt column). The witness is only a SEED HYPOTHESIS -- the multi-cam blob arbiter
    decides whether the refined pose actually lands on the raw blobs, and corrupt frames are DROPPED
    (never overwritten with the witness), so this stays non-circular. Use a fixed witness (e.g.
    g2-unified-best) shared across all scored candidates so the dropped denominator is symmetric.

    NOTE: imports cv2 (via blob_explain) -- run under the BASE conda. Aligns to the cleaned-GT
    reference produced by smooth_ref.build_reference (NOT the raw deflip), since that is exactly
    what the scorer compares against."""
    from smooth_ref import build_reference  # local: numpy-only, no cycle
    capture = Path(capture)
    telem = capture / "telemetry"
    ref = build_reference(telem, dev, apply_blobfix=False)
    if ref is None:
        return None
    hp = load_head_pose(telem)
    if hp is None:
        return None

    if cams is None:
        g2cam, _ = BE._lazy_imports()
        cams_json = RC.cams_for_capture(capture)
        cams = DF.load_cameras(str(cams_json))
        g2cams = g2cam.load_cams(cams_json)
        led_model = g2cam.load_led_model(g2cam.CTRL_LEFT if dev == 1 else g2cam.CTRL_RIGHT)
    if cache is None:
        cache = BE.BlobCache(_frames_dir(capture))

    wit_t = wit_p = wit_q = wit_v = None
    if witness_csv_dir is not None:
        wit_t, wit_p, wit_q, wit_v = _load_witness_csv(Path(witness_csv_dir) / f"dev{dev}.csv")

    # independent SLAM anchor verdicts on the RAW committed stream (a corroborator, NOT required).
    anchor = anchor_from_capture(telem, dev)
    anchor_by_t = {int(t): int(v) for t, v in zip(anchor.t_ns, anchor.verdict)}

    M = ref.t_ns.shape[0]
    verdict = np.full(M, V_SPARSE, dtype=int)
    gt_conf = np.zeros(M, dtype=bool)
    gt_reproj = np.full(M, np.nan)
    fix_conf = np.zeros(M, dtype=bool)
    fix_reproj = np.full(M, np.nan)
    fix_quat = np.full((M, 4), np.nan)
    fix_pos = np.full((M, 3), np.nan)
    fix_dq = np.full(M, np.nan)
    fix_dp = np.full(M, np.nan)
    anc_flag = np.zeros(M, dtype=bool)
    n_covis = np.zeros(M, dtype=int)

    valid_idx = np.where(ref.valid)[0]
    if limit and valid_idx.shape[0] > limit:
        # evenly subsample valid frames for a fast smoke run; unselected stay V_SPARSE (untouched)
        valid_idx = valid_idx[np.linspace(0, valid_idx.shape[0] - 1, limit).astype(int)]
    keep_eval = np.zeros(M, dtype=bool)
    keep_eval[valid_idx] = True
    for i in range(M):
        if not ref.valid[i] or not keep_eval[i]:
            continue
        ts = int(ref.t_ns[i])
        hpos, hq = _head_interp(hp, ts)
        if hpos is None:
            continue
        Rh = DF.quat_to_R(hq)
        q_gt, p_gt = ref.quat[i], ref.pos[i]
        gt = BE.explain_pose(DF.quat_to_R(q_gt), p_gt, Rh, hpos, ts, cams, g2cams, led_model, cache)
        n_covis[i] = gt.n_cams_covisible
        gt_conf[i] = gt.blob_confirmed
        gt_reproj[i] = gt.pooled_reproj_px
        av = anchor_by_t.get(ts, A_GOOD)
        anc_flag[i] = av in (A_TILT_FLIP, A_YAW_FLIP)

        if gt.n_cams_covisible < BE.CONFIRM_CAMS:
            verdict[i] = V_SPARSE
            continue
        if gt.blob_confirmed:
            verdict[i] = V_GOOD
            continue

        # GT not confirmed but MULTI-CAM co-visible. The INDEPENDENT arbiter: enumerate the mirror-twin
        # branch SEEDS of the GT (the GT itself, its yaw-180 heading twin, and its tilt-mirror twins),
        # refine EACH onto the actual DETECTED blobs (cv2 PnP -- the blobs choose the pose, NO tracker),
        # and keep the best multi-cam blob-confirmed, physically-near refine. Enumerating branch seeds
        # is what lets PnP (a local optimiser) reach the CORRECT distant branch when the GT sits on the
        # wrong one. If such a blob-true pose exists and differs from the GT beyond scoring tolerance,
        # the GT is provably on a wrong yaw/position branch -> CORRUPT (we DROP it, never overwrite it).
        # seeds: the GT's geometric branch twins (at the GT position) + the fixed-witness pose (at
        # ITS OWN position, which recovers lock-loss frames where the GT position is wholly wrong).
        seeds = [(q_seed, p_gt) for q_seed in _branch_seeds(q_gt)]
        wq, wp = _witness_seed(wit_t, wit_p, wit_q, wit_v, ts)
        if wq is not None:
            seeds.append((wq, wp))
        best = None  # (reproj, q_fix, p_fix, dq, dp)
        for q_seed, p_seed in seeds:
            q_fix, p_fix, _nc = BE.blob_refine_pose(q_seed, p_seed, Rh, hpos, ts,
                                                    cams, g2cams, led_model, cache)
            if q_fix is None:
                continue
            er = BE.explain_pose(DF.quat_to_R(q_fix), p_fix, Rh, hpos, ts, cams, g2cams, led_model, cache)
            if er.blob_confirmed and (best is None or er.pooled_reproj_px < best[0]):
                best = (er.pooled_reproj_px, q_fix, p_fix,
                        float(G.quat_geodesic_deg(q_fix, q_gt)),
                        float(np.linalg.norm(p_fix - p_gt) * 100.0))
        if best is not None:
            fix_conf[i] = True
            fix_reproj[i], fix_quat[i], fix_pos[i], fix_dq[i], fix_dp[i] = \
                best[0], best[1], best[2], best[3], best[4]

        # CORRUPT only when a blob-confirmed fix exists AND it differs from the GT by MORE than the
        # scoring tolerance (else the GT, though slightly smoothed off the blobs, is still within
        # tolerance of the truth and would score a blob-correct candidate correctly -> keep it).
        if fix_conf[i] and (fix_dq[i] > FIX_MIN_DQ_DEG or fix_dp[i] > FIX_MIN_DP_CM):
            verdict[i] = V_CORRUPT
        else:
            verdict[i] = V_ABSTAIN

    scoreable = ref.valid
    corrupt = verdict == V_CORRUPT
    # SPARSE is only meaningful among EVALUATED valid frames (unevaluated/invalid frames keep the
    # V_SPARSE sentinel but are not part of the reference denominator).
    eval_valid = keep_eval & scoreable
    stats = dict(
        n_ref=int(M),
        n_valid=int(scoreable.sum()),
        n_evaluated=int(eval_valid.sum()),
        n_good=int((verdict == V_GOOD).sum()),
        n_corrupt=int(corrupt.sum()),
        n_abstain=int((verdict == V_ABSTAIN).sum()),
        n_sparse=int((eval_valid & (verdict == V_SPARSE)).sum()),
        n_corrupt_anchor_corroborated=int((corrupt & anc_flag).sum()),
        corrupt_median_dq_deg=float(np.nanmedian(fix_dq[corrupt])) if corrupt.any() else 0.0,
        corrupt_median_dp_cm=float(np.nanmedian(fix_dp[corrupt])) if corrupt.any() else 0.0,
        corrupt_pct_of_valid=100.0 * int(corrupt.sum()) / max(int(scoreable.sum()), 1),
    )
    provenance = _cache_provenance(capture, dev, witness_csv_dir, limit)
    stats["algorithm_version"] = provenance["algorithm_version"]
    stats["has_witness_seed"] = provenance["has_witness_seed"]
    res = CorruptResult(device_id=dev, t_ns=ref.t_ns, verdict=verdict, gt_confirmed=gt_conf,
                        gt_reproj_px=gt_reproj, fix_confirmed=fix_conf, fix_reproj_px=fix_reproj,
                        fix_quat=fix_quat, fix_pos=fix_pos, fix_dq_deg=fix_dq, fix_dp_cm=fix_dp,
                        anchor_flag=anc_flag, n_covis=n_covis, provenance=provenance, stats=stats)
    if write:
        p = _cache_path(capture, dev)
        np.savez(p, device_id=dev, t_ns=res.t_ns, verdict=res.verdict, gt_confirmed=res.gt_confirmed,
                 gt_reproj_px=res.gt_reproj_px, fix_confirmed=res.fix_confirmed,
                 fix_reproj_px=res.fix_reproj_px, fix_quat=res.fix_quat, fix_pos=res.fix_pos,
                 fix_dq_deg=res.fix_dq_deg, fix_dp_cm=res.fix_dp_cm,
                 anchor_flag=res.anchor_flag, n_covis=res.n_covis, provenance=res.provenance,
                 stats=res.stats)
    return res


def _load_witness_csv(path: Path):
    """Load a fixed-witness binary's per-frame opt pose (t_ns, pos, quat, valid) from dev{n}.csv."""
    import csv
    rows = list(csv.DictReader(Path(path).open()))
    t = np.array([int(r["t_ns"]) for r in rows], np.int64)
    p = np.array([[float(r["opt_px"]), float(r["opt_py"]), float(r["opt_pz"])] for r in rows])
    q = np.array([[float(r["opt_qx"]), float(r["opt_qy"]), float(r["opt_qz"]), float(r["opt_qw"])]
                  for r in rows])
    v = np.array([int(float(r.get("opt_valid", "1"))) != 0 for r in rows]) & np.isfinite(p).all(1)
    return t, p, q, v


def _witness_seed(wit_t, wit_p, wit_q, wit_v, ts_ns, max_dt_ms=25.0):
    """Nearest valid witness (q, p) to ts_ns within max_dt_ms, else (None, None)."""
    if wit_t is None:
        return None, None
    j = int(np.searchsorted(wit_t, ts_ns))
    best = -1
    for c in (j - 1, j):
        if 0 <= c < wit_t.shape[0] and wit_v[c] and abs(int(wit_t[c]) - int(ts_ns)) <= max_dt_ms * 1e6:
            if best < 0 or abs(int(wit_t[c]) - int(ts_ns)) < abs(int(wit_t[best]) - int(ts_ns)):
                best = c
    return (wit_q[best], wit_p[best]) if best >= 0 else (None, None)


def _branch_seeds(q_gt):
    """The mirror-twin branch seeds to refine from when the GT is not blob-confirmed. A near-planar
    LED ring has a small set of geometric twins (heading reversal + tilt mirrors); seeding PnP from
    each lets the local optimiser reach the CORRECT branch even when the GT sits on a wrong one.
    The blobs (via PnP + multi-cam confirm) select which seed is right -- this is pose-source
    independent (no tracker/candidate pose is ever used)."""
    yield q_gt
    yield G.yaw_180(q_gt)                                   # heading reversal (pure-yaw twin)
    yield G.quat_mul(np.array([1.0, 0.0, 0.0, 0.0]), q_gt)  # 180 about world-X (tilt mirror)
    yield G.quat_mul(np.array([0.0, 0.0, 1.0, 0.0]), q_gt)  # 180 about world-Z (tilt mirror)


def _frames_dir(capture: Path) -> Path:
    d = Path(capture)
    for sub in ("frames", "frames-session2"):
        if (d / sub).exists():
            return d / sub
    return d / "frames"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--dev", type=int, choices=(1, 2), default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="subsample N valid frames (smoke test)")
    ap.add_argument("--witness", type=Path, default=None,
                    help="fixed-witness replay out/ dir (dev{n}.csv) used as a blob-PnP seed source")
    args = ap.parse_args()
    devs = (args.dev,) if args.dev else (1, 2)
    g2cam, _ = BE._lazy_imports()
    cams_json = RC.cams_for_capture(args.capture)
    cams = DF.load_cameras(str(cams_json))
    g2cams = g2cam.load_cams(cams_json)
    cache = BE.BlobCache(_frames_dir(args.capture))
    for dev in devs:
        mdl = g2cam.load_led_model(g2cam.CTRL_LEFT if dev == 1 else g2cam.CTRL_RIGHT)
        res = build_corrupt_mask(args.capture, dev, cams, g2cams, mdl, cache,
                                 write=not args.no_write, limit=args.limit,
                                 witness_csv_dir=args.witness)
        if res is None:
            print(f"device {dev}: no reference, skipped")
            continue
        s = res.stats
        print(f"==== device {dev} ({DEVICE_NAMES[dev]}) GT blob-fix ====")
        print(f"  valid GT frames        : {s['n_valid']}")
        print(f"  GOOD (blob-confirmed)  : {s['n_good']} ({100*s['n_good']/max(s['n_valid'],1):.0f}%)")
        print(f"  CORRUPT (invalidated)  : {s['n_corrupt']} ({s['corrupt_pct_of_valid']:.0f}% of valid)  "
              f"[anchor-corroborated {s['n_corrupt_anchor_corroborated']}; "
              f"median dq={s['corrupt_median_dq_deg']:.0f}deg dp={s['corrupt_median_dp_cm']:.1f}cm]")
        print(f"  ABSTAIN (kept as-is)   : {s['n_abstain']} ({100*s['n_abstain']/max(s['n_valid'],1):.0f}%)")
        print(f"  SPARSE  (kept as-is)   : {s['n_sparse']} ({100*s['n_sparse']/max(s['n_valid'],1):.0f}%)")
        print(f"  provenance             : {res.provenance['algorithm_version']} "
              f"witness_seed={res.provenance['has_witness_seed']}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
