#!/usr/bin/env python3
"""diag_coast.py -- classify the FN frames of the g2-cardinality canonical replay into the four
GT-independent categories and measure the COAST/PRED-stream recoverability of the genuine-sparse
(category b) misses.

READ-ONLY diagnostic. Reuses the existing tracking_metrics reference grid + scorer (the exact
FULL-recall yardstick of baseline-g2-cardinality.md), the headpose_anchor independent SLAM verdict,
and the blob_explain multi-cam arbiter. No re-replay: consumes the saved canonical CSVs.

Categories (per the diag-coast task):
  (a) ABSTAIN-GT-SIDE     : tracker committed an opt pose that IS blob_confirmed (>=2 cams) yet
                            the cleaned-GT marks the frame FN -> GT wrong/uncertain, not a tracker miss.
  (b) GENUINE-NO_COMMIT-SPARSE : tracker did NOT commit opt AND <4 detectable LEDs in any one cam
                            -> physically sparse; coast/IMU is the only lever.
  (c) NO_COMMIT-MATCHABLE : tracker did NOT commit opt but >=4 LEDs detectable -> matcher accept-gate lever.
  (d) GENUINE-WRONG-COMMIT: tracker committed an opt pose that is NOT correct AND NOT blob_confirmed
                            -> precision/matcher lever.

For (b), each frame is binned by coast age since the last opt-correct fix into the recall-ceiling
bands (0-80 free / 80-160 convertible / 160-320 / >320 physically-lost), and we report whether the
PRED stream already recovers it within the 5cm/15deg tolerance (the pred-stream upside that exists
today) vs whether better coast (P2.5/body-prior/preintegration) could plausibly convert it.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tracking_metrics as TM
import headpose_anchor as HA
import blob_explain as BE
import detection_f1 as DF
import g2_geom as G
from detection_f1 import load_cameras, load_led_model, R_to_quat, quat_to_R

CAPTURES = {
    "xv1": Path("/home/mrwhite0racle/g2-linux-research/captures/20260528-080421-xv-session1"),
    "clean2": Path("/home/mrwhite0racle/g2-linux-research/captures/20260526-175615-clean-session2"),
}
CSV_ROOT = Path("/home/mrwhite0racle/g2-linux-research/docs/sota-research/roadmap/a1-artifacts")
CAMS_JSON = str(__import__("pathlib").Path(__file__).resolve().parent / "data/hmd-cameras-replay.json")  # pinned: live driver rewrites the ~/.config copy
CTRL = {
    1: str(Path.home() / ".config/monado/wmr/controller_A85K1111630014L.json"),
    2: str(Path.home() / ".config/monado/wmr/controller_A85K5091930012R.json"),
}
POS_CM = 5.0
ORI_DEG = 15.0
MATCH_MS = 25.0
HEAD_MATCH_MS = 50.0
MAX_REF_GAP_MS = 150.0

# coast bands (ms) -- the recall-ceiling finding's bands
def coast_band(age_ms: float) -> str:
    if age_ms <= 80:
        return "0_80ms"
    if age_ms <= 160:
        return "80_160ms"
    if age_ms <= 320:
        return "160_320ms"
    return "gt_320ms"


def analyze_cell(cap_name: str, dev: int, cams, g2cams, blob_cache_dir, n_arbiter: int):
    cap = CAPTURES[cap_name]
    ctrl = CTRL[dev]
    g2cam = BE._lazy_imports()[0]
    grid = TM.build_reference_grid(cap, dev, cams, ctrl, MAX_REF_GAP_MS, HEAD_MATCH_MS,
                                   detect_leds=3, pose_leds=4, high_leds=7,
                                   reference_qc="confirmed")
    csv_path = CSV_ROOT / cap_name / "out" / f"dev{dev}.csv"
    opt = TM.load_csv_stream(csv_path, "opt")
    pred = TM.load_csv_stream(csv_path, "pred")
    visible = grid.detect_visible  # FULL recall in-view mask (same as baseline default)

    opt_score = TM.score_stream(grid, opt, visible, MATCH_MS, POS_CM, ORI_DEG)
    pred_score = TM.score_stream(grid, pred, visible, MATCH_MS, POS_CM, ORI_DEG)

    scoreable = grid.scoreable
    in_view = visible & scoreable
    opt_correct = opt_score["_frame_has_correct"]
    opt_accept = opt_score["_frame_has_accept"]
    pred_correct = pred_score["_frame_has_correct"]
    pred_accept = pred_score["_frame_has_accept"]
    opt_poscorr = ~np.isnan(opt_score["_pos_err_cm_by_frame"]) & (opt_score["_pos_err_cm_by_frame"] < POS_CM)

    fn = in_view & ~opt_correct
    fn_idx = np.flatnonzero(fn)

    led_max = grid.led_count.max(axis=1)              # GT-independent: in-FOV model LEDs in the richest cam
    led_total = grid.led_count.sum(axis=1)            # pooled in-FOV LEDs across cams
    blob_total = grid.frames.blob_total               # raw detected blobs (cluster-flagged) summed over cams
    blob_max = grid.frames.blob_max_cam               # raw detected blobs in the richest cam

    # coast age since last opt-correct (on the frame grid, scoreable frames only)
    age_ms = np.full(grid.frames.t_ns.shape[0], np.inf)
    last_t = None
    for i, t in enumerate(grid.frames.t_ns):
        if not scoreable[i]:
            continue
        if opt_correct[i]:
            last_t = int(t)
        if last_t is not None:
            age_ms[i] = (int(t) - last_t) / 1e6

    led_model = g2cam.load_led_model(g2cam.CTRL_LEFT if dev == 1 else g2cam.CTRL_RIGHT)
    cache = BE.BlobCache(cap / "frames", disk_cache=blob_cache_dir / f"{cap_name}_blobs.npz")
    opt_t = opt.t_ns

    def arbiter(i, R_dev, p_dev):
        """Multi-cam blob-explain of a world pose at FN frame i. Returns (ExplainResult,
        max_matched_in_any_cam) -- the detection-grounded 'detectable+matchable LEDs in the
        richest single camera' (the <4 vs >=4 detectable signal)."""
        t_frame = int(grid.frames.t_ns[i])
        R_hmd = quat_to_R(grid.head_quat[i][None, :])[0]
        ex = BE.explain_pose(R_dev, p_dev, R_hmd, grid.head_pos[i], t_frame,
                             cams, g2cams, led_model, cache)
        max_matched = max((c.n_matched for c in ex.per_cam), default=0)
        return ex, max_matched

    # --- categorize every FN frame on detection-grounded GT-independent evidence ---
    # The grid led_count is GEOMETRIC in-FOV (facing cone); whether those LEDs were actually
    # DETECTED+MATCHABLE is the arbiter's job (a pose is multi-cam blob_confirmed only if its LEDs
    # land tightly on detected blobs in >=2 cams). So sparse-vs-matchable is decided by the arbiter
    # on the cleaned-GT pose (GT-independent blobs), not by the geometric count alone.
    cat = {}
    detail = {}   # frame -> arbiter evidence (for the report + spot frames)
    committed_fn = [i for i in fn_idx if bool(opt_accept[i])]
    nocommit_fn = [i for i in fn_idx if not bool(opt_accept[i])]

    # committed-FN: (a) GT-side abstain if the COMMITTED opt pose is itself blob_confirmed (the
    # tracker was right, GT marks it FN); else (d) genuine wrong-commit.
    for i in committed_fn:
        t_frame = int(grid.frames.t_ns[i])
        j = TM._nearest_index(opt_t, t_frame, int(MATCH_MS * 1e6))
        if j < 0 or not opt.valid[j]:
            cat[int(i)] = "d"
            detail[int(i)] = dict(why="accept_no_finite_pose")
            continue
        R_dev = quat_to_R(opt.quat[j][None, :])[0]
        ex, mm = arbiter(i, R_dev, opt.pos[j])
        cat[int(i)] = "a" if ex.blob_confirmed else "d"
        detail[int(i)] = dict(led_geo=int(led_max[i]), max_matched=mm,
                              n_cams_confirm=ex.n_cams_confirm,
                              match_frac=round(ex.pooled_match_frac, 3),
                              blob_confirmed=bool(ex.blob_confirmed))

    # no-commit-FN: arbiter on the cleaned-GT pose. >=2 cams confirm with >=4 matched in the richest
    # cam => the LEDs WERE detectable+matchable (the depth-resolving unit existed) -> (c) matcher
    # accept-gate lever. Otherwise the controller is physically sparse / single-cam-only at this
    # instant (no >=2-cam multi-LED fit on detected blobs) -> (b) genuine sparse / lock-loss.
    for i in nocommit_fn:
        R_dev = quat_to_R(grid.ref_quat[i][None, :])[0]
        ex, mm = arbiter(i, R_dev, grid.ref_pos[i])
        matchable = ex.blob_confirmed and mm >= 4
        cat[int(i)] = "c" if matchable else "b"
        detail[int(i)] = dict(led_geo=int(led_max[i]), max_matched=mm,
                              n_cams_covisible=ex.n_cams_covisible,
                              n_cams_confirm=ex.n_cams_confirm,
                              match_frac=round(ex.pooled_match_frac, 3),
                              blob_confirmed=bool(ex.blob_confirmed))
    cache.save_disk()

    # ---- independent head-pose anchor verdict on the opt stream (for the report context) ----
    anchor = HA.score_candidate_against_anchor(cap / "telemetry", dev, opt.t_ns,
                                               opt.quat) or {}

    # ---- a few example frames per category for the report ----
    sample_check = {}
    rng = np.random.default_rng(0)
    for letter in ("a", "b", "c", "d"):
        cand = [i for i in fn_idx if cat[int(i)] == letter]
        if not cand:
            sample_check[letter] = []
            continue
        pick = rng.choice(cand, size=min(n_arbiter, len(cand)), replace=False)
        sample_check[letter] = [dict(frame=int(i), t_ns=int(grid.frames.t_ns[i]),
                                     age_ms=(round(float(age_ms[i]), 1) if np.isfinite(age_ms[i]) else None),
                                     pred_correct=bool(pred_correct[i]), **detail[int(i)]) for i in pick]

    # ---- tally ----
    n_fn = int(fn.sum())
    counts = Counter(cat[int(i)] for i in fn_idx)
    cat_counts = {k: counts.get(k, 0) for k in ("a", "b", "c", "d")}

    # ---- (b) coast-band breakdown + pred recoverability ----
    b_frames = [i for i in fn_idx if cat[int(i)] == "b"]
    band_tally = {k: 0 for k in ("0_80ms", "80_160ms", "160_320ms", "gt_320ms")}
    band_pred_recovered = {k: 0 for k in band_tally}   # pred within tol on this FN frame TODAY
    band_pred_err = {k: [] for k in band_tally}
    no_prior_lock = 0
    for i in b_frames:
        a = age_ms[i]
        if not np.isfinite(a):
            no_prior_lock += 1          # never had an opt-correct fix before -> cold (no coast origin)
            continue
        band = coast_band(a)
        band_tally[band] += 1
        if pred_correct[i]:
            band_pred_recovered[band] += 1
        pe = pred_score["_pos_err_cm_by_frame"][i]
        if np.isfinite(pe):
            band_pred_err[band].append(float(pe))

    # ---- whole-cell pred-recovers-the-opt-FN (any band) ----
    pred_recovers_fn = int(np.sum(fn & pred_correct))

    return dict(
        capture=cap_name, dev=dev,
        n_scoreable=int(scoreable.sum()),
        n_in_view=int(in_view.sum()),
        opt_TP=opt_score["TP"], opt_FN=opt_score["FN"], opt_recall=round(opt_score["recall"], 4),
        pred_TP=pred_score["TP"], pred_FN=pred_score["FN"], pred_recall=round(pred_score["recall"], 4),
        n_fn=n_fn,
        cat_counts=cat_counts,
        cat_pct={k: round(100.0 * v / max(n_fn, 1), 1) for k, v in cat_counts.items()},
        b_no_prior_lock=no_prior_lock,
        b_band_tally=band_tally,
        b_band_pred_recovered=band_pred_recovered,
        b_band_pred_err_median_cm={k: (round(float(np.median(v)), 1) if v else None)
                                   for k, v in band_pred_err.items()},
        pred_recovers_fn_total=pred_recovers_fn,
        anchor={k: anchor.get(k) for k in ("n_judged", "tilt_flip_rate_pct", "yaw_flip_rate_pct",
                                            "wrong_branch_rate_pct")},
        arbiter_committed_fn={"n": len(committed_fn),
                              "a_confirmed": sum(1 for i in committed_fn if cat[int(i)] == "a"),
                              "d_unconfirmed": sum(1 for i in committed_fn if cat[int(i)] == "d")},
        sample_check=sample_check,
    )


def main():
    n_arbiter = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    g2cam, _ = BE._lazy_imports()
    cams = DF.load_cameras(g2cam.HMD_CAMERAS.as_posix())   # detection_f1 cameras (grid + world<-cam chain)
    g2cams = g2cam.load_cams()                              # g2cam camera dict (projection), keyed by cam id
    blob_cache_dir = Path("/tmp/diag_coast_cache")
    blob_cache_dir.mkdir(exist_ok=True)
    out = {}
    for cap_name in ("xv1", "clean2"):
        for dev in (1, 2):
            key = f"{cap_name}_dev{dev}"
            print(f"==== {key} ====", flush=True)
            res = analyze_cell(cap_name, dev, cams, g2cams, blob_cache_dir, n_arbiter)
            out[key] = res
            print(json.dumps({k: v for k, v in res.items() if k != "sample_check"}, indent=2), flush=True)
    Path("/tmp/diag_coast_result.json").write_text(json.dumps(out, indent=2))
    print("\nwrote /tmp/diag_coast_result.json")


if __name__ == "__main__":
    main()
