#!/usr/bin/env python3
"""fov_edge_position.py -- POSITION-only recall + accuracy at the FOV edge / sparse-LED regime.

THE QUESTION
------------
Full 6DoF is geometrically hard when the controller is at the image border, seen by only a
few LEDs, or visible to a single camera (the regime the L1 / ladder lever targets). There we may
still pin the controller's POSITION even when its ORIENTATION branch (the mirror-twin) is
unresolved. This tool measures exactly that: for the FOV-edge population it computes

  * POSITION recall  : a frame where the committed/predicted pose lands <= POS_TOL_CM (default 10 cm)
                       of the cleaned-GT, REGARDLESS of orientation.
  * POSITION accuracy: RMSE / median / p95 of the position error on the scored frames.
  * the 6DoF-vs-position-only delta: how much recall relaxing to position-only recovers.

It contrasts the FOV-edge cell with the full-6DoF recall on the SAME frames so the lift from
relaxing to position-only is explicit, and reports the same per controller, for both the optical
front-end (opt) and the ESKF prediction (pred).

DEFINING THE FOV-EDGE / SPARSE POPULATION (precise, GT-projected)
-----------------------------------------------------------------
Built on the SAME cleaned-GT + head-pose + per-camera LED census the canonical scorer uses
(tracking_metrics.build_reference_grid -> detection_f1.in_fov_per_cam, the exact world<-cam +
rt8 + facing chain the live tracker uses). For each scoreable frame we know, per camera, how many
model LEDs project in-frame (front-facing) and the bounding box of those LEDs. The population is
the UNION of three independently-sufficient sparse conditions (a frame is FOV-edge if ANY holds):

  EDGE_BORDER   : the visible-LED centroid in the best (most-LEDs) camera lies within EDGE_FRAC of
                  an image border (default 0.12 -> within ~57px of a 480px / ~77px of a 640px edge).
  EDGE_FEWLED   : the best camera sees only [1..POSE_LEDS-1] LEDs (default 1..3) -- below the 4-LED
                  PnP floor, so no single view can resolve full 6DoF on its own.
  EDGE_SINGLECAM: only ONE camera sees >= DETECT_LEDS (default 3) LEDs -- no 2nd view to dissolve
                  the mirror twin (the single-cam-dwell regime).

This is the GT-projected census (where the controller REALLY was), so the population is defined
independently of what the tracker committed -- not self-referential to the candidate being scored.

Scoring mirrors tracking_metrics.score_stream exactly (nearest-frame join within MATCH_MS, per-frame
best by position error), so the FOV-edge numbers are directly comparable to the canonical recall.

Usage (g2vr conda):
  PYTHONNOUSERSITE=1 ~/miniconda3/envs/g2vr/bin/python fov_edge_position.py \
      --cell xv1:DIR:CSVDIR --cell clean2:DIR:CSVDIR [--pos-tol-cm 10] [--out JSON]
where CSVDIR holds dev1.csv / dev2.csv with opt_* and pred_* columns (run_ab / s7s8 replay output).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import DEVICE_NAMES  # noqa: E402
from tracking_metrics import build_reference_grid, load_csv_stream, _nearest_indices  # noqa: E402
from detection_f1 import (  # noqa: E402
    DEFAULT_CAMS,
    DEFAULT_CTRL_LEFT,
    DEFAULT_CTRL_RIGHT,
    Camera,
    cam_visible,
    load_cameras,
    load_led_model,
    pose_flip_YZ,
    pose_mul,
    pose_inv,
    quat_geodesic_deg,
    quat_to_R,
    rt8_project,
)

# --- FOV-edge population knobs (each justified; this IS the definition) ---
EDGE_FRAC = 0.12          # border band as a fraction of each image dimension (~57px of 480, ~77px of 640)
POSE_LEDS = 4             # the single-view 6DoF PnP floor (pose_metrics 4-LED minimum)
DETECT_LEDS = 3           # a camera "sees" the controller when >= this many LEDs project in-frame

# --- correctness thresholds (match the canonical scorer; pos-tol relaxed for position-only) ---
FULL_POS_CM = 5.0         # full-6DoF position threshold (tracking_metrics default)
FULL_ORI_DEG = 15.0       # full-6DoF orientation threshold (tracking_metrics default)
POS_TOL_CM = 10.0         # relaxed position-only acceptance radius (any orientation)
MATCH_MS = 25.0           # candidate<->frame nearest join (tracking_metrics default)
WRONG_BRANCH_DEG = 90.0   # orientation error at/above this is a mirror/yaw wrong-branch


def _visible_centroid_per_cam(R_dev, t_dev, R_hmd, t_hmd, led_pos, led_nrm,
                              cams: list[Camera]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For one frame, per camera return (n_visible, centroid_uv (2,), border_frac).

    Mirrors detection_f1.in_fov_per_cam's geometry exactly (same world<-cam flip chain + rt8 +
    facing test) but additionally returns the in-frame visible-LED CENTROID and its normalised
    distance to the nearest image border (0 = on the edge, 0.5 = dead centre). border_frac is NaN
    when the camera sees no visible LEDs."""
    nC = len(cams)
    n_vis = np.zeros(nC, dtype=np.int32)
    cuv = np.full((nC, 2), np.nan)
    border = np.full(nC, np.nan)
    R_cv_hmd, t_cv_hmd = pose_flip_YZ(R_hmd, t_hmd)
    R_cv_model, t_cv_model = pose_flip_YZ(R_dev, t_dev)
    for j, c in enumerate(cams):
        R_cv_cam, t_cv_cam = pose_mul(R_cv_hmd, t_cv_hmd, c.P_imu_cam_R, c.P_imu_cam_t)
        R_cam_cv, t_cam_cv = pose_inv(R_cv_cam, t_cv_cam)
        R_cam_model, t_cam_model = pose_mul(R_cam_cv, t_cam_cv, R_cv_model, t_cv_model)
        led_c = led_pos @ R_cam_model.T + t_cam_model
        nrm_c = led_nrm @ R_cam_model.T
        vis = cam_visible(led_c, nrm_c, c)
        nv = int(vis.sum())
        n_vis[j] = nv
        if nv == 0:
            continue
        u, v, _ = rt8_project(led_c[vis], c)
        cu, cv = float(np.mean(u)), float(np.mean(v))
        cuv[j] = (cu, cv)
        # normalised distance to the nearest of the four borders
        bx = min(cu, c.width - cu) / c.width
        by = min(cv, c.height - cv) / c.height
        border[j] = min(bx, by)
    return n_vis, cuv, border


@dataclass
class FrameCensus:
    n_frames: int
    scoreable: np.ndarray          # (N,) bool
    max_leds: np.ndarray           # (N,) best-cam visible LED count
    n_cams_detect: np.ndarray      # (N,) cams with >= DETECT_LEDS LEDs
    best_border_frac: np.ndarray   # (N,) border frac of the best (most-LEDs) camera (NaN if none)
    edge_border: np.ndarray        # (N,) bool
    edge_fewled: np.ndarray        # (N,) bool
    edge_singlecam: np.ndarray     # (N,) bool
    fov_edge: np.ndarray           # (N,) bool -- the union population
    interior: np.ndarray           # (N,) bool -- scoreable & ~fov_edge (the easy regime, for contrast)


def build_census(grid, cams, led_pos, led_nrm) -> FrameCensus:
    """Per-frame FOV-edge classification over the reference grid's scoreable frames."""
    N = grid.frames.t_ns.shape[0]
    scoreable = grid.scoreable
    max_leds = grid.led_count.max(axis=1) if grid.led_count.size else np.zeros(N, dtype=np.int32)
    n_cams_detect = (grid.led_count >= DETECT_LEDS).sum(axis=1)
    best_border = np.full(N, np.nan)
    R_ref = quat_to_R(grid.ref_quat)
    R_head = quat_to_R(grid.head_quat)
    for i in np.flatnonzero(scoreable):
        n_vis, _cuv, border = _visible_centroid_per_cam(
            R_ref[i], grid.ref_pos[i], R_head[i], grid.head_pos[i], led_pos, led_nrm, cams)
        if n_vis.max() > 0:
            best_cam = int(np.argmax(n_vis))
            best_border[i] = border[best_cam]

    edge_border = scoreable & np.isfinite(best_border) & (best_border < EDGE_FRAC)
    edge_fewled = scoreable & (max_leds >= 1) & (max_leds < POSE_LEDS)
    edge_singlecam = scoreable & (n_cams_detect == 1)
    fov_edge = scoreable & (edge_border | edge_fewled | edge_singlecam)
    interior = scoreable & ~fov_edge
    return FrameCensus(
        n_frames=N, scoreable=scoreable, max_leds=max_leds, n_cams_detect=n_cams_detect,
        best_border_frac=best_border, edge_border=edge_border, edge_fewled=edge_fewled,
        edge_singlecam=edge_singlecam, fov_edge=fov_edge, interior=interior)


def score_per_frame(grid, candidate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame (over the grid) best position error (cm), orientation error (deg), and an
    accept mask -- exactly tracking_metrics.score_stream's join (nearest within MATCH_MS, keep the
    lowest-position-error candidate per frame). Returns (pos_err_cm, ori_err_deg, has_accept)."""
    N = grid.frames.t_ns.shape[0]
    pos_err = np.full(N, np.nan)
    ori_err = np.full(N, np.nan)
    has_accept = np.zeros(N, dtype=bool)
    valid = (candidate.valid
             & np.isfinite(candidate.pos).all(axis=1)
             & np.isfinite(candidate.quat).all(axis=1))
    sidx = np.flatnonzero(valid)
    fidx = _nearest_indices(grid.frames.t_ns, candidate.t_ns[sidx], int(MATCH_MS * 1e6))
    keep = fidx >= 0
    sidx, fidx = sidx[keep], fidx[keep]
    for si, fi in zip(sidx, fidx):
        has_accept[fi] = True
        if not grid.scoreable[fi]:
            continue
        pe = float(np.linalg.norm(candidate.pos[si] - grid.ref_pos[fi]) * 100.0)
        if np.isnan(pos_err[fi]) or pe < pos_err[fi]:
            pos_err[fi] = pe
            ori_err[fi] = float(quat_geodesic_deg(candidate.quat[si], grid.ref_quat[fi]))
    return pos_err, ori_err, has_accept


def _acc(pe: np.ndarray) -> dict[str, Any]:
    pe = pe[np.isfinite(pe)]
    if pe.size == 0:
        return {"n": 0, "rmse_cm": None, "median_cm": None, "p95_cm": None}
    return {
        "n": int(pe.size),
        "rmse_cm": float(np.sqrt(np.mean(pe * pe))),
        "median_cm": float(np.median(pe)),
        "p95_cm": float(np.percentile(pe, 95)),
    }


def recall_on(mask: np.ndarray, pos_err, ori_err, has_accept, pos_tol_cm: float) -> dict[str, Any]:
    """Recall + accuracy on a frame population `mask` (already scoreable).

    Denominator = all in-population frames (a missed/un-accepted frame is a recall miss, matching
    the canonical FULL-recall denominator). full = pos<FULL_POS_CM AND ori<FULL_ORI_DEG;
    position-only = pos<=pos_tol_cm (any orientation)."""
    denom = int(mask.sum())
    scored = mask & np.isfinite(pos_err)            # a candidate was joined to this frame
    pos_ok = scored & (pos_err <= pos_tol_cm)
    full_ok = scored & (pos_err < FULL_POS_CM) & (ori_err < FULL_ORI_DEG)
    # of the position-correct frames, how many have a WRONG orientation branch (the recovered-by-
    # relaxing set): position pinned but orientation flipped/unresolved.
    pos_ok_wrong_ori = pos_ok & (ori_err >= WRONG_BRANCH_DEG)
    pos_ok_ori_mid = pos_ok & (ori_err >= FULL_ORI_DEG) & (ori_err < WRONG_BRANCH_DEG)
    return {
        "frames": denom,
        "scored_frames": int(scored.sum()),
        "no_candidate_frames": int((mask & ~np.isfinite(pos_err)).sum()),
        "position_recall_pct": 100.0 * int(pos_ok.sum()) / max(denom, 1),
        "full6dof_recall_pct": 100.0 * int(full_ok.sum()) / max(denom, 1),
        "delta_pp": 100.0 * (int(pos_ok.sum()) - int(full_ok.sum())) / max(denom, 1),
        "position_correct_n": int(pos_ok.sum()),
        "full6dof_correct_n": int(full_ok.sum()),
        "pos_ok_but_wrong_branch_n": int(pos_ok_wrong_ori.sum()),
        "pos_ok_but_ori_15_90_n": int(pos_ok_ori_mid.sum()),
        "position_accuracy_all_scored": _acc(pos_err[scored]),
        "position_accuracy_position_correct": _acc(pos_err[pos_ok]),
    }


def analyse_cell(tag, capture: Path, csv_dir: Path, cams, ctrl_left, ctrl_right,
                 pos_tol_cm: float) -> list[dict[str, Any]]:
    out = []
    for dev, ctrl in ((1, ctrl_left), (2, ctrl_right)):
        grid = build_reference_grid(
            capture, dev, cams, ctrl,
            max_ref_gap_ms=150.0, head_match_ms=50.0,
            detect_leds=DETECT_LEDS, pose_leds=POSE_LEDS, high_leds=7,
            reference_qc="confirmed")
        led_pos, led_nrm = load_led_model(ctrl)
        census = build_census(grid, cams, led_pos, led_nrm)
        csv = csv_dir / f"dev{dev}.csv"
        cell = {
            "cell": f"{tag}-d{dev}",
            "capture": str(capture),
            "csv": str(csv),
            "device": dev,
            "device_name": DEVICE_NAMES.get(dev, str(dev)),
            "population": {
                "scoreable_frames": int(census.scoreable.sum()),
                "fov_edge_frames": int(census.fov_edge.sum()),
                "fov_edge_pct_of_scoreable": 100.0 * int(census.fov_edge.sum()) / max(int(census.scoreable.sum()), 1),
                "interior_frames": int(census.interior.sum()),
                "edge_border_frames": int(census.edge_border.sum()),
                "edge_fewled_frames": int(census.edge_fewled.sum()),
                "edge_singlecam_frames": int(census.edge_singlecam.sum()),
            },
            "scores": {},
        }
        for col in ("opt", "pred"):
            stream = load_csv_stream(csv, col)
            pe, oe, acc = score_per_frame(grid, stream)
            cell["scores"][col] = {
                "fov_edge": recall_on(census.fov_edge, pe, oe, acc, pos_tol_cm),
                "interior": recall_on(census.interior, pe, oe, acc, pos_tol_cm),
                "all_scoreable": recall_on(census.scoreable, pe, oe, acc, pos_tol_cm),
                "edge_border": recall_on(census.edge_border, pe, oe, acc, pos_tol_cm),
                "edge_fewled": recall_on(census.edge_fewled, pe, oe, acc, pos_tol_cm),
                "edge_singlecam": recall_on(census.edge_singlecam, pe, oe, acc, pos_tol_cm),
            }
        out.append(cell)
    return out


def print_report(cells: list[dict[str, Any]]) -> None:
    print(f"\n{'='*112}")
    print("FOV-EDGE POSITION-ONLY RECALL + ACCURACY")
    print(f"  FOV-edge = best-cam LED centroid within {EDGE_FRAC:.0%} of an image border  OR  "
          f"best-cam LEDs in [1..{POSE_LEDS-1}]  OR  only 1 cam sees >= {DETECT_LEDS} LEDs")
    print(f"  position recall: pos err <= {POS_TOL_CM:.0f} cm (any orientation) | "
          f"full 6DoF: pos < {FULL_POS_CM:.0f} cm AND ori < {FULL_ORI_DEG:.0f} deg | denom = all in-pop frames")
    print(f"{'='*112}")
    hdr = (f"{'cell':12s} {'col':4s} {'edgeFr':>7s} {'edge%':>6s} | "
           f"{'posR%':>6s} {'6dofR%':>7s} {'Δpp':>6s} | {'posRMSE':>8s} {'posMed':>7s} {'posP95':>7s} | "
           f"{'wrongBr':>7s} {'mid':>5s}")
    for cell in cells:
        print(f"\n--- {cell['cell']} ({cell['device_name']})  scoreable={cell['population']['scoreable_frames']} "
              f"fov_edge={cell['population']['fov_edge_frames']} "
              f"({cell['population']['fov_edge_pct_of_scoreable']:.0f}%)  "
              f"[border={cell['population']['edge_border_frames']} "
              f"fewLED={cell['population']['edge_fewled_frames']} "
              f"1cam={cell['population']['edge_singlecam_frames']}] ---")
        print(hdr)
        print("-" * 112)
        for col in ("opt", "pred"):
            for pop in ("fov_edge", "interior", "all_scoreable"):
                r = cell["scores"][col][pop]
                a = r["position_accuracy_all_scored"]
                rmse = f"{a['rmse_cm']:.1f}" if a["rmse_cm"] is not None else "n/a"
                med = f"{a['median_cm']:.1f}" if a["median_cm"] is not None else "n/a"
                p95 = f"{a['p95_cm']:.1f}" if a["p95_cm"] is not None else "n/a"
                print(f"{pop:12s} {col:4s} {r['frames']:7d} "
                      f"{100.0*r['frames']/max(cell['population']['scoreable_frames'],1):5.0f}% | "
                      f"{r['position_recall_pct']:6.1f} {r['full6dof_recall_pct']:7.1f} {r['delta_pp']:6.1f} | "
                      f"{rmse:>8s} {med:>7s} {p95:>7s} | "
                      f"{r['pos_ok_but_wrong_branch_n']:7d} {r['pos_ok_but_ori_15_90_n']:5d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", action="append", required=True,
                    help="tag:CAPTURE_DIR:CSV_DIR (repeatable). CSV_DIR holds dev1.csv/dev2.csv")
    ap.add_argument("--cams", default=DEFAULT_CAMS)
    ap.add_argument("--ctrl-left", default=DEFAULT_CTRL_LEFT)
    ap.add_argument("--ctrl-right", default=DEFAULT_CTRL_RIGHT)
    ap.add_argument("--pos-tol-cm", type=float, default=POS_TOL_CM)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    cams = load_cameras(args.cams)
    all_cells = []
    for spec in args.cell:
        tag, cap, csv = spec.split(":")
        all_cells.extend(analyse_cell(
            tag, Path(cap), Path(csv), cams, args.ctrl_left, args.ctrl_right, args.pos_tol_cm))

    print_report(all_cells)

    # pooled rollup over all cells (FOV-edge population), per col
    print(f"\n{'='*60}\nPOOLED over all cells (FOV-edge population)\n{'='*60}")
    for col in ("opt", "pred"):
        denom = sum(c["scores"][col]["fov_edge"]["frames"] for c in all_cells)
        pos = sum(c["scores"][col]["fov_edge"]["position_correct_n"] for c in all_cells)
        full = sum(c["scores"][col]["fov_edge"]["full6dof_correct_n"] for c in all_cells)
        wb = sum(c["scores"][col]["fov_edge"]["pos_ok_but_wrong_branch_n"] for c in all_cells)
        print(f"  {col}: frames={denom}  position_recall={100.0*pos/max(denom,1):.1f}%  "
              f"full6dof_recall={100.0*full/max(denom,1):.1f}%  "
              f"delta=+{100.0*(pos-full)/max(denom,1):.1f}pp  pos_ok_wrong_branch={wb}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(all_cells, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
