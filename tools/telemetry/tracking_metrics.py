#!/usr/bin/env python3
"""Fixed-reference tracking metrics for G2 controller replay output.

This scorer is deliberately not self-referential: the reference trajectory,
frame grid, head pose, and visibility masks are built once from a reference
capture, then every candidate is scored against that exact denominator.

It also accounts for frames that cannot be scored by the cleaned optical
reference. Those frames are not silently excluded: they are reported by reason
and by blob-richness so visible dropouts do not disappear from the analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import DEVICE_NAMES, Manifest  # noqa: E402
from smooth_ref import build_reference  # noqa: E402
import g2_geom as G  # noqa: E402
from replay_contract import cams_for_capture  # noqa: E402
from detection_f1 import (  # noqa: E402
    DEFAULT_CTRL_LEFT,
    DEFAULT_CTRL_RIGHT,
    R_to_quat,
    _interp_quat_pos,
    in_fov_per_cam,
    load_cameras,
    load_led_model,
    pose_attempt_to_xrworld_device,
    quat_geodesic_deg,
    quat_to_R,
)


SEARCH_RESULT_NAMES = {
    0: "success",
    1: "no_searchable_anchors",
    2: "no_anchor_with_3_neighbours",
    3: "no_p3p_trials",
    4: "no_pose_checks",
    5: "all_pose_checks_pruned",
    6: "best_not_good",
    7: "no_good_candidate",
    8: "roi_skip_no_blobs",
    9: "roi_skip_no_model",
    10: "roi_skip_untrusted_prior",
    11: "roi_skip_invalid_prior",
    12: "roi_skip_visible_lt3",
    13: "roi_skip_blobs_lt4",
    14: "roi_skip_full_equiv",
}

BNG_REASON_NAMES = {
    1 << 0: "matched_lt3",
    1 << 1: "prior_position_fail",
    1 << 2: "prior_orient_fail",
    1 << 3: "led_ids_fail",
    1 << 4: "reproj_fail",
    1 << 5: "clean_cluster_fail",
    1 << 6: "visible_cover_fail",
    1 << 7: "minimal_prior_fail",
    1 << 8: "priorless_large_fail",
}

BLOBFIX_MIN_VALID_RUN_MS = 150.0

OBS_KIND_NAMES = {
    0: "none",
    1: "pose",
    2: "position_only",
    3: "led_fold",
}

XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT = 1 << 4
XRT_SPACE_RELATION_POSITION_TRACKED_BIT = 1 << 5


@dataclass
class FrameGrid:
    t_ns: np.ndarray
    frame_group_cams: np.ndarray
    blob_total: np.ndarray
    blob_max_cam: np.ndarray
    full_group: np.ndarray
    exposure: np.ndarray
    source_frame_groups: int = 0
    filtered_non_controller_groups: int = 0
    controller_exposure: int = 0


@dataclass
class ReferenceGrid:
    device: int
    frames: FrameGrid
    ref_pos: np.ndarray
    ref_quat: np.ndarray
    ref_valid: np.ndarray
    unknown_reason: np.ndarray
    head_pos: np.ndarray
    head_quat: np.ndarray
    head_valid: np.ndarray
    led_count: np.ndarray
    detect_visible: np.ndarray
    pose_visible: np.ndarray
    high_visible: np.ndarray
    gt_blobfix_provenance: dict[str, Any] | None = None

    @property
    def scoreable(self) -> np.ndarray:
        return self.ref_valid & self.head_valid


@dataclass
class CandidateStream:
    kind: str
    col: str
    t_ns: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    finite: np.ndarray
    valid: np.ndarray
    total_rows: int
    position_valid: np.ndarray | None = None
    tracked: np.ndarray | None = None
    relation_flags: np.ndarray | None = None
    fusion_state: np.ndarray | None = None
    last_optical_age_ms: np.ndarray | None = None
    drop_optical: np.ndarray | None = None
    obs_kind: np.ndarray | None = None
    obs_led_count: np.ndarray | None = None
    cam_id: np.ndarray | None = None
    source_path: str = ""


def _counter_dict(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in Counter(values.tolist()).items()}


def _unknown_reason_summary(grid: "ReferenceGrid", frame_has_accept: np.ndarray | None = None) -> dict[str, Any]:
    unknown = ~grid.scoreable
    reasons = sorted({str(x) for x in grid.unknown_reason[unknown].tolist()})
    out: dict[str, Any] = {}
    for reason in reasons:
        mask = unknown & (grid.unknown_reason == reason)
        row: dict[str, Any] = {
            "frames": int(mask.sum()),
            "blob_total_zero": int(np.sum(mask & (grid.frames.blob_total == 0))),
            "blob_total_1_3": int(np.sum(mask & (grid.frames.blob_total >= 1) & (grid.frames.blob_total <= 3))),
            "blob_total_4_7": int(np.sum(mask & (grid.frames.blob_total >= 4) & (grid.frames.blob_total <= 7))),
            "blob_total_ge8": int(np.sum(mask & (grid.frames.blob_total >= 8))),
            "full_4cam_groups": int(np.sum(mask & grid.frames.full_group)),
        }
        if frame_has_accept is not None:
            row["accepted"] = int(np.sum(mask & frame_has_accept))
            row["accepted_blob_total_ge4"] = int(np.sum(mask & (grid.frames.blob_total >= 4) & frame_has_accept))
            row["accepted_blob_total_ge8"] = int(np.sum(mask & (grid.frames.blob_total >= 8) & frame_has_accept))
        out[reason] = row
    return out


def _nearest_index(t_sorted: np.ndarray, t: int, max_dt_ns: int) -> int:
    j = int(np.searchsorted(t_sorted, t))
    best = -1
    for cand in (j - 1, j):
        if 0 <= cand < t_sorted.shape[0]:
            dt = abs(int(t_sorted[cand]) - int(t))
            if dt <= max_dt_ns and (best < 0 or dt < abs(int(t_sorted[best]) - int(t))):
                best = cand
    return best


def _nearest_indices(t_sorted: np.ndarray, t_values: np.ndarray, max_dt_ns: int) -> np.ndarray:
    out = np.full(t_values.shape[0], -1, dtype=np.int64)
    for i, t in enumerate(t_values):
        out[i] = _nearest_index(t_sorted, int(t), max_dt_ns)
    return out


def load_frame_grid(ref_telem: Path) -> FrameGrid:
    manifest = Manifest.load(ref_telem)
    frame = G.load_stream(ref_telem, manifest, "frame")
    t_all = frame["hw_ts_ns"].astype(np.int64)
    t_unique, inv = np.unique(t_all, return_inverse=True)
    cam_sets: list[set[int]] = [set() for _ in range(t_unique.shape[0])]
    exposure_sets: list[set[int]] = [set() for _ in range(t_unique.shape[0])]
    blob_total = np.zeros(t_unique.shape[0], dtype=np.int32)
    blob_max = np.zeros(t_unique.shape[0], dtype=np.int32)
    has_exposure = "exposure" in frame.dtype.names
    for i, group in enumerate(inv):
        cam_sets[int(group)].add(int(frame["cam_id"][i]))
        if has_exposure:
            exposure_sets[int(group)].add(int(frame["exposure"][i]))
        blobs = int(frame["n_blobs"][i])
        blob_total[int(group)] += blobs
        blob_max[int(group)] = max(blob_max[int(group)], blobs)
    n_cams = np.array([len(s) for s in cam_sets], dtype=np.int16)
    group_exposure = np.zeros(t_unique.shape[0], dtype=np.int32)
    controller_exposure = 0
    if has_exposure:
        uniform_full = sorted(
            int(next(iter(exps)))
            for exps, cams in zip(exposure_sets, n_cams)
            if cams == 4 and len(exps) == 1
        )
        if uniform_full:
            controller_exposure = int(uniform_full[0])
        for i, exps in enumerate(exposure_sets):
            group_exposure[i] = int(next(iter(exps))) if len(exps) == 1 else -1
    keep = np.ones(t_unique.shape[0], dtype=bool)
    if controller_exposure > 0:
        keep = group_exposure == controller_exposure
    return FrameGrid(
        t_ns=t_unique[keep],
        frame_group_cams=n_cams[keep],
        blob_total=blob_total[keep],
        blob_max_cam=blob_max[keep],
        full_group=(n_cams == 4)[keep],
        exposure=group_exposure[keep],
        source_frame_groups=int(t_unique.shape[0]),
        filtered_non_controller_groups=int(np.sum(~keep)),
        controller_exposure=controller_exposure,
    )


def _load_head_pose(telem: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    # Delegate to the trimmed loader (drops warm-up + post-crash-tail rows). Returning the
    # unpacked tuple keeps every existing caller's contract.
    from headpose_anchor import load_head_pose
    hp = load_head_pose(telem)
    if hp is None or hp.t_ns.size == 0:
        return None
    return hp.t_ns, hp.pos, hp.quat


def _interp_head(t_head: np.ndarray,
                 pos_head: np.ndarray,
                 quat_head: np.ndarray,
                 t_query: np.ndarray,
                 max_gap_ms: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, p = _interp_quat_pos(t_head, quat_head, pos_head, t_query)
    idx = _nearest_indices(t_head, t_query, int(max_gap_ms * 1e6))
    valid = idx >= 0
    p[~valid] = np.nan
    q[~valid] = np.nan
    return q, p, valid


def interpolate_reference_with_reasons(ref, t_query: np.ndarray, max_gap_ms: float):
    valid_idx = np.flatnonzero(ref.valid)
    out_q = np.full((t_query.shape[0], 4), np.nan)
    out_p = np.full((t_query.shape[0], 3), np.nan)
    valid = np.zeros(t_query.shape[0], dtype=bool)
    reason = np.full(t_query.shape[0], "invalid_ref_bracket", dtype=object)
    if valid_idx.shape[0] == 0:
        reason[:] = "no_reference"
        return out_q, out_p, valid, reason

    t_valid = ref.t_ns[valid_idx].astype(np.int64)
    p_valid = ref.pos[valid_idx]
    q_valid = ref.quat[valid_idx]
    max_gap_ns = int(max_gap_ms * 1e6)

    for k, t_raw in enumerate(t_query):
        t = int(t_raw)
        if t < int(t_valid[0]):
            reason[k] = "before_ref"
            continue
        if t > int(t_valid[-1]):
            reason[k] = "after_ref"
            continue
        hi = int(np.searchsorted(t_valid, t))
        if hi < t_valid.shape[0] and int(t_valid[hi]) == t:
            out_q[k] = q_valid[hi]
            out_p[k] = p_valid[hi]
            valid[k] = True
            reason[k] = "valid"
            continue
        lo = hi - 1
        if lo < 0 or hi >= t_valid.shape[0]:
            reason[k] = "invalid_ref_bracket"
            continue
        span = int(t_valid[hi]) - int(t_valid[lo])
        if span <= 0:
            reason[k] = "invalid_ref_bracket"
            continue
        if span > max_gap_ns:
            reason[k] = "ref_gap_gt_max"
            continue
        frac = (t - int(t_valid[lo])) / span
        out_p[k] = (1.0 - frac) * p_valid[lo] + frac * p_valid[hi]
        out_q[k] = G.quat_slerp(q_valid[lo], q_valid[hi], float(frac))
        valid[k] = True
        reason[k] = "valid"
    return out_q, out_p, valid, reason


def drop_short_valid_runs(t_ns: np.ndarray,
                          valid: np.ndarray,
                          reason: np.ndarray,
                          max_run_ms: float,
                          invalid_reason: str,
                          boundary_reason: str | None = None) -> int:
    dropped = 0
    i = 0
    n = valid.shape[0]
    while i < n:
        if not valid[i]:
            i += 1
            continue
        j = i + 1
        while j < n and valid[j]:
            j += 1
        left_invalid = i > 0 and not valid[i - 1]
        right_invalid = j < n and not valid[j]
        if boundary_reason is not None and left_invalid and right_invalid:
            left_invalid = str(reason[i - 1]) == boundary_reason
            right_invalid = str(reason[j]) == boundary_reason
        duration_ms = (int(t_ns[j - 1]) - int(t_ns[i])) / 1e6 if j > i else 0.0
        if left_invalid and right_invalid and duration_ms < max_run_ms:
            valid[i:j] = False
            reason[i:j] = invalid_reason
            dropped += j - i
        i = j
    return dropped


def build_reference_grid(ref_capture: Path,
                         dev: int,
                         cams,
                         ctrl_json: str,
                         max_ref_gap_ms: float,
                         head_match_ms: float,
                         detect_leds: int,
                         pose_leds: int,
                         high_leds: int,
                         reference_qc: str) -> ReferenceGrid:
    ref_telem = ref_capture / "telemetry"
    frames = load_frame_grid(ref_telem)
    ref = build_reference(ref_telem, dev, apply_blobfix=False)
    if ref is None:
        raise RuntimeError(f"device {dev}: cannot build reference from {ref_telem}")
    hp = _load_head_pose(ref_telem)
    if hp is None:
        raise RuntimeError(f"no head_pose stream in {ref_telem}")
    hp_t, hp_pos, hp_q = hp
    ref_q, ref_p, ref_valid, reason = interpolate_reference_with_reasons(ref, frames.t_ns, max_ref_gap_ms)
    gt_blobfix_provenance = None
    if reference_qc != "none":
        try:
            from gt_blob_fix import load_corrupt_mask, VERDICT_NAME, FIX_MIN_DQ_DEG, FIX_MIN_DP_CM
            cr = load_corrupt_mask(ref_capture, dev)
            if cr is None and ref_capture.name.endswith("-framebin"):
                cr = load_corrupt_mask(ref_capture.with_name(ref_capture.name[:-len("-framebin")]), dev)
        except Exception:
            cr = None
            VERDICT_NAME = {}
            FIX_MIN_DQ_DEG = 15.0
            FIX_MIN_DP_CM = 5.0
        if cr is not None and cr.t_ns.shape[0]:
            gt_blobfix_provenance = dict(getattr(cr, "provenance", {}) or {})
            gt_blobfix_provenance["stats"] = dict(getattr(cr, "stats", {}) or {})
            cr_t = cr.t_ns.astype(np.int64)
            cr_idx = _nearest_indices(cr_t, frames.t_ns, int(max_ref_gap_ms * 1e6))
            cr_hit = cr_idx >= 0
            if reference_qc == "corrupt":
                bad = np.zeros(frames.t_ns.shape[0], dtype=bool)
                bad[cr_hit] = cr.corrupt_mask[cr_idx[cr_hit]]
            else:
                good = np.zeros(frames.t_ns.shape[0], dtype=bool)
                hit_idx = cr_idx[cr_hit]
                fix_within_score_gate = (
                    cr.fix_confirmed[hit_idx] &
                    np.isfinite(cr.fix_dq_deg[hit_idx]) &
                    np.isfinite(cr.fix_dp_cm[hit_idx]) &
                    (cr.fix_dq_deg[hit_idx] <= FIX_MIN_DQ_DEG) &
                    (cr.fix_dp_cm[hit_idx] <= FIX_MIN_DP_CM)
                )
                good[cr_hit] = cr.gt_confirmed[hit_idx] | fix_within_score_gate
                bad = ref_valid & cr_hit & ~good
            if np.any(bad):
                for verdict_value in sorted({int(v) for v in cr.verdict[cr_idx[bad]].tolist()}):
                    verdict_name = VERDICT_NAME.get(verdict_value, str(verdict_value)).lower()
                    mask = np.zeros(frames.t_ns.shape[0], dtype=bool)
                    mask[bad] = cr.verdict[cr_idx[bad]] == verdict_value
                    reason[mask] = f"blobfix_{verdict_name}"
                ref_valid[bad] = False
            if reference_qc == "confirmed":
                drop_short_valid_runs(frames.t_ns, ref_valid, reason, BLOBFIX_MIN_VALID_RUN_MS,
                                      "blobfix_short_island", boundary_reason="blobfix_corrupt")
    head_q, head_p, head_valid = _interp_head(hp_t, hp_pos, hp_q, frames.t_ns, head_match_ms)
    reason[(reason == "valid") & ~head_valid] = "head_missing"

    led_count = np.zeros((frames.t_ns.shape[0], len(cams)), dtype=np.int32)
    scoreable = ref_valid & head_valid
    if np.any(scoreable):
        led_pos, led_nrm = load_led_model(ctrl_json)
        led_count[scoreable] = in_fov_per_cam(
            quat_to_R(ref_q[scoreable]),
            ref_p[scoreable],
            quat_to_R(head_q[scoreable]),
            head_p[scoreable],
            led_pos,
            led_nrm,
            cams,
        )
    max_leds = led_count.max(axis=1) if led_count.size else np.zeros(frames.t_ns.shape[0], dtype=np.int32)
    return ReferenceGrid(
        device=dev,
        frames=frames,
        ref_pos=ref_p,
        ref_quat=ref_q,
        ref_valid=ref_valid,
        unknown_reason=reason,
        head_pos=head_p,
        head_quat=head_q,
        head_valid=head_valid,
        led_count=led_count,
        detect_visible=scoreable & (max_leds >= detect_leds),
        pose_visible=scoreable & (max_leds >= pose_leds),
        high_visible=scoreable & (max_leds >= high_leds),
        gt_blobfix_provenance=gt_blobfix_provenance,
    )


def load_csv_stream(path: Path, col: str) -> CandidateStream:
    rows: list[dict[str, str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"empty CSV {path}")

    def f(row: dict[str, str], key: str) -> float:
        value = row.get(key)
        if value is None or value == "":
            return float("nan")
        try:
            return float(value)
        except ValueError:
            return float("nan")

    t = np.array([int(r["t_ns"]) for r in rows], dtype=np.int64)
    pos = np.array([[f(r, f"{col}_px"), f(r, f"{col}_py"), f(r, f"{col}_pz")] for r in rows])
    quat = np.array([[f(r, f"{col}_qx"), f(r, f"{col}_qy"), f(r, f"{col}_qz"), f(r, f"{col}_qw")]
                     for r in rows])
    pos_finite = np.isfinite(pos).all(axis=1)
    if col == "opt":
        tracked = np.array([int(f(r, "opt_valid") if np.isfinite(f(r, "opt_valid")) else 1) != 0 for r in rows],
                           dtype=bool)
        obs_kind = None
        obs_led_count = None
        if "opt_kind" in rows[0]:
            obs_kind = np.array([int(f(r, "opt_kind") if np.isfinite(f(r, "opt_kind")) else 0)
                                 for r in rows], dtype=np.int16)
        if "opt_led_count" in rows[0]:
            obs_led_count = np.array([int(f(r, "opt_led_count") if np.isfinite(f(r, "opt_led_count")) else 0)
                                      for r in rows], dtype=np.int16)
        if "opt_position_valid" in rows[0]:
            pos_tracked = np.array([int(f(r, "opt_position_valid") if np.isfinite(f(r, "opt_position_valid")) else 0) != 0
                                    for r in rows], dtype=bool)
            pos = np.array([[f(r, "opt_position_px"), f(r, "opt_position_py"), f(r, "opt_position_pz")]
                            if pos_tracked[i] else [pos[i, 0], pos[i, 1], pos[i, 2]]
                            for i, r in enumerate(rows)])
            pos_finite = np.isfinite(pos).all(axis=1)
            position_valid = pos_tracked & pos_finite
        else:
            position_valid = tracked & pos_finite
        relation_flags = None
        fusion_state = None
        last_optical_age_ms = None
    else:
        obs_kind = None
        obs_led_count = None
        pred_tracked = np.array([int(f(r, "pred_tracked") if np.isfinite(f(r, "pred_tracked")) else 1) != 0
                                 for r in rows], dtype=bool)
        has_relation_flags = "pred_flags" in rows[0]
        relation_flags = np.array([int(f(r, "pred_flags") if np.isfinite(f(r, "pred_flags")) else 0)
                                   for r in rows], dtype=np.uint64) if has_relation_flags else None
        if relation_flags is not None:
            position_tracked = (relation_flags & XRT_SPACE_RELATION_POSITION_TRACKED_BIT) != 0
            orientation_tracked = (relation_flags & XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT) != 0
            tracked = position_tracked & orientation_tracked
            position_valid = position_tracked & pos_finite
        else:
            tracked = pred_tracked
            position_valid = pred_tracked & pos_finite
        fusion_state = np.array([int(f(r, "fusion_state") if np.isfinite(f(r, "fusion_state")) else -1)
                                 for r in rows], dtype=np.int16)
        last_optical_age_ms = np.array([f(r, "last_optical_age_ms") for r in rows], dtype=float)
    drop_optical = None
    if "drop_optical" in rows[0]:
        drop_optical = np.array([int(f(r, "drop_optical") if np.isfinite(f(r, "drop_optical")) else 0) != 0
                                 for r in rows], dtype=bool)
    finite = np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat, axis=1) > 0.5)
    valid = tracked & finite
    quat[finite] = G.quat_normalize(quat[finite])
    return CandidateStream(
        kind="csv",
        col=col,
        t_ns=t,
        pos=pos,
        quat=quat,
        finite=finite,
        valid=valid,
        position_valid=position_valid,
        total_rows=len(rows),
        tracked=tracked,
        relation_flags=relation_flags,
        fusion_state=fusion_state,
        last_optical_age_ms=last_optical_age_ms,
        drop_optical=drop_optical,
        obs_kind=obs_kind,
        obs_led_count=obs_led_count,
        source_path=str(path),
    )


def load_pose_attempt_stream(telem: Path,
                             dev: int,
                             cams,
                             ref_head_t: np.ndarray,
                             ref_head_p: np.ndarray,
                             ref_head_q: np.ndarray) -> CandidateStream:
    manifest = Manifest.load(telem)
    pa = G.load_stream(telem, manifest, "pose_attempt")
    pa = pa[pa["device_id"] == dev]
    accepted = pa[(pa["outcome"] == 1) | (pa["outcome"] == 2)]
    t = accepted["hw_ts_ns"].astype(np.int64)
    pos = np.full((accepted.shape[0], 3), np.nan)
    quat = np.full((accepted.shape[0], 4), np.nan)
    valid = np.zeros(accepted.shape[0], dtype=bool)
    cam_id = accepted["cam_id"].astype(np.int16) if accepted.shape[0] else np.array([], dtype=np.int16)
    for i, row in enumerate(accepted):
        cam = int(row["cam_id"])
        if cam < 0 or cam >= len(cams):
            continue
        head_q, head_p = _interp_quat_pos(ref_head_t, ref_head_q, ref_head_p, np.array([int(t[i])], dtype=np.int64))
        if not np.isfinite(head_p[0, 0]):
            continue
        R_world, t_world = pose_attempt_to_xrworld_device(row, cams[cam], head_q[0], head_p[0])
        pos[i] = t_world
        quat[i] = R_to_quat(R_world)
        valid[i] = True
    return CandidateStream(
        kind="telemetry_pose_attempt",
        col="opt",
        t_ns=t,
        pos=pos,
        quat=quat,
        finite=valid.copy(),
        valid=valid,
        position_valid=valid.copy(),
        total_rows=int(pa.shape[0]),
        tracked=valid.copy(),
        cam_id=cam_id,
        source_path=str(telem),
    )


def _safe_pct(num: int | float, den: int | float) -> float:
    return 100.0 * float(num) / max(float(den), 1.0)


def _stat_block(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"median": None, "p75": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def score_stream(grid: ReferenceGrid,
                 candidate: CandidateStream,
                 visible_mask: np.ndarray,
                 match_ms: float,
                 pos_cm_thresh: float,
                 ori_deg_thresh: float) -> dict[str, Any]:
    t_grid = grid.frames.t_ns
    max_dt_ns = int(match_ms * 1e6)
    valid_samples = candidate.valid & np.isfinite(candidate.pos).all(axis=1) & np.isfinite(candidate.quat).all(axis=1)
    if candidate.position_valid is None:
        position_valid_samples = valid_samples.copy()
    else:
        position_valid_samples = candidate.position_valid & np.isfinite(candidate.pos).all(axis=1)

    sample_idx = np.flatnonzero(valid_samples)
    frame_idx = _nearest_indices(t_grid, candidate.t_ns[sample_idx], max_dt_ns)
    matched = frame_idx >= 0
    n_candidate_unmatched = int(np.sum(~matched))
    sample_idx = sample_idx[matched]
    frame_idx = frame_idx[matched]

    frame_has_accept = np.zeros(t_grid.shape[0], dtype=bool)
    frame_has_correct = np.zeros(t_grid.shape[0], dtype=bool)
    frame_has_wrong = np.zeros(t_grid.shape[0], dtype=bool)
    pos_err_cm_by_frame = np.full(t_grid.shape[0], np.nan)
    ori_err_deg_by_frame = np.full(t_grid.shape[0], np.nan)
    pos_errors = []
    ori_errors = []

    scoreable = grid.scoreable
    for si, fi in zip(sample_idx, frame_idx):
        frame_has_accept[fi] = True
        if not scoreable[fi]:
            continue
        pos_err_cm = float(np.linalg.norm(candidate.pos[si] - grid.ref_pos[fi]) * 100.0)
        ori_err_deg = float(quat_geodesic_deg(candidate.quat[si], grid.ref_quat[fi]))
        correct = pos_err_cm < pos_cm_thresh and ori_err_deg < ori_deg_thresh
        if np.isnan(pos_err_cm_by_frame[fi]) or pos_err_cm < pos_err_cm_by_frame[fi]:
            pos_err_cm_by_frame[fi] = pos_err_cm
            ori_err_deg_by_frame[fi] = ori_err_deg
        pos_errors.append(pos_err_cm)
        ori_errors.append(ori_err_deg)
        if correct:
            frame_has_correct[fi] = True
        elif ori_err_deg >= 90.0:
            frame_has_wrong[fi] = True

    pos_sample_idx = np.flatnonzero(position_valid_samples)
    pos_frame_idx = _nearest_indices(t_grid, candidate.t_ns[pos_sample_idx], max_dt_ns)
    pos_matched = pos_frame_idx >= 0
    pos_sample_idx = pos_sample_idx[pos_matched]
    pos_frame_idx = pos_frame_idx[pos_matched]

    frame_has_position_accept = np.zeros(t_grid.shape[0], dtype=bool)
    frame_has_position_correct = np.zeros(t_grid.shape[0], dtype=bool)
    position_err_cm_by_frame = np.full(t_grid.shape[0], np.nan)
    position_errors = []

    for si, fi in zip(pos_sample_idx, pos_frame_idx):
        frame_has_position_accept[fi] = True
        if not scoreable[fi]:
            continue
        pos_err_cm = float(np.linalg.norm(candidate.pos[si] - grid.ref_pos[fi]) * 100.0)
        if np.isnan(position_err_cm_by_frame[fi]) or pos_err_cm < position_err_cm_by_frame[fi]:
            position_err_cm_by_frame[fi] = pos_err_cm
        position_errors.append(pos_err_cm)
        if pos_err_cm < pos_cm_thresh:
            frame_has_position_correct[fi] = True

    frame_obs_pose = np.zeros(t_grid.shape[0], dtype=bool)
    frame_obs_position_only = np.zeros(t_grid.shape[0], dtype=bool)
    frame_obs_led_fold = np.zeros(t_grid.shape[0], dtype=bool)
    obs_kind_row_counts: dict[str, int] = {}
    if candidate.obs_kind is not None:
        obs_kinds = candidate.obs_kind.astype(np.int16)
        obs_kind_row_counts = {
            OBS_KIND_NAMES.get(int(k), f"kind_{int(k)}"): int(v)
            for k, v in Counter(obs_kinds.tolist()).items()
        }
        obs_frame_idx = _nearest_indices(t_grid, candidate.t_ns, max_dt_ns)
        obs_matched = obs_frame_idx >= 0
        for kind, fi in zip(obs_kinds[obs_matched], obs_frame_idx[obs_matched]):
            if int(kind) == 1:
                frame_obs_pose[fi] = True
            elif int(kind) == 2:
                frame_obs_position_only[fi] = True
            elif int(kind) == 3:
                frame_obs_led_fold[fi] = True

    scored = scoreable
    in_view = visible_mask & scored
    TP = int(np.sum(in_view & frame_has_correct))
    FN = int(np.sum(in_view & ~frame_has_correct))
    FP = int(np.sum(scored & frame_has_accept & ~frame_has_correct))
    TN = int(np.sum(scored & ~in_view & ~frame_has_accept))
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    pos_arr = np.array(pos_errors, dtype=float)
    ori_arr = np.array(ori_errors, dtype=float)
    position_arr = np.array(position_errors, dtype=float)
    position_TP = int(np.sum(in_view & frame_has_position_correct))
    position_FN = int(np.sum(in_view & ~frame_has_position_correct))
    position_FP = int(np.sum(scored & frame_has_position_accept & ~frame_has_position_correct))
    position_precision = position_TP / max(position_TP + position_FP, 1)
    position_recall = position_TP / max(position_TP + position_FN, 1)
    position_f1 = 2 * position_precision * position_recall / max(position_precision + position_recall, 1e-12)
    unknown = ~scoreable
    unknown_blob_rich = unknown & (grid.frames.blob_total >= 4)
    accepted_unknown = int(np.sum(unknown & frame_has_accept))
    unknown_blob4_accept = int(np.sum(unknown_blob_rich & frame_has_accept))
    unknown_by_reason = _unknown_reason_summary(grid, frame_has_accept)
    visible_no_pose_with_position = int(np.sum(in_view & ~frame_has_accept & frame_has_position_accept))
    visible_no_pose_with_led_fold = int(np.sum(in_view & ~frame_has_accept & frame_obs_led_fold))
    unknown_position_only = int(np.sum(unknown & frame_obs_position_only))
    unknown_led_fold = int(np.sum(unknown & frame_obs_led_fold))

    return {
        "kind": candidate.kind,
        "col": candidate.col,
        "source_path": candidate.source_path,
        "n_candidate_rows": int(candidate.total_rows),
        "n_candidate_valid_samples": int(valid_samples.sum()),
        "n_candidate_unmatched_samples": n_candidate_unmatched,
        "candidate_unmatched_pct_valid": _safe_pct(n_candidate_unmatched, int(valid_samples.sum())),
        "n_candidate_position_valid_samples": int(position_valid_samples.sum()),
        "n_candidate_matched_to_frame": int(sample_idx.shape[0]),
        "n_candidate_position_matched_to_frame": int(pos_sample_idx.shape[0]),
        "obs_kind_row_counts": obs_kind_row_counts,
        "n_scoreable_frames": int(scored.sum()),
        "n_visible_frames": int(in_view.sum()),
        "n_unknown_frames": int(unknown.sum()),
        "accepted_unknown_frames": accepted_unknown,
        "accepted_unknown_pct_unknown": _safe_pct(accepted_unknown, int(unknown.sum())),
        "accepted_blob_rich_unknown_frames": unknown_blob4_accept,
        "accepted_blob_rich_unknown_pct_blob_rich_unknown": _safe_pct(unknown_blob4_accept,
                                                                      int(unknown_blob_rich.sum())),
        "unknown_by_reason": unknown_by_reason,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "position_TP": position_TP,
        "position_FP": position_FP,
        "position_FN": position_FN,
        "position_precision": position_precision,
        "position_recall": position_recall,
        "position_f1": position_f1,
        "visible_no_accept": int(np.sum(in_view & ~frame_has_accept)),
        "visible_no_position_accept": int(np.sum(in_view & ~frame_has_position_accept)),
        "visible_no_pose_with_position": visible_no_pose_with_position,
        "visible_no_pose_with_led_fold": visible_no_pose_with_led_fold,
        "visible_position_only_observation_frames": int(np.sum(in_view & frame_obs_position_only)),
        "visible_led_fold_observation_frames": int(np.sum(in_view & frame_obs_led_fold)),
        "unknown_position_only_observation_frames": unknown_position_only,
        "unknown_led_fold_observation_frames": unknown_led_fold,
        "visible_wrong_accept": int(np.sum(in_view & frame_has_accept & ~frame_has_correct)),
        "visible_wrong_position_accept": int(np.sum(in_view & frame_has_position_accept & ~frame_has_position_correct)),
        "wrong_branch_frames": int(np.sum(scored & frame_has_wrong)),
        "wrong_branch_pct_scored": _safe_pct(int(np.sum(scored & frame_has_wrong)), int(scored.sum())),
        "yield_pct_frame_rows": _safe_pct(int(valid_samples.sum()), int(candidate.total_rows)),
        "position_yield_pct_frame_rows": _safe_pct(int(position_valid_samples.sum()), int(candidate.total_rows)),
        "pos_rmse_cm": float(np.sqrt(np.mean(pos_arr * pos_arr))) if pos_arr.size else None,
        "pos_median_cm": float(np.median(pos_arr)) if pos_arr.size else None,
        "pos_p95_cm": float(np.percentile(pos_arr, 95)) if pos_arr.size else None,
        "position_rmse_cm": float(np.sqrt(np.mean(position_arr * position_arr))) if position_arr.size else None,
        "position_median_cm": float(np.median(position_arr)) if position_arr.size else None,
        "position_p95_cm": float(np.percentile(position_arr, 95)) if position_arr.size else None,
        "ori_rms_deg": float(np.sqrt(np.mean(ori_arr * ori_arr))) if ori_arr.size else None,
        "ori_median_deg": float(np.median(ori_arr)) if ori_arr.size else None,
        "ori_p95_deg": float(np.percentile(ori_arr, 95)) if ori_arr.size else None,
        "_frame_has_accept": frame_has_accept,
        "_frame_has_correct": frame_has_correct,
        "_frame_has_position_accept": frame_has_position_accept,
        "_frame_has_position_correct": frame_has_position_correct,
        "_pos_err_cm_by_frame": pos_err_cm_by_frame,
        "_position_err_cm_by_frame": position_err_cm_by_frame,
    }


def summarize_reference(grid: ReferenceGrid) -> dict[str, Any]:
    scoreable = grid.scoreable
    unknown = ~scoreable
    raw_observable_blob_ge4 = grid.frames.blob_total >= 4
    unknown_blob_rich = np.flatnonzero(unknown & (grid.frames.blob_total >= 4))
    unknown_samples = [
        {
            "frame_index": int(i),
            "t_ns": int(grid.frames.t_ns[i]),
            "reason": str(grid.unknown_reason[i]),
            "blob_total": int(grid.frames.blob_total[i]),
            "blob_max_cam": int(grid.frames.blob_max_cam[i]),
            "full_4cam": bool(grid.frames.full_group[i]),
        }
        for i in unknown_blob_rich[:32]
    ]
    return {
        "total_frame_groups": int(grid.frames.t_ns.shape[0]),
        "source_frame_groups": int(grid.frames.source_frame_groups),
        "controller_exposure": int(grid.frames.controller_exposure),
        "filtered_non_controller_groups": int(grid.frames.filtered_non_controller_groups),
        "full_4cam_groups": int(grid.frames.full_group.sum()),
        "non_4cam_groups": int((~grid.frames.full_group).sum()),
        "scoreable_frames": int(scoreable.sum()),
        "scoreable_pct_total": _safe_pct(int(scoreable.sum()), int(grid.frames.t_ns.shape[0])),
        "unknown_frames": int(unknown.sum()),
        "unknown_pct_total": _safe_pct(int(unknown.sum()), int(grid.frames.t_ns.shape[0])),
        "unknown_reasons": _counter_dict(grid.unknown_reason[unknown]),
        "unknown_by_reason": _unknown_reason_summary(grid),
        "unknown_blob_total_ge4": int(np.sum(unknown & (grid.frames.blob_total >= 4))),
        "unknown_blob_ge4_pct_total": _safe_pct(int(np.sum(unknown & (grid.frames.blob_total >= 4))),
                                                int(grid.frames.t_ns.shape[0])),
        "unknown_blob_total_ge8": int(np.sum(unknown & (grid.frames.blob_total >= 8))),
        "unknown_blob_ge8_pct_total": _safe_pct(int(np.sum(unknown & (grid.frames.blob_total >= 8))),
                                                int(grid.frames.t_ns.shape[0])),
        "unknown_blob_total_zero": int(np.sum(unknown & (grid.frames.blob_total == 0))),
        "raw_observable_blob_ge4_frames": int(raw_observable_blob_ge4.sum()),
        "raw_observable_blob_ge4_unscored_frames": int(np.sum(raw_observable_blob_ge4 & unknown)),
        "raw_observable_blob_ge4_scoreable_pct": _safe_pct(int(np.sum(raw_observable_blob_ge4 & scoreable)),
                                                           int(raw_observable_blob_ge4.sum())),
        "unknown_blob_rich_samples": unknown_samples,
        "gt_blobfix_provenance": grid.gt_blobfix_provenance,
        "detect_visible_frames": int(grid.detect_visible.sum()),
        "pose_visible_frames": int(grid.pose_visible.sum()),
        "high_visible_frames": int(grid.high_visible.sum()),
        "blob_total_p50": float(np.percentile(grid.frames.blob_total, 50)),
        "blob_total_p75": float(np.percentile(grid.frames.blob_total, 75)),
        "blob_total_p95": float(np.percentile(grid.frames.blob_total, 95)),
    }


def load_candidate_for_path(candidate_path: Path,
                            dev: int,
                            col: str,
                            cams,
                            grid: ReferenceGrid) -> CandidateStream:
    if candidate_path.is_dir() and (candidate_path / f"dev{dev}.csv").is_file():
        return load_csv_stream(candidate_path / f"dev{dev}.csv", col)
    if candidate_path.is_file() and candidate_path.suffix == ".csv":
        return load_csv_stream(candidate_path, col)
    telem = candidate_path / "telemetry" if (candidate_path / "telemetry").is_dir() else candidate_path
    if telem.is_dir() and (telem / "manifest.json").is_file():
        return load_pose_attempt_stream(telem, dev, cams, grid.frames.t_ns, grid.head_pos, grid.head_quat)
    raise RuntimeError(f"cannot infer candidate type from {candidate_path}")


def _candidate_telemetry_dir(candidate_path: Path) -> Path | None:
    """Find the telemetry directory associated with a candidate.

    Replays are commonly passed to this scorer as either:
      - the replay root, containing telemetry/ plus csv/dev*.csv
      - the csv/ subdirectory itself
      - a telemetry directory directly

    Search/candidate diagnostics must still be available in the csv/ case, or
    A/B reports hide the exact candidate-generation failures we are trying to
    measure.
    """
    candidates = []
    if candidate_path.is_dir():
        candidates.extend([
            candidate_path / "telemetry",
            candidate_path,
            candidate_path.parent / "telemetry",
        ])
    else:
        candidates.extend([
            candidate_path.parent / "telemetry",
            candidate_path.parent.parent / "telemetry",
        ])
    for telem in candidates:
        if (telem / "manifest.json").is_file():
            return telem
    return None


def summarize_search(candidate_path: Path,
                     grid: ReferenceGrid,
                     score: dict[str, Any],
                     match_ms: float) -> dict[str, Any] | None:
    telem = _candidate_telemetry_dir(candidate_path)
    if telem is None:
        return None
    manifest = Manifest.load(telem)
    if "search" not in manifest.streams:
        return None
    search = G.load_stream(telem, manifest, "search")
    search = search[search["device_id"] == grid.device]
    if search.shape[0] == 0:
        return {"rows": 0}
    frame_idx = _nearest_indices(grid.frames.t_ns, search["t_mono_ns"].astype(np.int64), int(match_ms * 1e6))
    ok = frame_idx >= 0
    search = search[ok]
    frame_idx = frame_idx[ok]
    result_names = np.array([SEARCH_RESULT_NAMES.get(int(x), f"result_{int(x)}") for x in search["result"]], dtype=object)
    visible_no_accept = grid.detect_visible & ~score["_frame_has_accept"]
    unknown_blob4 = (~grid.scoreable) & (grid.frames.blob_total >= 4)
    out: dict[str, Any] = {
        "rows": int(search.shape[0]),
        "result_counts_all": _counter_dict(result_names),
    }
    for label, mask in (("visible_no_accept", visible_no_accept), ("unknown_blob_ge4", unknown_blob4)):
        rows = np.flatnonzero(mask[frame_idx])
        out[f"result_counts_{label}"] = _counter_dict(result_names[rows]) if rows.size else {}
        if rows.size:
            out[f"{label}_median_input_blobs"] = float(np.median(search["input_blobs"][rows]))
            out[f"{label}_median_searchable_anchors"] = float(np.median(search["searchable_anchors"][rows]))
            out[f"{label}_median_pose_checks"] = float(np.median(search["num_pose_checks"][rows]))
            out[f"{label}_median_pose_checks_pruned"] = float(np.median(search["num_pose_checks_pruned"][rows]))
    return out


def summarize_unknown_adjudication(candidate_path: Path,
                                   grid: ReferenceGrid,
                                   score: dict[str, Any],
                                   pred_stream: CandidateStream | None,
                                   match_ms: float) -> dict[str, Any]:
    unknown = ~grid.scoreable
    blob0 = unknown & (grid.frames.blob_total == 0)
    blob1_3 = unknown & (grid.frames.blob_total >= 1) & (grid.frames.blob_total <= 3)
    blob4 = unknown & (grid.frames.blob_total >= 4)
    frame_has_accept = score["_frame_has_accept"]
    frame_has_position_accept = score["_frame_has_position_accept"]
    forced_drop = np.zeros(grid.frames.t_ns.shape[0], dtype=bool)
    if pred_stream is not None and pred_stream.drop_optical is not None:
        forced_idx = np.flatnonzero(pred_stream.drop_optical)
        frame_idx = _nearest_indices(grid.frames.t_ns, pred_stream.t_ns[forced_idx], int(match_ms * 1e6))
        for fi in frame_idx[frame_idx >= 0]:
            forced_drop[fi] = True

    n_frames = grid.frames.t_ns.shape[0]
    has_search = np.zeros(n_frames, dtype=bool)
    search_success = np.zeros(n_frames, dtype=bool)
    search_best_not_good = np.zeros(n_frames, dtype=bool)
    search_bng_reason_flags = np.zeros(n_frames, dtype=np.uint32)
    telem = _candidate_telemetry_dir(candidate_path)
    if telem is not None:
        manifest = Manifest.load(telem)
        if "search" in manifest.streams:
            search = G.load_stream(telem, manifest, "search")
            search = search[search["device_id"] == grid.device]
            frame_idx = _nearest_indices(grid.frames.t_ns, search["t_mono_ns"].astype(np.int64), int(match_ms * 1e6))
            keep = frame_idx >= 0
            have_bng_reasons = "bng_reason_flags" in search.dtype.names
            reason_values = search["bng_reason_flags"][keep] if have_bng_reasons else np.zeros(int(np.sum(keep)),
                                                                                               dtype=np.uint32)
            for result, reason_flags, fi in zip(search["result"][keep], reason_values, frame_idx[keep]):
                has_search[fi] = True
                if int(result) == 0:
                    search_success[fi] = True
                elif int(result) == 6:
                    search_best_not_good[fi] = True
                    search_bng_reason_flags[fi] |= np.uint32(int(reason_flags))

    classes: dict[str, np.ndarray] = {
        "unobservable_zero_blob": blob0,
        "sparse_raw_blobs_1_3": blob1_3,
        "reference_gap_tracker_pose": blob4 & frame_has_accept,
        "reference_gap_tracker_position": blob4 & ~frame_has_accept & frame_has_position_accept,
        "reference_gap_search_success_no_commit": blob4 & ~frame_has_accept & ~frame_has_position_accept & search_success,
        "blob_rich_search_best_not_good": blob4 & ~frame_has_accept & ~frame_has_position_accept &
        ~search_success & search_best_not_good,
        "blob_rich_no_success_other": blob4 & ~frame_has_accept & ~frame_has_position_accept &
        ~search_success & has_search & ~search_best_not_good,
        "blob_rich_forced_drop_no_search": blob4 & ~frame_has_accept & ~frame_has_position_accept &
        ~has_search & forced_drop,
        "blob_rich_no_search": blob4 & ~frame_has_accept & ~frame_has_position_accept & ~has_search &
        ~forced_drop,
    }

    by_class = {name: int(mask.sum()) for name, mask in classes.items()}
    by_class_pct_unknown = {name: _safe_pct(count, int(unknown.sum())) for name, count in by_class.items()}
    bng_mask = classes["blob_rich_search_best_not_good"]
    bng_reason_counts = {
        name: int(np.sum(bng_mask & ((search_bng_reason_flags & np.uint32(bit)) != 0)))
        for bit, name in BNG_REASON_NAMES.items()
    }
    bng_reason_counts["reason_unset"] = int(np.sum(bng_mask & (search_bng_reason_flags == 0)))
    by_reason: dict[str, dict[str, int]] = {}
    for reason in sorted({str(x) for x in grid.unknown_reason[unknown].tolist()}):
        rmask = unknown & (grid.unknown_reason == reason)
        by_reason[reason] = {name: int(np.sum(mask & rmask)) for name, mask in classes.items()}

    explained = (
        classes["unobservable_zero_blob"] |
        classes["sparse_raw_blobs_1_3"] |
        classes["reference_gap_tracker_pose"] |
        classes["reference_gap_tracker_position"] |
        classes["reference_gap_search_success_no_commit"] |
        classes["blob_rich_forced_drop_no_search"]
    )
    tracker_unresolved = (
        classes["blob_rich_search_best_not_good"] |
        classes["blob_rich_no_success_other"] |
        classes["blob_rich_no_search"]
    )
    return {
        "unknown_frames": int(unknown.sum()),
        "blob_rich_unknown_frames": int(blob4.sum()),
        "explained_or_reference_gap": int(np.sum(unknown & explained)),
        "explained_or_reference_gap_pct_unknown": _safe_pct(int(np.sum(unknown & explained)), int(unknown.sum())),
        "tracker_unresolved_blob_rich": int(np.sum(unknown & tracker_unresolved)),
        "tracker_unresolved_blob_rich_pct_blob_rich_unknown": _safe_pct(int(np.sum(unknown & tracker_unresolved)),
                                                                        int(blob4.sum())),
        "tracker_unresolved_blob_rich_pct_unknown": _safe_pct(int(np.sum(unknown & tracker_unresolved)),
                                                             int(unknown.sum())),
        "by_class": by_class,
        "by_class_pct_unknown": by_class_pct_unknown,
        "bng_reason_counts": bng_reason_counts,
        "by_reason": by_reason,
    }


def _scored_candidate_rows(candidate_path: Path,
                           grid: ReferenceGrid,
                           cams,
                           match_ms: float,
                           pos_cm_thresh: float,
                           ori_deg_thresh: float) -> tuple[np.ndarray | None, list[dict[str, Any]] | None]:
    telem = _candidate_telemetry_dir(candidate_path)
    if telem is None:
        return None, None
    manifest = Manifest.load(telem)
    if "candidate" not in manifest.streams:
        return None, None
    rows = G.load_stream(telem, manifest, "candidate")
    rows = rows[rows["device_id"] == grid.device]
    if rows.shape[0] == 0:
        return rows, []

    row_fields = set(rows.dtype.names or ())

    def field_float(row, name: str) -> float:
        return float(row[name]) if name in row_fields else float("nan")

    matched_rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        frame = _nearest_index(grid.frames.t_ns, int(row["t_mono_ns"]), int(match_ms * 1e6))
        if frame < 0 or not grid.scoreable[frame]:
            continue
        cam = int(row["cam_id"])
        if cam < 0 or cam >= len(cams):
            continue
        R_world, t_world = pose_attempt_to_xrworld_device(row, cams[cam], grid.head_quat[frame], grid.head_pos[frame])
        q_world = R_to_quat(R_world)
        pos_err_cm = float(np.linalg.norm(t_world - grid.ref_pos[frame]) * 100.0)
        ori_err_deg = float(quat_geodesic_deg(q_world, grid.ref_quat[frame]))
        position_correct = pos_err_cm < pos_cm_thresh
        matched_rows.append({
            "row_index": row_index,
            "frame": int(frame),
            "t_ns": int(grid.frames.t_ns[frame]),
            "stage": int(row["stage"]),
            "outcome": int(row["outcome"]),
            "selected": int(row["selected"]),
            "cam": cam,
            "match_flags": int(row["match_flags"]),
            "blobs_matched": int(row["blobs_matched"]),
            "leds_visible": int(row["leds_visible"]),
            "unmatched_blobs": int(row["unmatched_blobs"]),
            "reproj_err_px": float(row["reproj_err_px"]),
            "prior_pos_abs_max": float(max(abs(float(row["prior_pos_err_x"])),
                                           abs(float(row["prior_pos_err_y"])),
                                           abs(float(row["prior_pos_err_z"])))),
            "prior_rot_abs_max_deg": float(np.degrees(max(abs(float(row["prior_rot_err_x"])),
                                                       abs(float(row["prior_rot_err_y"])),
                                                       abs(float(row["prior_rot_err_z"]))))),
            "blob_var_mean_px2": field_float(row, "blob_var_mean_px2"),
            "blob_brightness_mean": field_float(row, "blob_brightness_mean"),
            "blob_area_mean": field_float(row, "blob_area_mean"),
            "pos_err_cm": pos_err_cm,
            "ori_err_deg": ori_err_deg,
            "position_correct": position_correct,
            "correct": position_correct and ori_err_deg < ori_deg_thresh,
            "wrong_branch": ori_err_deg >= 90.0,
        })
    return rows, matched_rows


def summarize_candidate_rows(candidate_path: Path,
                             grid: ReferenceGrid,
                             cams,
                             match_ms: float,
                             pos_cm_thresh: float,
                             ori_deg_thresh: float) -> dict[str, Any] | None:
    rows, matched_rows = _scored_candidate_rows(candidate_path, grid, cams, match_ms, pos_cm_thresh, ori_deg_thresh)
    if rows is None or matched_rows is None:
        return None
    if rows.shape[0] == 0:
        return {"rows": 0}

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            return {"rows": 0}
        pos = np.array([x["pos_err_cm"] for x in items])
        ori = np.array([x["ori_err_deg"] for x in items])
        matched = np.array([x["blobs_matched"] for x in items])
        visible = np.array([x["leds_visible"] for x in items])
        unmatched = np.array([x["unmatched_blobs"] for x in items])
        reproj = np.array([x["reproj_err_px"] for x in items])
        prior_pos = np.array([x["prior_pos_abs_max"] for x in items])
        prior_rot = np.array([x["prior_rot_abs_max_deg"] for x in items])
        blob_var = np.array([x["blob_var_mean_px2"] for x in items])
        blob_brightness = np.array([x["blob_brightness_mean"] for x in items])
        blob_area = np.array([x["blob_area_mean"] for x in items])
        flags = Counter(hex(x["match_flags"]) for x in items)
        return {
            "rows": len(items),
            "correct": int(sum(x["correct"] for x in items)),
            "correct_pct": _safe_pct(sum(x["correct"] for x in items), len(items)),
            "position_correct": int(sum(x["position_correct"] for x in items)),
            "position_correct_pct": _safe_pct(sum(x["position_correct"] for x in items), len(items)),
            "wrong_branch": int(sum(x["wrong_branch"] for x in items)),
            "wrong_branch_pct": _safe_pct(sum(x["wrong_branch"] for x in items), len(items)),
            "pos_median_cm": float(np.median(pos)),
            "pos_p95_cm": float(np.percentile(pos, 95)),
            "ori_median_deg": float(np.median(ori)),
            "ori_p95_deg": float(np.percentile(ori, 95)),
            "matched_median": float(np.median(matched)),
            "visible_median": float(np.median(visible)),
            "unmatched_median": float(np.median(unmatched)),
            "reproj_median_px": float(np.median(reproj)),
            "prior_pos_abs_max_median_m": float(np.median(prior_pos)),
            "prior_rot_abs_max_median_deg": float(np.median(prior_rot)),
            "blob_var_mean_px2": _stat_block(blob_var),
            "blob_brightness_mean": _stat_block(blob_brightness),
            "blob_area_mean": _stat_block(blob_area),
            "flag_counts": dict(flags.most_common(12)),
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in matched_rows:
        stage = STAGE_NAMES_FOR_OUTPUT.get(item["stage"], str(item["stage"]))
        key = f"{stage}:selected{item['selected']}:outcome{item['outcome']}"
        groups[key].append(item)

    cold_search_unselected = [
        item for item in matched_rows
        if item["stage"] == 6 and item["outcome"] == 0 and item["selected"] == 0
    ]
    return {
        "rows": int(rows.shape[0]),
        "matched_scoreable_rows": len(matched_rows),
        "groups": {key: summarize_group(value) for key, value in sorted(groups.items())},
        "cold_search_unselected": summarize_group(cold_search_unselected),
    }


def summarize_candidate_frame_gaps(candidate_path: Path,
                                   grid: ReferenceGrid,
                                   cams,
                                   score: dict[str, Any],
                                   visible_mask: np.ndarray,
                                   match_ms: float,
                                   pos_cm_thresh: float,
                                   ori_deg_thresh: float) -> dict[str, Any] | None:
    rows, matched_rows = _scored_candidate_rows(candidate_path, grid, cams, match_ms, pos_cm_thresh, ori_deg_thresh)
    if rows is None or matched_rows is None:
        return None
    if rows.shape[0] == 0:
        return {"candidate_rows": 0}

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in matched_rows:
        by_frame[item["frame"]].append(item)

    scoreable_visible = grid.scoreable & visible_mask
    frame_has_correct = score["_frame_has_correct"]
    frame_has_accept = score["_frame_has_accept"]
    failure_frames = np.flatnonzero(scoreable_visible & ~frame_has_correct)
    no_accept_frames = np.flatnonzero(scoreable_visible & ~frame_has_accept)
    wrong_accept_frames = np.flatnonzero(scoreable_visible & frame_has_accept & ~frame_has_correct)

    def has_any(frame: int, predicate) -> bool:
        return any(predicate(item) for item in by_frame.get(int(frame), []))

    def summarize_frames(frames: np.ndarray) -> dict[str, Any]:
        correct_alt = [int(f) for f in frames if has_any(int(f), lambda x: x["correct"])]
        pos_alt = [int(f) for f in frames if has_any(int(f), lambda x: x["position_correct"])]
        pos_only_alt = [int(f) for f in frames if has_any(int(f), lambda x: x["position_correct"] and not x["correct"])]
        generated_any = [int(f) for f in frames if by_frame.get(int(f))]
        selected_wrong_branch = [
            int(f) for f in frames
            if has_any(int(f), lambda x: x["selected"] != 0 and x["outcome"] == 1 and x["wrong_branch"])
        ]
        stage_correct: Counter[str] = Counter()
        stage_position_correct: Counter[str] = Counter()
        for f in frames:
            for item in by_frame.get(int(f), []):
                stage = STAGE_NAMES_FOR_OUTPUT.get(item["stage"], str(item["stage"]))
                if item["correct"]:
                    stage_correct[stage] += 1
                if item["position_correct"]:
                    stage_position_correct[stage] += 1
        return {
            "frames": int(frames.size),
            "any_candidate_generated": len(generated_any),
            "correct_alt_exists": len(correct_alt),
            "position_correct_alt_exists": len(pos_alt),
            "position_only_alt_exists": len(pos_only_alt),
            "no_position_correct_alt": int(frames.size) - len(pos_alt),
            "selected_wrong_branch": len(selected_wrong_branch),
            "correct_alt_by_source": dict(stage_correct),
            "position_correct_alt_by_source": dict(stage_position_correct),
        }

    return {
        "candidate_rows": int(rows.shape[0]),
        "matched_scoreable_rows": len(matched_rows),
        "scoreable_visible_frames": int(scoreable_visible.sum()),
        "selected_correct_frames": int(np.sum(scoreable_visible & frame_has_correct)),
        "failure_frames": summarize_frames(failure_frames),
        "visible_no_accept_frames": summarize_frames(no_accept_frames),
        "visible_wrong_accept_frames": summarize_frames(wrong_accept_frames),
    }


STAGE_NAMES_FOR_OUTPUT = {
    0: "absent",
    1: "prior_pose",
    2: "last_seen",
    3: "labelled_pnp",
    4: "prior_labelled_pnp",
    5: "joint_pnp",
    6: "cold_search",
    7: "triangulation",
}

FUSION_STATE_NAMES = {
    -1: "missing",
    0: "Invalid",
    1: "VisualAccuracy",
    2: "InertialFastMotion",
    3: "WorldLocked",
    4: "BodyLocked",
    5: "ConfusedPosition",
}


def fusion_diagnostics(pred_stream: CandidateStream | None) -> dict[str, Any] | None:
    if pred_stream is None or pred_stream.fusion_state is None:
        return None
    states = pred_stream.fusion_state
    tracked = pred_stream.tracked if pred_stream.tracked is not None else pred_stream.valid
    position_tracked = pred_stream.position_valid if pred_stream.position_valid is not None else tracked
    age = pred_stream.last_optical_age_ms
    by_state: dict[str, dict[str, int]] = {}
    for value in sorted({int(x) for x in states.tolist()}):
        mask = states == value
        by_state[FUSION_STATE_NAMES.get(value, str(value))] = {
            "frames": int(mask.sum()),
            "tracked": int(np.sum(mask & tracked)),
            "position_tracked": int(np.sum(mask & position_tracked)),
        }

    out: dict[str, Any] = {
        "rows": int(states.shape[0]),
        "tracked": int(np.sum(tracked)),
        "position_tracked": int(np.sum(position_tracked)),
        "by_state": by_state,
    }
    if age is not None:
        finite = np.isfinite(age)
        out["last_optical_age_ms"] = {
            "finite": int(finite.sum()),
            "median": float(np.median(age[finite])) if np.any(finite) else None,
            "p95": float(np.percentile(age[finite], 95)) if np.any(finite) else None,
            "max": float(np.max(age[finite])) if np.any(finite) else None,
            "gt_120ms": int(np.sum(finite & (age > 120.0))),
            "gt_120ms_tracked": int(np.sum(finite & (age > 120.0) & tracked)),
            "gt_120ms_position_tracked": int(np.sum(finite & (age > 120.0) & position_tracked)),
            "gt_320ms": int(np.sum(finite & (age > 320.0))),
            "gt_320ms_tracked": int(np.sum(finite & (age > 320.0) & tracked)),
            "gt_320ms_position_tracked": int(np.sum(finite & (age > 320.0) & position_tracked)),
        }
    return out


def jitter_diagnostics(grid: ReferenceGrid,
                       stream: CandidateStream | None,
                       match_ms: float,
                       max_pair_gap_ms: float = 80.0) -> dict[str, Any] | None:
    if stream is None:
        return None

    sample_valid = stream.position_valid if stream.position_valid is not None else stream.valid
    sample_valid = sample_valid & np.isfinite(stream.pos).all(axis=1)
    sample_idx = np.flatnonzero(sample_valid)
    frame_idx = _nearest_indices(grid.frames.t_ns, stream.t_ns[sample_idx], int(match_ms * 1e6))
    keep = frame_idx >= 0
    sample_idx = sample_idx[keep]
    frame_idx = frame_idx[keep]
    if sample_idx.size == 0:
        return {"matched_samples": 0}

    order = np.argsort(stream.t_ns[sample_idx], kind="stable")
    sample_idx = sample_idx[order]
    frame_idx = frame_idx[order]

    report_t = stream.t_ns[sample_idx].astype(np.int64)
    report_pos = stream.pos[sample_idx]
    report_tracked = stream.tracked[sample_idx] if stream.tracked is not None else stream.valid[sample_idx]
    scoreable = grid.scoreable[frame_idx]
    ref_pos = grid.ref_pos[frame_idx]
    residual = report_pos - ref_pos

    visible = grid.detect_visible[frame_idx] & scoreable
    valid_residual = scoreable & np.isfinite(residual).all(axis=1)
    tracked_scoreable = valid_residual & report_tracked
    tracked_visible = tracked_scoreable & visible

    dt_ms = np.diff(report_t) / 1e6
    consecutive = (dt_ms > 0.0) & (dt_ms <= max_pair_gap_ms)
    pair_scoreable = consecutive & tracked_scoreable[:-1] & tracked_scoreable[1:]
    pair_visible = consecutive & tracked_visible[:-1] & tracked_visible[1:]

    report_step_cm = np.linalg.norm(np.diff(report_pos, axis=0), axis=1) * 100.0
    residual_delta_cm = np.linalg.norm(np.diff(residual, axis=0), axis=1) * 100.0
    ref_step_cm = np.linalg.norm(np.diff(ref_pos, axis=0), axis=1) * 100.0

    second_delta_cm = []
    second_visible_cm = []
    for i in range(1, residual.shape[0] - 1):
        if (report_t[i] - report_t[i - 1]) / 1e6 > max_pair_gap_ms:
            continue
        if (report_t[i + 1] - report_t[i]) / 1e6 > max_pair_gap_ms:
            continue
        if not (tracked_scoreable[i - 1] and tracked_scoreable[i] and tracked_scoreable[i + 1]):
            continue
        d2 = residual[i + 1] - 2.0 * residual[i] + residual[i - 1]
        value = float(np.linalg.norm(d2) * 100.0)
        second_delta_cm.append(value)
        if tracked_visible[i - 1] and tracked_visible[i] and tracked_visible[i + 1]:
            second_visible_cm.append(value)

    stale_pair = np.zeros_like(pair_scoreable, dtype=bool)
    if stream.last_optical_age_ms is not None:
        age = stream.last_optical_age_ms[sample_idx]
        stale_pair = pair_scoreable & np.isfinite(age[:-1]) & np.isfinite(age[1:]) & ((age[:-1] > 120.0) | (age[1:] > 120.0))

    optical_reentry_steps = []
    if stream.last_optical_age_ms is not None:
        age = stream.last_optical_age_ms[sample_idx]
        for i in range(1, sample_idx.size):
            if not consecutive[i - 1] or not (valid_residual[i - 1] and valid_residual[i]):
                continue
            if not (np.isfinite(age[i - 1]) and np.isfinite(age[i])):
                continue
            if age[i - 1] > 120.0 and age[i] <= 40.0:
                optical_reentry_steps.append(float(report_step_cm[i - 1]))

    total_span_s = max((float(report_t[-1] - report_t[0]) / 1e9), 1e-9)
    return {
        "matched_samples": int(sample_idx.size),
        "sample_rate_hz": float(sample_idx.size / total_span_s),
        "tracked_scoreable_samples": int(tracked_scoreable.sum()),
        "tracked_visible_samples": int(tracked_visible.sum()),
        "report_interval_ms": _stat_block(dt_ms[dt_ms > 0.0]),
        "all_scoreable": {
            "pairs": int(pair_scoreable.sum()),
            "report_step_cm": _stat_block(report_step_cm[pair_scoreable]),
            "reference_step_cm": _stat_block(ref_step_cm[pair_scoreable]),
            "residual_delta_cm": _stat_block(residual_delta_cm[pair_scoreable]),
            "residual_second_delta_cm": _stat_block(np.array(second_delta_cm, dtype=float)),
        },
        "visible": {
            "pairs": int(pair_visible.sum()),
            "report_step_cm": _stat_block(report_step_cm[pair_visible]),
            "reference_step_cm": _stat_block(ref_step_cm[pair_visible]),
            "residual_delta_cm": _stat_block(residual_delta_cm[pair_visible]),
            "residual_second_delta_cm": _stat_block(np.array(second_visible_cm, dtype=float)),
        },
        "stale_optical": {
            "pairs": int(stale_pair.sum()),
            "report_step_cm": _stat_block(report_step_cm[stale_pair]),
            "residual_delta_cm": _stat_block(residual_delta_cm[stale_pair]),
        },
        "reentry": {
            "steps": int(len(optical_reentry_steps)),
            "report_step_cm": _stat_block(np.array(optical_reentry_steps, dtype=float)),
        },
    }


def oov_metrics(grid: ReferenceGrid,
                opt_score: dict[str, Any] | None,
                pred_stream: CandidateStream | None,
                match_ms: float,
                pos_cm_thresh: float,
                ori_deg_thresh: float) -> dict[str, Any] | None:
    if pred_stream is None or pred_stream.last_optical_age_ms is None:
        return None

    finite_samples = pred_stream.finite & np.isfinite(pred_stream.pos).all(axis=1) & np.isfinite(pred_stream.quat).all(axis=1)
    sample_idx = np.flatnonzero(finite_samples)
    frame_idx = _nearest_indices(grid.frames.t_ns, pred_stream.t_ns[sample_idx], int(match_ms * 1e6))
    keep = frame_idx >= 0
    sample_idx = sample_idx[keep]
    frame_idx = frame_idx[keep]

    n_frames = grid.frames.t_ns.shape[0]
    has_report = np.zeros(n_frames, dtype=bool)
    tracked = np.zeros(n_frames, dtype=bool)
    position_tracked = np.zeros(n_frames, dtype=bool)
    age_ms = np.full(n_frames, np.nan)
    fusion_state = np.full(n_frames, -1, dtype=np.int16)
    forced_drop = np.zeros(n_frames, dtype=bool)
    pos_err_cm = np.full(n_frames, np.nan)
    ori_err_deg = np.full(n_frames, np.nan)
    head_dist_m = np.full(n_frames, np.nan)
    for si, fi in zip(sample_idx, frame_idx):
        has_report[fi] = True
        tracked[fi] = bool(pred_stream.tracked[si]) if pred_stream.tracked is not None else bool(pred_stream.valid[si])
        position_tracked[fi] = bool(pred_stream.position_valid[si]) if pred_stream.position_valid is not None else tracked[fi]
        age_ms[fi] = float(pred_stream.last_optical_age_ms[si])
        if pred_stream.fusion_state is not None:
            fusion_state[fi] = int(pred_stream.fusion_state[si])
        if pred_stream.drop_optical is not None:
            forced_drop[fi] = bool(pred_stream.drop_optical[si])
        if grid.scoreable[fi]:
            pos_err_cm[fi] = float(np.linalg.norm(pred_stream.pos[si] - grid.ref_pos[fi]) * 100.0)
            ori_err_deg[fi] = float(quat_geodesic_deg(pred_stream.quat[si], grid.ref_quat[fi]))
            if grid.head_valid[fi]:
                head_dist_m[fi] = float(np.linalg.norm(pred_stream.pos[si] - grid.head_pos[fi]))

    freeze_err_cm = np.full(n_frames, np.nan)
    anchor_age_ms = np.full(n_frames, np.nan)
    if opt_score is not None:
        correct = opt_score.get("_frame_has_correct")
        last_correct_pos = None
        last_correct_t_ns = None
        for i in range(n_frames):
            if grid.scoreable[i] and correct is not None and bool(correct[i]):
                last_correct_pos = grid.ref_pos[i].copy()
                last_correct_t_ns = int(grid.frames.t_ns[i])
            if grid.scoreable[i] and last_correct_pos is not None:
                freeze_err_cm[i] = float(np.linalg.norm(last_correct_pos - grid.ref_pos[i]) * 100.0)
            if grid.scoreable[i] and last_correct_t_ns is not None:
                anchor_age_ms[i] = float(int(grid.frames.t_ns[i]) - last_correct_t_ns) / 1e6

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        mask = mask & grid.scoreable
        reported = mask & has_report
        stale120 = reported & np.isfinite(age_ms) & (age_ms > 120.0)
        stale320 = reported & np.isfinite(age_ms) & (age_ms > 320.0)
        tracked_reports = reported & tracked
        position_tracked_reports = reported & position_tracked
        pos_correct = reported & np.isfinite(pos_err_cm) & (pos_err_cm < pos_cm_thresh)
        pose_correct = pos_correct & np.isfinite(ori_err_deg) & (ori_err_deg < ori_deg_thresh)
        position_tracked_correct = position_tracked_reports & pos_correct
        tracked_pose_correct = tracked_reports & pose_correct
        state_counts: dict[str, int] = {}
        for value in sorted({int(x) for x in fusion_state[reported].tolist()}):
            state_counts[FUSION_STATE_NAMES.get(value, str(value))] = int(np.sum(reported & (fusion_state == value)))
        position_tracked_count = int(np.sum(reported & position_tracked))
        tracked_count = int(np.sum(reported & tracked))
        return {
            "frames": int(mask.sum()),
            "reported": int(reported.sum()),
            "tracked": tracked_count,
            "position_tracked": position_tracked_count,
            "reported_position_correct": int(pos_correct.sum()),
            "reported_pose_correct": int(pose_correct.sum()),
            "position_tracked_correct": int(position_tracked_correct.sum()),
            "tracked_pose_correct": int(tracked_pose_correct.sum()),
            "reported_pct": _safe_pct(int(reported.sum()), int(mask.sum())),
            "tracked_pct": _safe_pct(tracked_count, int(mask.sum())),
            "position_tracked_pct": _safe_pct(position_tracked_count, int(mask.sum())),
            "reported_position_correct_pct": _safe_pct(int(pos_correct.sum()), int(mask.sum())),
            "reported_pose_correct_pct": _safe_pct(int(pose_correct.sum()), int(mask.sum())),
            "position_tracked_correct_pct": _safe_pct(int(position_tracked_correct.sum()), int(mask.sum())),
            "tracked_pose_correct_pct": _safe_pct(int(tracked_pose_correct.sum()), int(mask.sum())),
            "position_tracked_accuracy_pct": _safe_pct(int(position_tracked_correct.sum()), position_tracked_count),
            "tracked_pose_accuracy_pct": _safe_pct(int(tracked_pose_correct.sum()), tracked_count),
            "age_gt_120ms": int(stale120.sum()),
            "age_gt_120ms_tracked": int(np.sum(stale120 & tracked)),
            "age_gt_120ms_position_tracked": int(np.sum(stale120 & position_tracked)),
            "age_gt_320ms": int(stale320.sum()),
            "age_gt_320ms_tracked": int(np.sum(stale320 & tracked)),
            "age_gt_320ms_position_tracked": int(np.sum(stale320 & position_tracked)),
            "pos_err_cm": _stat_block(pos_err_cm[reported]),
            "pos_err_tracked_cm": _stat_block(pos_err_cm[tracked_reports]),
            "pos_err_position_tracked_cm": _stat_block(pos_err_cm[position_tracked_reports]),
            "ori_err_deg": _stat_block(ori_err_deg[reported]),
            "ori_err_tracked_deg": _stat_block(ori_err_deg[tracked_reports]),
            "freeze_err_cm": _stat_block(freeze_err_cm[reported]),
            "head_dist_m": _stat_block(head_dist_m[reported]),
            "near_body_1p1m_pct": _safe_pct(int(np.sum(head_dist_m[reported] >= 1.09)), int(reported.sum())),
            "near_hard_1p5m_pct": _safe_pct(int(np.sum(head_dist_m[reported] >= 1.49)), int(reported.sum())),
            "states": state_counts,
        }

    def summarize_unscored(mask: np.ndarray) -> dict[str, Any]:
        mask = mask & ~grid.scoreable
        reported = mask & has_report
        stale120 = reported & np.isfinite(age_ms) & (age_ms > 120.0)
        stale320 = reported & np.isfinite(age_ms) & (age_ms > 320.0)
        state_counts: dict[str, int] = {}
        for value in sorted({int(x) for x in fusion_state[reported].tolist()}):
            state_counts[FUSION_STATE_NAMES.get(value, str(value))] = int(np.sum(reported & (fusion_state == value)))
        return {
            "frames": int(mask.sum()),
            "reported": int(reported.sum()),
            "tracked": int(np.sum(reported & tracked)),
            "position_tracked": int(np.sum(reported & position_tracked)),
            "reported_pct": _safe_pct(int(reported.sum()), int(mask.sum())),
            "tracked_pct": _safe_pct(int(np.sum(reported & tracked)), int(mask.sum())),
            "position_tracked_pct": _safe_pct(int(np.sum(reported & position_tracked)), int(mask.sum())),
            "age_gt_120ms": int(stale120.sum()),
            "age_gt_120ms_tracked": int(np.sum(stale120 & tracked)),
            "age_gt_120ms_position_tracked": int(np.sum(stale120 & position_tracked)),
            "age_gt_320ms": int(stale320.sum()),
            "age_gt_320ms_tracked": int(np.sum(stale320 & tracked)),
            "age_gt_320ms_position_tracked": int(np.sum(stale320 & position_tracked)),
            "states": state_counts,
        }

    scoreable = grid.scoreable
    unknown_blob_rich = (~scoreable) & (grid.frames.blob_total >= 4)
    actual_stale = has_report & np.isfinite(age_ms) & (age_ms > 80.0)
    return {
        "all_scoreable": summarize(scoreable),
        "not_detect_visible": summarize(~grid.detect_visible),
        "not_pose_visible": summarize(~grid.pose_visible),
        "detect_visible_but_stale": summarize(grid.detect_visible & actual_stale),
        "not_detect_visible_stale": summarize((~grid.detect_visible) & actual_stale),
        "forced_drop": summarize(forced_drop),
        "forced_drop_visible": summarize(forced_drop & grid.detect_visible),
        "forced_drop_visible_stale": summarize(forced_drop & grid.detect_visible & actual_stale),
        "forced_drop_visible_anchor_100ms": summarize(
            forced_drop & grid.detect_visible & np.isfinite(anchor_age_ms) & (anchor_age_ms <= 100.0)
        ),
        "forced_drop_visible_anchor_250ms": summarize(
            forced_drop & grid.detect_visible & np.isfinite(anchor_age_ms) & (anchor_age_ms <= 250.0)
        ),
        "forced_drop_visible_anchor_500ms": summarize(
            forced_drop & grid.detect_visible & np.isfinite(anchor_age_ms) & (anchor_age_ms <= 500.0)
        ),
        "unknown_blob_rich": summarize_unscored(unknown_blob_rich),
        "forced_drop_unknown_blob_rich": summarize_unscored(forced_drop & unknown_blob_rich),
    }


def _stream_position_by_frame(grid: ReferenceGrid, stream: CandidateStream | None, match_ms: float,
                              tracked_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    has_pos = np.zeros(grid.frames.t_ns.shape[0], dtype=bool)
    pos = np.full((grid.frames.t_ns.shape[0], 3), np.nan)
    if stream is None:
        return has_pos, pos
    valid = stream.position_valid if stream.position_valid is not None else stream.valid
    if tracked_only and stream.tracked is not None:
        valid = valid & stream.tracked.astype(bool)
    valid = valid & np.isfinite(stream.pos).all(axis=1)
    sample_idx = np.flatnonzero(valid)
    frame_idx = _nearest_indices(grid.frames.t_ns, stream.t_ns[sample_idx], int(match_ms * 1e6))
    matched = frame_idx >= 0
    for si, fi in zip(sample_idx[matched], frame_idx[matched]):
        has_pos[fi] = True
        pos[fi] = stream.pos[si]
    return has_pos, pos


def cross_device_identity_diagnostics(grids: dict[int, ReferenceGrid],
                                      pred_streams: dict[int, CandidateStream | None],
                                      match_ms: float,
                                      pos_cm_thresh: float) -> dict[int, dict[str, Any]]:
    if 1 not in grids or 2 not in grids:
        return {}
    if grids[1].frames.t_ns.shape != grids[2].frames.t_ns.shape or not np.array_equal(grids[1].frames.t_ns, grids[2].frames.t_ns):
        return {}

    has: dict[int, np.ndarray] = {}
    pos: dict[int, np.ndarray] = {}
    for dev in (1, 2):
        # Identity events are claims about TRACKED reports: a matcher locking onto the partner's
        # LEDs. Untracked coast reports may legitimately drift near the partner (hands-close
        # out-of-view); that drift is measured by the stale/drop error terms, and counting it here
        # would flag every honest coast policy as a swap (the legacy freeze policy never exposed
        # moving untracked reports, so this gate was previously vacuous).
        has[dev], pos[dev] = _stream_position_by_frame(grids[dev], pred_streams.get(dev), match_ms,
                                                       tracked_only=True)

    out: dict[int, dict[str, Any]] = {}
    for dev, other in ((1, 2), (2, 1)):
        grid = grids[dev]
        scoreable_visible = grid.scoreable & grid.detect_visible
        own_err_cm = np.full(grid.frames.t_ns.shape[0], np.nan)
        other_err_cm = np.full(grid.frames.t_ns.shape[0], np.nan)
        own = scoreable_visible & has[dev]
        other_has = scoreable_visible & has[other]
        own_err_cm[own] = np.linalg.norm(pos[dev][own] - grid.ref_pos[own], axis=1) * 100.0
        other_err_cm[other_has] = np.linalg.norm(pos[other][other_has] - grid.ref_pos[other_has], axis=1) * 100.0
        other_explains = scoreable_visible & has[dev] & has[other] & (own_err_cm >= pos_cm_thresh) & (other_err_cm < pos_cm_thresh)
        out[dev] = {
            "visible_scoreable_frames": int(scoreable_visible.sum()),
            "own_position_reports": int(np.sum(scoreable_visible & has[dev])),
            "other_position_reports": int(np.sum(scoreable_visible & has[other])),
            "other_report_explains_ref": int(np.sum(other_explains)),
            "other_report_explains_ref_pct_visible": _safe_pct(int(np.sum(other_explains)), int(scoreable_visible.sum())),
            "other_report_explains_ref_pct_own_reports": _safe_pct(int(np.sum(other_explains)),
                                                                    int(np.sum(scoreable_visible & has[dev]))),
            "own_bad_other_good_median_own_err_cm": float(np.median(own_err_cm[other_explains])) if np.any(other_explains) else None,
            "own_bad_other_good_median_other_err_cm": float(np.median(other_err_cm[other_explains])) if np.any(other_explains) else None,
        }

    both_scoreable = grids[1].scoreable & grids[2].scoreable & grids[1].detect_visible & grids[2].detect_visible
    both_reports = both_scoreable & has[1] & has[2]
    strict_swap = np.zeros(grids[1].frames.t_ns.shape[0], dtype=bool)
    if np.any(both_reports):
        e11 = np.linalg.norm(pos[1][both_reports] - grids[1].ref_pos[both_reports], axis=1) * 100.0
        e22 = np.linalg.norm(pos[2][both_reports] - grids[2].ref_pos[both_reports], axis=1) * 100.0
        e12 = np.linalg.norm(pos[1][both_reports] - grids[2].ref_pos[both_reports], axis=1) * 100.0
        e21 = np.linalg.norm(pos[2][both_reports] - grids[1].ref_pos[both_reports], axis=1) * 100.0
        indices = np.flatnonzero(both_reports)
        strict_swap[indices] = (e11 >= pos_cm_thresh) & (e22 >= pos_cm_thresh) & (e12 < pos_cm_thresh) & (e21 < pos_cm_thresh)
    paired = {
        "both_visible_scoreable_frames": int(both_scoreable.sum()),
        "both_position_reports": int(both_reports.sum()),
        "strict_two_way_swap_frames": int(strict_swap.sum()),
        "strict_two_way_swap_pct_both_visible": _safe_pct(int(strict_swap.sum()), int(both_scoreable.sum())),
        "strict_two_way_swap_pct_both_reports": _safe_pct(int(strict_swap.sum()), int(both_reports.sum())),
    }
    out[1]["paired"] = paired
    out[2]["paired"] = paired
    return out


def coast_metrics(grid: ReferenceGrid,
                  opt_score: dict[str, Any] | None,
                  pred_stream: CandidateStream | None,
                  match_ms: float,
                  pos_cm_thresh: float) -> dict[str, Any] | None:
    if opt_score is None or pred_stream is None:
        return None
    pred_finite = pred_stream.finite & np.isfinite(pred_stream.pos).all(axis=1)
    pred_idx = np.flatnonzero(pred_finite)
    pred_frame = _nearest_indices(grid.frames.t_ns, pred_stream.t_ns[pred_idx], int(match_ms * 1e6))
    keep = pred_frame >= 0
    pred_idx = pred_idx[keep]
    pred_frame = pred_frame[keep]
    report_pos_by_frame = np.full((grid.frames.t_ns.shape[0], 3), np.nan)
    report_tracked_by_frame = np.zeros(grid.frames.t_ns.shape[0], dtype=bool)
    has_report_by_frame = np.zeros(grid.frames.t_ns.shape[0], dtype=bool)
    for si, fi in zip(pred_idx, pred_frame):
        report_pos_by_frame[fi] = pred_stream.pos[si]
        has_report_by_frame[fi] = True
        if pred_stream.tracked is not None:
            report_tracked_by_frame[fi] = bool(pred_stream.tracked[si])
        else:
            report_tracked_by_frame[fi] = bool(pred_stream.valid[si])

    last_correct_t = None
    last_correct_pos = None
    bins = {
        "0_80ms": [],
        "80_160ms": [],
        "160_320ms": [],
        "gt_320ms": [],
    }
    freeze_bins = {k: [] for k in bins}
    reported_bins = {k: [] for k in bins}
    head_dist_bins = {k: [] for k in bins}
    tracked_bins = {k: [] for k in bins}
    reported_correct_bins = {k: 0 for k in bins}
    tracked_correct_bins = {k: 0 for k in bins}
    correct = opt_score["_frame_has_correct"]
    for i, t in enumerate(grid.frames.t_ns):
        if not grid.scoreable[i] or not has_report_by_frame[i]:
            continue
        if correct[i]:
            last_correct_t = int(t)
            last_correct_pos = grid.ref_pos[i].copy()
        if last_correct_t is None:
            continue
        age_ms = (int(t) - last_correct_t) / 1e6
        reported_err = float(np.linalg.norm(report_pos_by_frame[i] - grid.ref_pos[i]) * 100.0)
        freeze_err = float(np.linalg.norm(last_correct_pos - grid.ref_pos[i]) * 100.0)
        if age_ms <= 80:
            key = "0_80ms"
        elif age_ms <= 160:
            key = "80_160ms"
        elif age_ms <= 320:
            key = "160_320ms"
        else:
            key = "gt_320ms"
        if report_tracked_by_frame[i]:
            bins[key].append(reported_err)
        if reported_err < pos_cm_thresh:
            reported_correct_bins[key] += 1
            if report_tracked_by_frame[i]:
                tracked_correct_bins[key] += 1
        reported_bins[key].append(reported_err)
        freeze_bins[key].append(freeze_err)
        if grid.head_valid[i]:
            head_dist_bins[key].append(float(np.linalg.norm(report_pos_by_frame[i] - grid.head_pos[i])))
        tracked_bins[key].append(bool(report_tracked_by_frame[i]))
    out: dict[str, Any] = {}
    for key, values in bins.items():
        arr = np.array(values, dtype=float)
        fz = np.array(freeze_bins[key], dtype=float)
        reported = np.array(reported_bins[key], dtype=float)
        head_dist = np.array(head_dist_bins[key], dtype=float)
        tracked = np.array(tracked_bins[key], dtype=bool)
        out[key] = {
            "n": int(reported.size),
            "tracked_n": int(arr.size),
            "reported_position_correct_n": int(reported_correct_bins[key]),
            "tracked_position_correct_n": int(tracked_correct_bins[key]),
            "tracked_pct": _safe_pct(int(tracked.sum()), int(tracked.size)) if tracked.size else None,
            "reported_position_correct_pct": (
                _safe_pct(int(reported_correct_bins[key]), int(reported.size)) if reported.size else None
            ),
            "tracked_position_correct_pct": (
                _safe_pct(int(tracked_correct_bins[key]), int(reported.size)) if reported.size else None
            ),
            "tracked_position_accuracy_pct": (
                _safe_pct(int(tracked_correct_bins[key]), int(arr.size)) if arr.size else None
            ),
            "pred_median_cm": float(np.median(arr)) if arr.size else None,
            "pred_p75_cm": float(np.percentile(arr, 75)) if arr.size else None,
            "pred_p95_cm": float(np.percentile(arr, 95)) if arr.size else None,
            "reported_median_cm": float(np.median(reported)) if reported.size else None,
            "reported_p95_cm": float(np.percentile(reported, 95)) if reported.size else None,
            "freeze_median_cm": float(np.median(fz)) if fz.size else None,
            "freeze_p95_cm": float(np.percentile(fz, 95)) if fz.size else None,
            "head_dist_median_m": float(np.median(head_dist)) if head_dist.size else None,
            "head_dist_p95_m": float(np.percentile(head_dist, 95)) if head_dist.size else None,
            "head_dist_max_m": float(np.max(head_dist)) if head_dist.size else None,
            "near_body_1p1m_pct": _safe_pct(int(np.sum(head_dist >= 1.09)), int(head_dist.size)) if head_dist.size else None,
            "near_body_1p35m_pct": _safe_pct(int(np.sum(head_dist >= 1.34)), int(head_dist.size)) if head_dist.size else None,
            "near_hard_1p5m_pct": _safe_pct(int(np.sum(head_dist >= 1.49)), int(head_dist.size)) if head_dist.size else None,
        }
    return out


def strip_internal(score: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in score.items() if not k.startswith("_")}


def print_summary(results: list[dict[str, Any]], visibility_name: str) -> None:
    for result in results:
        print(f"\n=== device {result['device']} ({DEVICE_NAMES.get(result['device'], result['device'])}) ===")
        ref = result["reference"]
        print(
            f"reference: frames={ref['total_frame_groups']} scoreable={ref['scoreable_frames']} "
            f"unknown={ref['unknown_frames']} full4cam={ref['full_4cam_groups']}"
        )
        print(
            f"unknown: reasons={ref['unknown_reasons']} blob>=4={ref['unknown_blob_total_ge4']} "
            f"blob>=8={ref['unknown_blob_total_ge8']} zero_blob={ref['unknown_blob_total_zero']}"
        )
        print(
            f"visibility: detect={ref['detect_visible_frames']} pose={ref['pose_visible_frames']} "
            f"high={ref['high_visible_frames']} scoring={visibility_name}"
        )
        print("stream          TP    FP    FN  precision  recall     F1  posF1 posRec posRMSEcm  oriRMSdeg  wrongBr%  yield%")
        print("-" * 117)
        for name, score in result["scores"].items():
            print(
                f"{name:12s} {score['TP']:5d} {score['FP']:5d} {score['FN']:5d} "
                f"{score['precision']:9.3f} {score['recall']:7.3f} {score['f1']:7.3f} "
                f"{score.get('position_f1', float('nan')):6.3f} {score.get('position_recall', float('nan')):6.3f} "
                f"{score['pos_rmse_cm'] if score['pos_rmse_cm'] is not None else float('nan'):10.2f} "
                f"{score['ori_rms_deg'] if score['ori_rms_deg'] is not None else float('nan'):10.2f} "
                f"{score['wrong_branch_pct_scored']:8.2f} {score['yield_pct_frame_rows']:7.1f}"
            )
        if result.get("search"):
            print(f"search diagnostics: {result['search']}")
        if result.get("candidate_rows"):
            cold_unselected = result["candidate_rows"].get("cold_search_unselected", {})
            print(f"candidate rows: cold_search_unselected={cold_unselected}")
        if result.get("candidate_frame_gaps"):
            gaps = result["candidate_frame_gaps"].get("failure_frames", {})
            print(
                "candidate frame gaps: "
                f"failures={gaps.get('frames')} correct_alt={gaps.get('correct_alt_exists')} "
                f"position_alt={gaps.get('position_correct_alt_exists')} "
                f"no_position_alt={gaps.get('no_position_correct_alt')} "
                f"selected_wrong_branch={gaps.get('selected_wrong_branch')}"
            )
        if result.get("unknown_adjudication"):
            ua = result["unknown_adjudication"]
            print(
                "unknown adjudication: "
                f"blob_rich={ua.get('blob_rich_unknown_frames')} "
                f"reference/evidence={ua.get('explained_or_reference_gap')} "
                f"tracker_unresolved={ua.get('tracker_unresolved_blob_rich')}"
            )
        if result.get("cross_device_identity"):
            identity = result["cross_device_identity"]
            paired = identity.get("paired", {})
            print(
                "cross-device identity: "
                f"other_explains_ref={identity.get('other_report_explains_ref')} "
                f"({identity.get('other_report_explains_ref_pct_visible'):.2f}% visible) "
                f"strict_two_way_swaps={paired.get('strict_two_way_swap_frames')}"
            )
        if result.get("fusion"):
            fusion = result["fusion"]
            age = fusion.get("last_optical_age_ms", {})
            print(
                "fusion: "
                f"tracked={fusion.get('tracked')}/{fusion.get('rows')} "
                f"age>120ms={age.get('gt_120ms')} tracked={age.get('gt_120ms_tracked')} "
                f"age>320ms={age.get('gt_320ms')} tracked={age.get('gt_320ms_tracked')} "
                f"states={fusion.get('by_state')}"
            )
        if result.get("jitter"):
            pred_jitter = result["jitter"].get("pred") or {}
            visible = pred_jitter.get("visible", {})
            residual_delta = visible.get("residual_delta_cm", {})
            residual_second = visible.get("residual_second_delta_cm", {})
            interval = pred_jitter.get("report_interval_ms", {})
            reentry = pred_jitter.get("reentry", {}).get("report_step_cm", {})
            print(
                "jitter: "
                f"pred_rate_hz={pred_jitter.get('sample_rate_hz')} "
                f"interval_p95_ms={interval.get('p95')} "
                f"visible_residual_delta_p95_cm={residual_delta.get('p95')} "
                f"visible_second_delta_p95_cm={residual_second.get('p95')} "
                f"reentry_step_p95_cm={reentry.get('p95')}"
            )
        if result.get("oov"):
            not_visible = result["oov"].get("not_detect_visible", {})
            stale_visible = result["oov"].get("detect_visible_but_stale", {})
            forced_visible = result["oov"].get("forced_drop_visible", {})
            nv_pos = not_visible.get("pos_err_cm", {})
            sv_pos = stale_visible.get("pos_err_cm", {})
            fv_pos = forced_visible.get("pos_err_cm", {})
            print(
                "oov: "
                f"not_detect_visible frames={not_visible.get('frames')} reported={not_visible.get('reported')} "
                f"tracked={not_visible.get('tracked')} pos_p95_cm={nv_pos.get('p95')} "
                f"detect_visible_but_stale frames={stale_visible.get('frames')} "
                f"reported={stale_visible.get('reported')} pos_p95_cm={sv_pos.get('p95')} "
                f"forced_drop_visible frames={forced_visible.get('frames')} "
                f"reported={forced_visible.get('reported')} pos_p95_cm={fv_pos.get('p95')}"
            )
        if result.get("coast"):
            print(f"coast metrics: {result['coast']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_capture", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--cams", default=None,
                        help="override the camera config (default: the reference capture's own "
                             "provenance snapshot, else the pinned pre-provenance config)")
    parser.add_argument("--ctrl-left", default=DEFAULT_CTRL_LEFT)
    parser.add_argument("--ctrl-right", default=DEFAULT_CTRL_RIGHT)
    parser.add_argument("--match-ms", type=float, default=25.0)
    parser.add_argument("--head-match-ms", type=float, default=50.0)
    parser.add_argument("--max-ref-gap-ms", type=float, default=150.0)
    parser.add_argument("--pos-cm", type=float, default=5.0)
    parser.add_argument("--ori-deg", type=float, default=15.0)
    parser.add_argument("--detect-leds", type=int, default=3)
    parser.add_argument("--pose-leds", type=int, default=4)
    parser.add_argument("--high-leds", type=int, default=7)
    parser.add_argument("--reference-qc", choices=("none", "corrupt", "confirmed"), default="confirmed")
    parser.add_argument("--visibility", choices=("detect", "pose", "high"), default="detect")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cams = load_cameras(args.cams or cams_for_capture(args.reference_capture))
    results = []
    grids: dict[int, ReferenceGrid] = {}
    pred_streams: dict[int, CandidateStream | None] = {}
    for dev, ctrl in ((1, args.ctrl_left), (2, args.ctrl_right)):
        grid = build_reference_grid(
            args.reference_capture,
            dev,
            cams,
            ctrl,
            args.max_ref_gap_ms,
            args.head_match_ms,
            args.detect_leds,
            args.pose_leds,
            args.high_leds,
            args.reference_qc,
        )
        grids[dev] = grid
        visible_mask = {
            "detect": grid.detect_visible,
            "pose": grid.pose_visible,
            "high": grid.high_visible,
        }[args.visibility]
        scores: dict[str, dict[str, Any]] = {}
        opt_score_internal = None
        pred_stream = None
        if args.candidate.is_dir() and (args.candidate / f"dev{dev}.csv").is_file():
            for col in ("opt", "pred"):
                stream = load_csv_stream(args.candidate / f"dev{dev}.csv", col)
                score = score_stream(grid, stream, visible_mask, args.match_ms, args.pos_cm, args.ori_deg)
                scores[col] = strip_internal(score)
                if col == "opt":
                    opt_score_internal = score
                else:
                    pred_stream = stream
            pred_streams[dev] = pred_stream
        else:
            stream = load_candidate_for_path(args.candidate, dev, "opt", cams, grid)
            score = score_stream(grid, stream, visible_mask, args.match_ms, args.pos_cm, args.ori_deg)
            scores[stream.col] = strip_internal(score)
            opt_score_internal = score
            pred_streams[dev] = None

        search_summary = summarize_search(args.candidate, grid, opt_score_internal, args.match_ms) if opt_score_internal else None
        candidate_summary = summarize_candidate_rows(args.candidate, grid, cams, args.match_ms, args.pos_cm, args.ori_deg)
        frame_gap_summary = summarize_candidate_frame_gaps(
            args.candidate,
            grid,
            cams,
            opt_score_internal,
            visible_mask,
            args.match_ms,
            args.pos_cm,
            args.ori_deg,
        ) if opt_score_internal else None
        unknown_adjudication = summarize_unknown_adjudication(
            args.candidate,
            grid,
            opt_score_internal,
            pred_stream,
            args.match_ms,
        ) if opt_score_internal else None
        coast = (
            coast_metrics(grid, opt_score_internal, pred_stream, args.match_ms, args.pos_cm)
            if pred_stream is not None
            else None
        )
        fusion = fusion_diagnostics(pred_stream)
        jitter = {
            name: jitter_diagnostics(grid, load_csv_stream(args.candidate / f"dev{dev}.csv", name), args.match_ms)
            for name in ("opt", "pred")
        } if args.candidate.is_dir() and (args.candidate / f"dev{dev}.csv").is_file() else None
        oov = oov_metrics(
            grid,
            opt_score_internal,
            pred_stream,
            args.match_ms,
            args.pos_cm,
            args.ori_deg,
        ) if pred_stream is not None else None
        results.append({
            "device": dev,
            "candidate_name": args.candidate_name,
            "reference_qc": args.reference_qc,
            "reference": summarize_reference(grid),
            "scores": scores,
            "search": search_summary,
            "candidate_rows": candidate_summary,
            "candidate_frame_gaps": frame_gap_summary,
            "unknown_adjudication": unknown_adjudication,
            "coast": coast,
            "fusion": fusion,
            "jitter": jitter,
            "oov": oov,
        })

    identity = cross_device_identity_diagnostics(grids, pred_streams, args.match_ms, args.pos_cm)
    for result in results:
        result["cross_device_identity"] = identity.get(int(result["device"]))

    print_summary(results, args.visibility)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
