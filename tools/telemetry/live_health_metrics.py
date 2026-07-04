#!/usr/bin/env python3
"""Live-session health metrics for G2 controller tracking captures.

This intentionally does not compute precision/recall: a live capture has no
independent controller ground truth. It reports metrics that map to headset
feel: frame cadence, optical/fusion cadence, optical gaps, ESKF fold health,
search cost/clutter, and optional replay/report jumps.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import DEVICE_NAMES, EVENT_TYPES, FUSION_OUTCOME, Manifest, POSE_OUTCOME  # noqa: E402


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


def _telemetry_dir(path: Path) -> Path:
    if (path / "manifest.json").is_file():
        return path
    if (path / "telemetry" / "manifest.json").is_file():
        return path / "telemetry"
    raise FileNotFoundError(f"no manifest.json under {path}")


def _read_stream(d: Path, m: Manifest, name: str) -> np.ndarray:
    s = m.streams[name]
    p = d / s.file
    if not p.is_file():
        return np.zeros(0, dtype=s.structured_dtype())
    rows = p.stat().st_size // s.row_size
    return np.fromfile(p, dtype=s.structured_dtype(), count=rows)


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def _stat(values: np.ndarray) -> dict[str, float | int | None]:
    v = _finite(np.asarray(values, dtype=float))
    if v.size == 0:
        return {"n": 0, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
    }


def _rate_stats(t_ns: np.ndarray) -> dict[str, Any]:
    # keep the caller's dtype: device-clock u64 timestamps can exceed int64
    t = np.unique(np.asarray(t_ns))
    t.sort()
    if t.size < 2:
        return {
            "n": int(t.size),
            "span_s": 0.0,
            "hz": 0.0,
            "dt_ms": _stat(np.array([], dtype=float)),
            "gaps_gt_100ms": 0,
            "gaps_gt_200ms": 0,
            "gaps_gt_500ms": 0,
        }
    span_s = float((t[-1] - t[0]) * 1e-9)
    dt_ms = np.diff(t).astype(float) * 1e-6
    return {
        "n": int(t.size),
        "span_s": span_s,
        "hz": float(t.size / span_s) if span_s > 0 else 0.0,
        "dt_ms": _stat(dt_ms),
        "gaps_gt_100ms": int(np.sum(dt_ms > 100.0)),
        "gaps_gt_200ms": int(np.sum(dt_ms > 200.0)),
        "gaps_gt_500ms": int(np.sum(dt_ms > 500.0)),
    }


def _top_counts(values: np.ndarray, names: dict[int, str] | None = None, limit: int = 8) -> dict[str, int]:
    c = Counter(int(x) for x in values.tolist())
    out: dict[str, int] = {}
    for key, n in c.most_common(limit):
        label = names.get(key, str(key)) if names else str(key)
        out[label] = int(n)
    return out


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(a * b, axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _step_stats(pos: np.ndarray, quat: np.ndarray, t_ns: np.ndarray) -> dict[str, Any]:
    if len(t_ns) < 2:
        return {"pos_step_cm": _stat(np.array([])), "rot_step_deg": _stat(np.array([]))}
    order = np.argsort(t_ns)
    p = pos[order]
    q = quat[order]
    fin = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    p = p[fin]
    q = q[fin]
    if p.shape[0] < 2:
        return {"pos_step_cm": _stat(np.array([])), "rot_step_deg": _stat(np.array([]))}
    return {
        "pos_step_cm": _stat(np.linalg.norm(np.diff(p, axis=0), axis=1) * 100.0),
        "rot_step_deg": _stat(_quat_angle_deg(q[1:], q[:-1])),
    }


def _run_lengths_ms(t_ns: np.ndarray, bad: np.ndarray) -> list[float]:
    runs = []
    start = None
    for i, is_bad in enumerate(bad):
        if is_bad and start is None:
            start = i
        if not is_bad and start is not None:
            runs.append(float((t_ns[i - 1] - t_ns[start]) * 1e-6))
            start = None
    if start is not None and t_ns.size:
        runs.append(float((t_ns[-1] - t_ns[start]) * 1e-6))
    return runs


def _jitter_metric(t_ns: np.ndarray, pos: np.ndarray, win: int = 5, gap_ms: float = 120.0) -> dict[str, Any]:
    n = int(t_ns.shape[0])
    if n == 0:
        return {"rms_cm": 0.0, "p95_cm": 0.0, "n": 0}
    t_s = (t_ns - t_ns[0]).astype(float) * 1e-9
    resid = []
    for i in range(n):
        a = i
        while a > max(0, i - win) and (t_ns[a] - t_ns[a - 1]) * 1e-6 <= gap_ms:
            a -= 1
        b = i
        while b + 1 < min(n, i + win + 1) and (t_ns[b + 1] - t_ns[b]) * 1e-6 <= gap_ms:
            b += 1
        if b - a < 2:
            continue
        tt = t_s[a : b + 1]
        A = np.vstack([tt, np.ones(tt.shape[0])]).T
        d2 = 0.0
        for axis in range(3):
            coef, *_ = np.linalg.lstsq(A, pos[a : b + 1, axis], rcond=None)
            fit_i = coef[0] * t_s[i] + coef[1]
            d2 += float((pos[i, axis] - fit_i) ** 2)
        resid.append(math.sqrt(d2))
    if not resid:
        return {"rms_cm": 0.0, "p95_cm": 0.0, "n": 0}
    r = np.asarray(resid, dtype=float)
    return {
        "rms_cm": float(np.sqrt(np.mean(r * r)) * 100.0),
        "p95_cm": float(np.percentile(r, 95) * 100.0),
        "n": int(r.size),
    }


def _frame_metrics(frame: np.ndarray) -> dict[str, Any]:
    if frame.size == 0:
        return {}
    main_exposure = None
    if "exposure" in frame.dtype.names and frame.size:
        vals, counts = np.unique(frame["exposure"], return_counts=True)
        main_exposure = int(vals[int(np.argmax(counts))])
    groups: dict[int, list[Any]] = {}
    for r in frame:
        groups.setdefault(int(r["hw_ts_ns"]), []).append(r)
    complete_times = []
    complete_source_seq = []
    blob_totals = []
    cams_per_group = []
    mixed_source_sequence_groups = 0
    for key, rows in groups.items():
        cams = {int(r["cam_id"]) for r in rows}
        cams_per_group.append(len(cams))
        blob_totals.append(sum(int(r["n_blobs"]) for r in rows))
        if len(cams) == 4:
            complete_times.append(key)
            seqs = {int(r["frame_seq"]) for r in rows}
            if len(seqs) != 1:
                mixed_source_sequence_groups += 1
            complete_source_seq.append(min(seqs))
    # hw_ts_ns is a device-clock u64 and can exceed int64 on live captures
    order = np.argsort(np.asarray(complete_times, dtype=np.uint64)) if complete_times else np.asarray([], dtype=int)
    source_delta = np.array([], dtype=float)
    source_delta_forward = np.array([], dtype=float)
    source_backward_jumps = 0
    source_gaps_gt_2 = 0
    source_gaps_gt_5 = 0
    source_gaps_gt_10 = 0
    source_hard_jumps_gt_100 = 0
    if complete_source_seq and order.size >= 2:
        seq = np.asarray(complete_source_seq, dtype=np.int64)[order]
        delta = np.diff(seq)
        source_delta = delta.astype(float)
        forward = delta[delta > 0]
        source_delta_forward = forward.astype(float)
        source_backward_jumps = int(np.sum(delta < 0))
        source_gaps_gt_2 = int(np.sum(forward > 2))
        source_gaps_gt_5 = int(np.sum(forward > 5))
        source_gaps_gt_10 = int(np.sum(forward > 10))
        source_hard_jumps_gt_100 = int(np.sum(forward > 100))
    return {
        "rows": int(frame.size),
        "groups": int(len(groups)),
        "complete_4cam_groups": int(len(complete_times)),
        "main_exposure": main_exposure,
        "cadence": _rate_stats(np.asarray(complete_times, dtype=np.uint64)),
        "source_sequence": {
            "delta_signed": _stat(source_delta),
            "delta_forward": _stat(source_delta_forward),
            "gaps_gt_2": source_gaps_gt_2,
            "gaps_gt_5": source_gaps_gt_5,
            "gaps_gt_10": source_gaps_gt_10,
            "hard_jumps_gt_100": source_hard_jumps_gt_100,
            "backward_jumps": source_backward_jumps,
            "mixed_4cam_groups": mixed_source_sequence_groups,
        },
        "blob_total_per_group": _stat(np.asarray(blob_totals, dtype=float)),
        "cams_per_group": _stat(np.asarray(cams_per_group, dtype=float)),
    }


def _imu_metrics(imu: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if imu.size == 0:
        return out
    for dev in sorted({int(x) for x in imu["device_id"].tolist()}):
        rows = imu[imu["device_id"] == dev]
        gyro = np.stack([rows["gx"], rows["gy"], rows["gz"]], axis=1).astype(float)
        accel = np.stack([rows["ax"], rows["ay"], rows["az"]], axis=1).astype(float)
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "cadence": _rate_stats(rows["t_mono_ns"]),
            "gyro_norm_rad_s": _stat(np.linalg.norm(gyro, axis=1)),
            "accel_norm_m_s2": _stat(np.linalg.norm(accel, axis=1)),
        }
    return out


def _pose_metrics(pose: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if pose.size == 0:
        return out
    for dev in (1, 2):
        rows = pose[pose["device_id"] == dev]
        if rows.size == 0:
            continue
        accepted = rows[rows["outcome"] == 1]
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "rows": int(rows.size),
            "outcomes": _top_counts(rows["outcome"], POSE_OUTCOME),
            "accepted_cadence": _rate_stats(accepted["t_mono_ns"]),
            "reproj_err_px": _stat(accepted["reproj_err_px"].astype(float)) if accepted.size else _stat(np.array([])),
        }
    return out


def _fusion_metrics(fusion: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if fusion.size == 0:
        return out
    for dev in (1, 2):
        rows = fusion[fusion["device_id"] == dev]
        if rows.size == 0:
            continue
        rows = rows[np.argsort(rows["t_mono_ns"])]
        pos = np.stack([rows["pred_px"], rows["pred_py"], rows["pred_pz"]], axis=1).astype(float)
        quat = np.stack([rows["pred_qx"], rows["pred_qy"], rows["pred_qz"], rows["pred_qw"]], axis=1).astype(float)
        gaps_ms = np.diff(rows["t_mono_ns"].astype(np.int64)) * 1e-6 if rows.size > 1 else np.array([])
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "rows": int(rows.size),
            "outcomes": _top_counts(rows["outcome"], FUSION_OUTCOME),
            "cadence": _rate_stats(rows["t_mono_ns"]),
            "gap_ms": _stat(gaps_ms),
            "gaps_gt_500ms": int(np.sum(gaps_ms > 500.0)),
            "gaps_gt_1000ms": int(np.sum(gaps_ms > 1000.0)),
            "pos_residual_cm": _stat(rows["pos_residual_m"].astype(float) * 100.0),
            "rot_residual_deg": _stat(rows["rot_residual_deg"].astype(float)),
            **_step_stats(pos, quat, rows["t_mono_ns"].astype(np.int64)),
        }
    return out


#: Re-entry gap bins (C1 session-analysis 2026-07-03 definition: a coast episode is
#: >60 ms between ACCEPTED optical folds; rejected attempts do not end a coast).
REENTRY_GAP_BINS = (("60-100ms", 0.06, 0.1), ("100-300ms", 0.1, 0.3), ("300ms-1s", 0.3, 1.0),
                    ("1-3s", 1.0, 3.0), (">3s", 3.0, math.inf))
REENTRY_ROT_FLAG_DEG = 30.0


def _reentry_metrics(fusion: np.ndarray, t0_ns: int | None = None) -> dict[str, Any]:
    """GT-free re-entry scoring at the FIRST ACCEPTED fold after each optical gap:
    pos_residual is the felt snap distance and rot_residual the ATTITUDE error the coast
    carried (position-only snap scoring missed the 41-107 deg re-entry attitude errors
    that ARE the felt left-controller fly-off -- c1-left-oov/findings.md, 2026-07-03)."""
    out: dict[str, Any] = {}
    for dev in (1, 2):
        rows = fusion[fusion["device_id"] == dev]
        rows = rows[rows["outcome"] == 1]
        if rows.size < 3:
            continue
        rows = rows[np.argsort(rows["t_mono_ns"])]
        t_ns = rows["t_mono_ns"].astype(np.int64)
        gaps_s = np.diff(t_ns) * 1e-9
        snap_cm = rows["pos_residual_m"].astype(float) * 100.0
        rot_deg = rows["rot_residual_deg"].astype(float)
        idx = np.flatnonzero(gaps_s > REENTRY_GAP_BINS[0][1])
        re_i = idx + 1
        fin = np.isfinite(snap_cm[re_i]) & np.isfinite(rot_deg[re_i])
        idx, re_i = idx[fin], re_i[fin]
        bins: dict[str, dict[str, Any]] = {}
        for name, lo, hi in REENTRY_GAP_BINS:
            sel = (gaps_s[idx] >= lo) & (gaps_s[idx] < hi)
            if not sel.any():
                continue
            bins[name] = {"snap_cm": _stat(snap_cm[re_i[sel]]), "rot_deg": _stat(rot_deg[re_i[sel]])}
        t_base = int(t0_ns) if t0_ns is not None else (int(t_ns[0]) if t_ns.size else 0)
        worst_order = np.argsort(-rot_deg[re_i])[:5]
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "reentries": int(idx.size),
            "rot_gt30deg": int(np.sum(rot_deg[re_i] > REENTRY_ROT_FLAG_DEG)),
            "reentry_rot_deg": _stat(rot_deg[re_i]),
            "reentry_by_gap": bins,
            "worst_rot": [
                {
                    "t_re_rel_s": round(float((t_ns[re_i[k]] - t_base) * 1e-9), 2),
                    "gap_s": round(float(gaps_s[idx[k]]), 2),
                    "snap_cm": round(float(snap_cm[re_i[k]]), 1),
                    "rot_deg": round(float(rot_deg[re_i[k]]), 1),
                }
                for k in worst_order
            ],
        }
    return out


#: Live blob detection threshold: newer capture provenance snapshots carry
#: blob_detect_threshold in hmd-cameras.json (t_constellation_tracking.c:325); older ones
#: predate the field, so fall back to the deployed constant BLOB_THRESHOLD_MIN_WMR = 0x18
#: (wmr_hmd.c) that every session to date has run with.
BLOB_DETECT_THRESHOLD_WMR = 24


def _blob_detect_threshold(capture: Path) -> tuple[float, str]:
    for base in (capture, capture.parent):
        p = base / "provenance" / "hmd-cameras.json"
        if p.is_file():
            cams = json.loads(p.read_text()).get("cameras", [])
            vals = {c["blob_detect_threshold"] for c in cams if "blob_detect_threshold" in c}
            if vals:
                return float(max(vals)), str(p)
    return float(BLOB_DETECT_THRESHOLD_WMR), "default BLOB_THRESHOLD_MIN_WMR"


def _blob_margin_metrics(candidate: np.ndarray, threshold: float, threshold_src: str) -> dict[str, Any]:
    """Per-device selected-fold blob brightness vs the detection threshold. A device whose
    median selected-blob brightness sits AT the threshold is running with zero photometric
    margin -- the dim-edge failure feeder behind the 2026-07-03 left fly-offs (dev1 median
    23.8 == threshold 24 vs dev2 33.7; c1-left-oov/findings.md)."""
    out: dict[str, Any] = {"threshold": threshold, "threshold_source": threshold_src}
    for dev in (1, 2):
        rows = candidate[(candidate["device_id"] == dev) & (candidate["selected"] != 0)]
        if rows.size == 0:
            continue
        br = rows["blob_brightness_mean"].astype(float)
        br = br[np.isfinite(br) & (br > 0)]
        area = rows["blob_area_mean"].astype(float)
        area = area[np.isfinite(area) & (area > 0)]
        if br.size == 0:
            continue
        med = float(np.median(br))
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "n_selected": int(rows.size),
            "brightness_med": round(med, 1),
            "brightness_p25": round(float(np.percentile(br, 25)), 1),
            "brightness_p10": round(float(np.percentile(br, 10)), 1),
            "margin_med": round(med - threshold, 1),
            "at_or_below_threshold_pct": round(float(100.0 * np.mean(br <= threshold)), 1),
            "area_med": round(float(np.median(area)), 2) if area.size else None,
        }
    return out


#: Capture acceptance gate (audit 2026-06-09): a live capture failing these is a transport
#: repro, not a tracker capture — label it so before drawing tracking conclusions.
GATE_FRAME_MEDIAN_MS = 35.0
GATE_FRAME_GAPS_100MS_MAX = 5
GATE_REQUIRED_STREAMS = ("imu", "frame", "pose_attempt", "fusion", "head_pose")


def capture_gate(report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    rows = report.get("manifest_rows_written", {})
    for s in GATE_REQUIRED_STREAMS:
        checks[f"stream_{s}_nonempty"] = bool(rows.get(s))
    cadence = report.get("frame", {}).get("cadence", {})
    med = (cadence.get("dt_ms") or {}).get("median")
    checks["frame_group_median_le_35ms"] = med is not None and med <= GATE_FRAME_MEDIAN_MS
    gaps = cadence.get("gaps_gt_100ms")
    checks["frame_gaps100_lt_5"] = gaps is not None and gaps < GATE_FRAME_GAPS_100MS_MAX
    src = report.get("event", {}).get("camera_source_delta")
    checks["no_camera_source_gaps"] = src is None or src.get("n", 0) == 0
    checks["PASS"] = all(v for k, v in checks.items() if k != "PASS")
    return checks


def _event_metrics(event: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"counts": {}}
    if event.size == 0:
        return out
    out["counts"] = _top_counts(event["event_type"], EVENT_TYPES, limit=32)
    for etype, key in (
        (17, "tracker_seq_delta"),
        (18, "tracker_blob_ms"),
        (19, "tracker_fast_ms"),
        (21, "camera_source_delta"),
    ):
        rows = event[event["event_type"] == etype]
        if rows.size:
            out[key] = _stat(rows["value"].astype(float))
    dump_drop = event[event["event_type"] == 20]
    if dump_drop.size:
        values = dump_drop["value"].astype(float)
        out["frame_dump_dropped"] = {
            "events": int(dump_drop.size),
            "last_total": float(values[-1]),
            "max_total": float(np.max(values)),
        }
    for dev in (1, 2):
        rows = event[event["device_id"] == dev]
        if rows.size == 0:
            continue
        fold = rows[rows["event_type"] == 7]
        seen = rows[rows["event_type"] == 8]
        item: dict[str, Any] = {"name": DEVICE_NAMES.get(dev, str(dev)), "counts": _top_counts(rows["event_type"], EVENT_TYPES, 32)}
        if fold.size:
            item["eskf_fold_count"] = _stat(fold["value"].astype(float))
            item["eskf_zero_fold_pct"] = float(100.0 * np.sum(fold["value"].astype(float) <= 0.0) / fold.size)
        if seen.size:
            item["eskf_leds_seen"] = _stat(seen["value"].astype(float))
        if fold.size and seen.size:
            item["eskf_folded_over_seen"] = float(np.sum(fold["value"].astype(float)) / max(1e-9, np.sum(seen["value"].astype(float))))
        out[str(dev)] = item
    return out


def _search_metrics(search: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if search.size == 0:
        return out
    for dev in (1, 2):
        rows = search[search["device_id"] == dev]
        if rows.size == 0:
            continue
        work_rows = rows[rows["result"] <= 7]
        stat_rows = work_rows if work_rows.size else rows
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "rows": int(rows.size),
            "work_rows": int(work_rows.size),
            "results": _top_counts(rows["result"], SEARCH_RESULT_NAMES),
            "input_blobs": _stat(stat_rows["input_blobs"].astype(float)),
            "pose_checks": _stat(stat_rows["num_pose_checks"].astype(float)),
            "pose_checks_pruned": _stat(stat_rows["num_pose_checks_pruned"].astype(float)),
        }
    return out


def _candidate_metrics(candidate: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if candidate.size == 0:
        return out
    for dev in (1, 2):
        rows = candidate[candidate["device_id"] == dev]
        if rows.size == 0:
            continue
        selected = rows[rows["selected"] != 0]
        out[str(dev)] = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "rows": int(rows.size),
            "selected": int(selected.size),
            "selected_cadence": _rate_stats(selected["t_mono_ns"]),
            "reproj_err_px": _stat(selected["reproj_err_px"].astype(float)) if selected.size else _stat(np.array([])),
        }
    return out


def _replay_metrics(replay_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dev in (1, 2):
        p = replay_dir / f"dev{dev}.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        t = np.asarray([int(r["t_ns"]) for r in rows], dtype=np.int64)
        opt_valid = np.asarray([int(float(r["opt_valid"])) != 0 for r in rows], dtype=bool)
        opt_kind = np.asarray([int(float(r.get("opt_kind", "0"))) for r in rows], dtype=np.int16)
        opt_led_count = np.asarray([int(float(r.get("opt_led_count", "0"))) for r in rows], dtype=np.int32)
        opt_led_folded = np.asarray([int(float(r.get("opt_led_folded", "0"))) for r in rows], dtype=np.int32)
        pred_tracked = np.asarray([int(float(r.get("pred_tracked", "0"))) != 0 for r in rows], dtype=bool)
        pos = np.asarray([[float(r["pred_px"]), float(r["pred_py"]), float(r["pred_pz"])] for r in rows], dtype=float)
        quat = np.asarray([[float(r["pred_qx"]), float(r["pred_qy"]), float(r["pred_qz"]), float(r["pred_qw"])] for r in rows], dtype=float)
        no_opt_runs = _run_lengths_ms(t, ~opt_valid)
        first_opt_valid_ms = None
        if np.any(opt_valid):
            first_opt_valid_ms = float((t[np.flatnonzero(opt_valid)[0]] - t[0]) * 1e-6)
        first_pred_tracked_ms = None
        if np.any(pred_tracked):
            first_pred_tracked_ms = float((t[np.flatnonzero(pred_tracked)[0]] - t[0]) * 1e-6)
        age_ms = np.full(t.shape, np.nan, dtype=float)
        last_opt = None
        for i, ok in enumerate(opt_valid):
            if ok:
                last_opt = t[i]
                age_ms[i] = 0.0
            elif last_opt is not None:
                age_ms[i] = (t[i] - last_opt) * 1e-6
        item = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "rows": int(len(rows)),
            "cadence": _rate_stats(t),
            "opt_valid_pct": float(100.0 * np.sum(opt_valid) / len(rows)),
            "pred_tracked_pct": float(100.0 * np.sum(pred_tracked) / len(rows)),
            "first_opt_valid_ms": first_opt_valid_ms,
            "first_pred_tracked_ms": first_pred_tracked_ms,
            "no_opt_run_ms": _stat(np.asarray(no_opt_runs, dtype=float)),
            "optical_age_ms": _stat(age_ms),
            **_step_stats(pos, quat, t),
        }
        attempted = opt_led_count > 0
        if np.any(attempted):
            item["replay_led_folded_over_seen"] = float(np.sum(opt_led_folded[attempted]) /
                                                        max(1, np.sum(opt_led_count[attempted])))
            item["replay_zero_fold_pct"] = float(100.0 * np.sum((opt_led_folded == 0) & attempted) /
                                                 np.sum(attempted))
            item["replay_leds_seen"] = _stat(opt_led_count[attempted].astype(float))
            item["replay_leds_folded"] = _stat(opt_led_folded[attempted].astype(float))
            by_kind: dict[str, Any] = {}
            for kind, name in ((1, "pose"), (3, "led_fold")):
                mask = attempted & (opt_kind == kind)
                if np.any(mask):
                    by_kind[name] = {
                        "rows": int(np.sum(mask)),
                        "folded_over_seen": float(np.sum(opt_led_folded[mask]) /
                                                   max(1, np.sum(opt_led_count[mask]))),
                        "zero_fold_pct": float(100.0 * np.sum((opt_led_folded == 0) & mask) /
                                               np.sum(mask)),
                    }
            item["replay_led_fold_by_kind"] = by_kind
        out[str(dev)] = item
    return out


def _render_replay_metrics(replay_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dev in (1, 2):
        p = replay_dir / f"dev{dev}_render.csv"
        if not p.is_file():
            continue
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        t = np.asarray([int(r["t_ns"]) for r in rows], dtype=np.int64)
        pred_tracked = np.asarray([int(float(r.get("pred_tracked", "0"))) != 0 for r in rows], dtype=bool)
        drop_optical = np.asarray([int(float(r.get("drop_optical", "0"))) != 0 for r in rows], dtype=bool)
        pos = np.asarray([[float(r["pred_px"]), float(r["pred_py"]), float(r["pred_pz"])] for r in rows], dtype=float)
        quat = np.asarray([[float(r["pred_qx"]), float(r["pred_qy"]), float(r["pred_qz"]), float(r["pred_qw"])] for r in rows], dtype=float)
        vel = np.asarray(
            [[float(r.get("pred_vx", "nan")), float(r.get("pred_vy", "nan")), float(r.get("pred_vz", "nan"))]
             for r in rows],
            dtype=float,
        )
        optical_age_ms = np.asarray([float(r.get("optical_age_ms", "nan")) for r in rows], dtype=float)
        pred_to_hmd_m = np.asarray([float(r.get("pred_to_hmd_m", "nan")) for r in rows], dtype=float)
        finite = np.isfinite(pos).all(axis=1) & np.isfinite(quat).all(axis=1)
        finite_vel = np.isfinite(vel).all(axis=1)
        usable = pred_tracked & finite
        first_pred_tracked_ms = None
        if np.any(pred_tracked):
            first_pred_tracked_ms = float((t[np.flatnonzero(pred_tracked)[0]] - t[0]) * 1e-6)
        item = {
            "name": DEVICE_NAMES.get(dev, str(dev)),
            "rows": int(len(rows)),
            "cadence": _rate_stats(t),
            "drop_optical_pct": float(100.0 * np.sum(drop_optical) / len(rows)),
            "pred_tracked_pct": float(100.0 * np.sum(pred_tracked) / len(rows)),
            "first_pred_tracked_ms": first_pred_tracked_ms,
            "untracked_run_ms": _stat(np.asarray(_run_lengths_ms(t, ~pred_tracked), dtype=float)),
            "optical_age_ms": _stat(optical_age_ms),
            "pred_to_hmd_cm": _stat(pred_to_hmd_m * 100.0),
            "jitter_cm": _jitter_metric(t[usable], pos[usable]),
            **_step_stats(pos[usable], quat[usable], t[usable]),
        }
        if np.any(finite_vel):
            speed = np.linalg.norm(vel, axis=1)
            item["linear_speed_m_s"] = _stat(speed[usable & finite_vel])
            if "kf_body_anchored" in rows[0]:
                body = np.asarray([int(float(r.get("kf_body_anchored", "0"))) != 0 for r in rows], dtype=bool)
                body_usable = usable & finite_vel & body
                item["body_locked_linear_speed_m_s"] = _stat(speed[body_usable])
                if np.any(body_usable):
                    item["body_locked_velocity_nonzero_pct"] = float(
                        100.0 * np.sum(speed[body_usable] > 1.0e-4) / np.sum(body_usable)
                    )
        out[str(dev)] = item
    return out


def build_report(capture: Path, replay_dir: Path | None = None) -> dict[str, Any]:
    d = _telemetry_dir(capture)
    m = Manifest.load(d)
    streams = {name: _read_stream(d, m, name) for name in m.streams}
    report: dict[str, Any] = {
        "capture": str(capture),
        "telemetry_dir": str(d),
        "manifest_rows_written": {name: m.streams[name].rows_written for name in m.streams},
        "streams": {name: int(arr.size) for name, arr in streams.items()},
    }
    if "frame" in streams:
        report["frame"] = _frame_metrics(streams["frame"])
    if "head_pose" in streams:
        report["head_pose"] = {"cadence": _rate_stats(streams["head_pose"]["t_mono_ns"])}
    if "imu" in streams:
        report["imu"] = _imu_metrics(streams["imu"])
    if "pose_attempt" in streams:
        report["pose_attempt"] = _pose_metrics(streams["pose_attempt"])
    if "fusion" in streams:
        report["fusion"] = _fusion_metrics(streams["fusion"])
        t0_ns = int(streams["imu"]["t_mono_ns"].min()) if streams.get("imu") is not None and streams["imu"].size else None
        report["reentry"] = _reentry_metrics(streams["fusion"], t0_ns)
    if "event" in streams:
        report["event"] = _event_metrics(streams["event"])
    if "search" in streams:
        report["search"] = _search_metrics(streams["search"])
    if "candidate" in streams:
        report["candidate"] = _candidate_metrics(streams["candidate"])
        threshold, threshold_src = _blob_detect_threshold(capture)
        report["blob_margin"] = _blob_margin_metrics(streams["candidate"], threshold, threshold_src)
    if replay_dir is not None:
        report["replay"] = _replay_metrics(replay_dir)
        report["render_replay"] = _render_replay_metrics(replay_dir)
    return report


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"
    return str(v)


def print_summary(report: dict[str, Any]) -> None:
    print(f"# {report['capture']}")
    print(f"streams: {report['streams']}")
    frame = report.get("frame", {})
    if frame:
        c = frame["cadence"]
        b = frame["blob_total_per_group"]
        sseq = frame.get("source_sequence", {})
        fwd = sseq.get("delta_forward", {})
        print(
            "frame: "
            f"groups={frame['groups']} complete4={frame['complete_4cam_groups']} "
            f"hz={_fmt(c['hz'], 2)} dt_med/p95/max_ms="
            f"{_fmt(c['dt_ms']['median'])}/{_fmt(c['dt_ms']['p95'])}/{_fmt(c['dt_ms']['max'])} "
            f"gaps>100/200ms={c['gaps_gt_100ms']}/{c['gaps_gt_200ms']} "
            f"blob_total_med/p95={_fmt(b['median'])}/{_fmt(b['p95'])}"
        )
        if sseq:
            print(
                "frame_source_seq: "
                f"delta_fwd_med/p95/max={_fmt(fwd.get('median'))}/{_fmt(fwd.get('p95'))}/{_fmt(fwd.get('max'))} "
                f"gaps>2/5/10={sseq.get('gaps_gt_2', 0)}/{sseq.get('gaps_gt_5', 0)}/{sseq.get('gaps_gt_10', 0)} "
                f"hard>100={sseq.get('hard_jumps_gt_100', 0)} "
                f"backwards={sseq.get('backward_jumps', 0)} mixed4={sseq.get('mixed_4cam_groups', 0)}"
            )
    hp = report.get("head_pose", {}).get("cadence")
    if hp:
        print(
            "head_pose: "
            f"hz={_fmt(hp['hz'], 2)} dt_med/p95/max_ms="
            f"{_fmt(hp['dt_ms']['median'])}/{_fmt(hp['dt_ms']['p95'])}/{_fmt(hp['dt_ms']['max'])}"
        )
    event = report.get("event", {})
    if event.get("camera_source_delta"):
        seq = event["camera_source_delta"]
        print(
            "camera_source_delta: "
            f"events={seq['n']} delta_med/p95/max={_fmt(seq['median'])}/{_fmt(seq['p95'])}/{_fmt(seq['max'])}"
        )
    if event.get("tracker_seq_delta"):
        seq = event["tracker_seq_delta"]
        print(
            "tracker_seq_delta: "
            f"events={seq['n']} delta_med/p95/max={_fmt(seq['median'])}/{_fmt(seq['p95'])}/{_fmt(seq['max'])}"
        )
    if event.get("tracker_blob_ms") or event.get("tracker_fast_ms"):
        blob = event.get("tracker_blob_ms", {})
        fast = event.get("tracker_fast_ms", {})
        print(
            "tracker_timing_ms: "
            f"blob_med/p95/max={_fmt(blob.get('median'))}/{_fmt(blob.get('p95'))}/{_fmt(blob.get('max'))} "
            f"fast_med/p95/max={_fmt(fast.get('median'))}/{_fmt(fast.get('p95'))}/{_fmt(fast.get('max'))}"
        )
    if event.get("frame_dump_dropped"):
        drop = event["frame_dump_dropped"]
        print(
            "frame_dump_dropped: "
            f"events={drop['events']} last_total={_fmt(drop['last_total'], 0)} max_total={_fmt(drop['max_total'], 0)}"
        )
    for dev in ("1", "2"):
        name = DEVICE_NAMES.get(int(dev), dev)
        imu = report.get("imu", {}).get(dev, {})
        pose = report.get("pose_attempt", {}).get(dev, {})
        fus = report.get("fusion", {}).get(dev, {})
        ev = report.get("event", {}).get(dev, {})
        search = report.get("search", {}).get(dev, {})
        replay = report.get("replay", {}).get(dev, {})
        render = report.get("render_replay", {}).get(dev, {})
        print(f"\n{dev} {name}:")
        if imu:
            print(f"  imu_hz={_fmt(imu['cadence']['hz'], 1)} gyro_p95={_fmt(imu['gyro_norm_rad_s']['p95'], 2)}rad/s")
        if pose:
            pc = pose["accepted_cadence"]
            print(
                f"  pose_accept_hz={_fmt(pc['hz'], 2)} dt_p95/max_ms="
                f"{_fmt(pc['dt_ms']['p95'])}/{_fmt(pc['dt_ms']['max'])} "
                f"reproj_med/p95_px={_fmt(pose['reproj_err_px']['median'], 2)}/{_fmt(pose['reproj_err_px']['p95'], 2)}"
            )
        if fus:
            fc = fus["cadence"]
            print(
                f"  fusion_hz={_fmt(fc['hz'], 2)} gap_p95/max_ms={_fmt(fus['gap_ms']['p95'])}/{_fmt(fus['gap_ms']['max'])} "
                f"resid_cm_med/p95={_fmt(fus['pos_residual_cm']['median'], 1)}/{_fmt(fus['pos_residual_cm']['p95'], 1)} "
                f"resid_deg_med/p95={_fmt(fus['rot_residual_deg']['median'], 1)}/{_fmt(fus['rot_residual_deg']['p95'], 1)} "
                f"step_cm_p95={_fmt(fus['pos_step_cm']['p95'], 1)} step_deg_p95={_fmt(fus['rot_step_deg']['p95'], 1)}"
            )
        if ev and "eskf_fold_count" in ev:
            print(
                f"  eskf_folded/seen={_fmt(ev.get('eskf_folded_over_seen'), 3)} "
                f"zero_fold_pct={_fmt(ev.get('eskf_zero_fold_pct'), 1)} "
                f"seen_med={_fmt(ev['eskf_leds_seen']['median'], 1)} fold_med={_fmt(ev['eskf_fold_count']['median'], 1)}"
            )
        margin = report.get("blob_margin", {}).get(dev, {})
        if margin:
            thr = report["blob_margin"]["threshold"]
            flag = "  ** ZERO-MARGIN **" if margin["margin_med"] <= 0.5 else ""
            print(
                f"  blob_brightness_med/p25={_fmt(margin['brightness_med'], 1)}/{_fmt(margin['brightness_p25'], 1)} "
                f"vs threshold {_fmt(thr, 0)} (margin_med={_fmt(margin['margin_med'], 1)}, "
                f"<=thr {_fmt(margin['at_or_below_threshold_pct'], 1)}%) "
                f"area_med={_fmt(margin['area_med'], 2)}{flag}"
            )
        reentry = report.get("reentry", {}).get(dev, {})
        if reentry:
            rr = reentry["reentry_rot_deg"]
            print(
                f"  reentries={reentry['reentries']} rot_med/p95/max_deg="
                f"{_fmt(rr['median'], 1)}/{_fmt(rr['p95'], 1)}/{_fmt(rr['max'], 1)} "
                f"rot>30deg={reentry['rot_gt30deg']}"
            )
            for e in reentry["worst_rot"]:
                if e["rot_deg"] <= REENTRY_ROT_FLAG_DEG:
                    break
                print(
                    f"    reentry t+{e['t_re_rel_s']}s gap={e['gap_s']}s "
                    f"snap={e['snap_cm']}cm rot={e['rot_deg']}deg"
                )
        if search:
            print(
                f"  search_rows={search['work_rows']}/{search['rows']} work/total "
                f"search_results={search['results']} "
                f"input_blobs_med/p95={_fmt(search['input_blobs']['median'])}/{_fmt(search['input_blobs']['p95'])} "
                f"pose_checks_p95={_fmt(search['pose_checks']['p95'], 0)}"
            )
        if replay:
            print(
                f"  replay_opt_pct={_fmt(replay['opt_valid_pct'], 1)} tracked_pct={_fmt(replay['pred_tracked_pct'], 1)} "
                f"first_opt/tracked_ms={_fmt(replay.get('first_opt_valid_ms'))}/{_fmt(replay.get('first_pred_tracked_ms'))} "
                f"opt_age_p95/max_ms={_fmt(replay['optical_age_ms']['p95'])}/{_fmt(replay['optical_age_ms']['max'])} "
                f"no_opt_run_p95/max_ms={_fmt(replay['no_opt_run_ms']['p95'])}/{_fmt(replay['no_opt_run_ms']['max'])} "
                f"report_step_p95_cm/deg={_fmt(replay['pos_step_cm']['p95'], 1)}/{_fmt(replay['rot_step_deg']['p95'], 1)}"
            )
        if render:
            rc = render["cadence"]
            jit = render["jitter_cm"]
            print(
                f"  render_hz={_fmt(rc['hz'], 1)} tracked_pct={_fmt(render['pred_tracked_pct'], 1)} "
                f"first_tracked_ms={_fmt(render.get('first_pred_tracked_ms'))} "
                f"untracked_run_p95/max_ms={_fmt(render['untracked_run_ms']['p95'])}/{_fmt(render['untracked_run_ms']['max'])} "
                f"opt_age_p95/max_ms={_fmt(render['optical_age_ms']['p95'])}/{_fmt(render['optical_age_ms']['max'])} "
                f"step_p95_cm/deg={_fmt(render['pos_step_cm']['p95'], 1)}/{_fmt(render['rot_step_deg']['p95'], 1)} "
                f"jitter_rms/p95_cm={_fmt(jit['rms_cm'], 2)}/{_fmt(jit['p95_cm'], 2)} "
                f"head_dist_p95_cm={_fmt(render['pred_to_hmd_cm']['p95'], 1)}"
            )
            body_speed = render.get("body_locked_linear_speed_m_s")
            if body_speed:
                print(
                    f"  body_locked_speed_med/p95_m_s={_fmt(body_speed['median'], 3)}/{_fmt(body_speed['p95'], 3)} "
                    f"nonzero_pct={_fmt(render.get('body_locked_velocity_nonzero_pct'), 1)}"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path, help="capture dir or telemetry dir")
    ap.add_argument("--replay-dir", type=Path, help="optional offline_vio_replay output dir with dev*.csv")
    ap.add_argument("--json", type=Path, help="write JSON report")
    ap.add_argument("--quiet", action="store_true", help="suppress text summary")
    args = ap.parse_args()

    report = build_report(args.capture, args.replay_dir)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not args.quiet:
        print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
