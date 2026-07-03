#!/usr/bin/env python3
"""analyze_capture.py -- diagnostic distributions + symptom-regime slices on a capture.

Goal: answer the user's question "what specifically went wrong in MY capture and why?"
without baking another threshold-based binary classification. We compute distributions
and per-regime slices of the per-attempt pose error against the cleaned-GT reference.

Per (device, cam, motion-regime) we report:
  - pos_err_cm distribution: median, p25, p75, p95, p99, max
  - ori_err_deg distribution: same
  - accept count, in-view count
  - %-correct at multiple thresholds (5cm/15deg, 10cm/30deg, 25cm/60deg)

Motion regimes:
  - speed: stationary (<0.1 m/s), slow (0.1-0.5), fast (0.5-2.0), very fast (>2.0)
  - controller-vs-head Z position: below (z<-0.1m relative to head), level (-0.1..0.1), above (>0.1)
  - controller-vs-head lateral offset: centered (<0.3), side (>0.3 absolute X or Y)

We REUSE detection_f1.py's pose composition (already verified to produce 76-85% F1
correctness on this capture, so the math is mostly right).

Outputs:
  - /tmp/analyze_<cap_basename>.txt: human-readable report
  - /tmp/analyze_<cap_basename>.json: machine-readable dump

Usage:
    analyze_capture.py <capture_dir> [--cams JSON] [--ctrl-left JSON] [--ctrl-right JSON]
                       [--out-prefix /tmp/analyze]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest, DEVICE_NAMES  # noqa: E402
from smooth_ref import build_reference  # noqa: E402
import g2_geom as G  # noqa: E402
from detection_f1 import (  # noqa: E402
    quat_to_R, quat_geodesic_deg, R_to_quat, pose_mul, pose_inv, pose_flip_YZ,
    load_cameras, load_led_model, in_fov_per_cam,
    pose_attempt_to_xrworld_device, _interp_quat_pos, _interp_reference_to_times,
    _load_head_pose, P_YZ_FLIP_R,
    DEFAULT_CAMS, DEFAULT_CTRL_LEFT, DEFAULT_CTRL_RIGHT,
)


SPEED_BINS_M_S = [(0.0, 0.1, "static"), (0.1, 0.5, "slow"), (0.5, 2.0, "fast"), (2.0, 1e9, "vfast")]
VERTICAL_BINS = [(-1e9, -0.10, "below"), (-0.10, 0.10, "level"), (0.10, 1e9, "above")]
LATERAL_BINS = [(0.0, 0.30, "centered"), (0.30, 1e9, "side")]

CORRECTNESS_THRESHOLDS = [(5.0, 15.0, "tight"), (10.0, 30.0, "moderate"), (25.0, 60.0, "loose")]


def _pct(arr, p):
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def _summarize(values):
    if len(values) == 0:
        return {"n": 0}
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"n": 0}
    return {
        "n": int(len(a)),
        "med": float(np.median(a)),
        "p25": _pct(a, 25),
        "p75": _pct(a, 75),
        "p95": _pct(a, 95),
        "p99": _pct(a, 99),
        "max": float(np.max(a)),
    }


def _classify_speed(speed_m_s):
    for lo, hi, name in SPEED_BINS_M_S:
        if lo <= speed_m_s < hi:
            return name
    return "vfast"


def _classify_vertical(dz):
    for lo, hi, name in VERTICAL_BINS:
        if lo <= dz < hi:
            return name
    return "above"


def _classify_lateral(dxy):
    for lo, hi, name in LATERAL_BINS:
        if lo <= dxy < hi:
            return name
    return "side"


def _quat_inv(q):
    """Quaternion inverse for unit quaternion (x,y,z,w)."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def verify_composition(telem: Path, dev: int, cams, ctrl_json: str, hp_t, hp_pos, hp_q):
    """Pose-composition sanity check.

    Pick the SINGLE cleanest accepted attempt by score (most matched blobs, smallest reproj),
    reconstruct world pose, compare with the cleaned-GT reference at that t. If we're <1cm/<2deg
    on a "perfect" attempt, composition is right.
    """
    m = Manifest.load(telem)
    pa = G.load_stream(telem, m, "pose_attempt")
    pa = pa[(pa["device_id"] == dev) & ((pa["outcome"] == 1) | (pa["outcome"] == 2))]
    if len(pa) == 0:
        return None
    # Best = most matched + lowest reproj
    score = pa["blobs_matched"].astype(int) * 10 - pa["reproj_err_px"].astype(float)
    best_idx = int(np.argmax(score))
    row = pa[best_idx]
    cam_id = int(row["cam_id"])
    if cam_id >= len(cams):
        return None
    cam = cams[cam_id]
    t = int(row["hw_ts_ns"])
    hp_q_t, hp_p_t = _interp_quat_pos(hp_t, hp_q, hp_pos, np.array([t]))
    if not np.isfinite(hp_p_t[0, 0]):
        return None
    R_wo, t_wo = pose_attempt_to_xrworld_device(row, cam, hp_q_t[0], hp_p_t[0])
    q_wo = R_to_quat(R_wo)
    # Compare with cleaned-GT
    ref = build_reference(telem, dev)
    if ref is None:
        return None
    # Nearest reference sample within match_ms = 25
    j = np.searchsorted(ref.t_ns, t)
    best = -1
    for cand in (j - 1, j):
        if 0 <= cand < ref.t_ns.shape[0]:
            if abs(int(ref.t_ns[cand]) - t) <= int(25e6):
                if best < 0 or abs(int(ref.t_ns[cand]) - t) < abs(int(ref.t_ns[best]) - t):
                    best = cand
    if best < 0:
        return None
    pos_err_cm = float(np.linalg.norm(t_wo - ref.pos[best]) * 100.0)
    ori_err_deg = float(quat_geodesic_deg(q_wo, ref.quat[best]))
    return {
        "device": dev,
        "cam": cam_id,
        "matched_blobs": int(row["blobs_matched"]),
        "reproj_err_px": float(row["reproj_err_px"]),
        "ref_idx": best,
        "ref_quality": int(ref.quality[best]) if hasattr(ref, "quality") and ref.quality is not None else None,
        "t_ns": t,
        "reconstructed_pos": t_wo.tolist(),
        "ref_pos": ref.pos[best].tolist(),
        "pos_err_cm": pos_err_cm,
        "reconstructed_quat": q_wo.tolist(),
        "ref_quat": ref.quat[best].tolist(),
        "ori_err_deg": ori_err_deg,
    }


def analyse_device(telem: Path, dev: int, cams, ctrl_json: str,
                   hp_t, hp_pos, hp_q, match_ms: float, max_ref_gap_ms: float):
    led_pos, led_nrm = load_led_model(ctrl_json)
    ref = build_reference(telem, dev)
    if ref is None:
        return None

    m = Manifest.load(telem)
    pa = G.load_stream(telem, m, "pose_attempt")
    pa = pa[pa["device_id"] == dev]
    accepted = pa[(pa["outcome"] == 1) | (pa["outcome"] == 2)]
    if len(accepted) == 0:
        return None
    t_acc = accepted["hw_ts_ns"].astype(np.int64)

    # Cleaned-GT velocity for speed-binning
    t_ref = ref.t_ns
    pos_ref = ref.pos
    quat_ref = ref.quat
    valid_ref = ref.valid
    # Central differences in mm/ns; convert to m/s
    speed = np.zeros(t_ref.shape[0], dtype=float)
    for i in range(1, t_ref.shape[0] - 1):
        dt = (t_ref[i + 1] - t_ref[i - 1]) / 1e9
        if dt > 0 and valid_ref[i - 1] and valid_ref[i + 1]:
            speed[i] = np.linalg.norm(pos_ref[i + 1] - pos_ref[i - 1]) / dt

    # Per-attempt error + regime
    per_attempt = []
    for k in range(len(accepted)):
        row = accepted[k]
        t = int(t_acc[k])
        cam_id = int(row["cam_id"])
        if cam_id >= len(cams):
            continue
        cam = cams[cam_id]
        # Reference at attempt time (use _interp_reference_to_times-style)
        j = np.searchsorted(t_ref, t)
        best = -1
        for cand in (j - 1, j):
            if 0 <= cand < t_ref.shape[0] and valid_ref[cand]:
                if abs(int(t_ref[cand]) - t) <= int(match_ms * 1e6):
                    if best < 0 or abs(int(t_ref[cand]) - t) < abs(int(t_ref[best]) - t):
                        best = cand
        if best < 0:
            continue
        # Head pose at attempt time
        hp_q_t, hp_p_t = _interp_quat_pos(hp_t, hp_q, hp_pos, np.array([t]))
        if not np.isfinite(hp_p_t[0, 0]):
            continue
        # Reconstruct world pose
        R_wo, t_wo = pose_attempt_to_xrworld_device(row, cam, hp_q_t[0], hp_p_t[0])
        q_wo = R_to_quat(R_wo)
        pos_err_cm = float(np.linalg.norm(t_wo - pos_ref[best]) * 100.0)
        ori_err_deg = float(quat_geodesic_deg(q_wo, quat_ref[best]))
        # Motion regime
        sp = float(speed[best])
        dz = float(pos_ref[best, 2] - hp_p_t[0, 2])     # controller Z minus head Z (world)
        dxy = float(np.linalg.norm(pos_ref[best, :2] - hp_p_t[0, :2]))
        per_attempt.append({
            "t_ns": t,
            "cam_id": cam_id,
            "matched": int(row["blobs_matched"]),
            "visible": int(row["leds_visible"]),
            "reproj_err_px": float(row["reproj_err_px"]),
            "outcome": int(row["outcome"]),
            "pos_err_cm": pos_err_cm,
            "ori_err_deg": ori_err_deg,
            "speed_m_s": sp,
            "dz_m": dz,
            "dxy_m": dxy,
            "speed_bin": _classify_speed(sp),
            "vertical_bin": _classify_vertical(dz),
            "lateral_bin": _classify_lateral(dxy),
        })

    if not per_attempt:
        return None
    pa_arr = per_attempt

    # --- aggregations ---
    pos_errs = [x["pos_err_cm"] for x in pa_arr]
    ori_errs = [x["ori_err_deg"] for x in pa_arr]
    summary = {
        "device": dev,
        "n_accepted": len(pa_arr),
        "pos_err_cm": _summarize(pos_errs),
        "ori_err_deg": _summarize(ori_errs),
    }

    # Correctness at multiple thresholds (no binary classification — show the curve)
    thresh_curve = {}
    for pcm, odeg, name in CORRECTNESS_THRESHOLDS:
        ok = sum(1 for x in pa_arr if x["pos_err_cm"] < pcm and x["ori_err_deg"] < odeg)
        thresh_curve[name] = {
            "pos_cm": pcm, "ori_deg": odeg,
            "pct_correct": 100.0 * ok / len(pa_arr),
            "n_correct": ok,
        }
    summary["correctness_curve"] = thresh_curve

    # Per-cam
    by_cam = {}
    for j in range(len(cams)):
        sub = [x for x in pa_arr if x["cam_id"] == j]
        if not sub:
            by_cam[j] = {"n": 0}
            continue
        by_cam[j] = {
            "n": len(sub),
            "pos_err_cm": _summarize([x["pos_err_cm"] for x in sub]),
            "ori_err_deg": _summarize([x["ori_err_deg"] for x in sub]),
            "pct_pos_lt_5cm": 100.0 * sum(1 for x in sub if x["pos_err_cm"] < 5) / len(sub),
            "pct_pos_lt_25cm": 100.0 * sum(1 for x in sub if x["pos_err_cm"] < 25) / len(sub),
            "median_matched": int(np.median([x["matched"] for x in sub])),
            "median_visible": int(np.median([x["visible"] for x in sub])),
        }
    summary["by_cam"] = by_cam

    # Per speed regime
    by_speed = {}
    for _, _, name in SPEED_BINS_M_S:
        sub = [x for x in pa_arr if x["speed_bin"] == name]
        if not sub:
            by_speed[name] = {"n": 0}
            continue
        by_speed[name] = {
            "n": len(sub),
            "pos_err_cm": _summarize([x["pos_err_cm"] for x in sub]),
            "ori_err_deg": _summarize([x["ori_err_deg"] for x in sub]),
            "pct_pos_lt_5cm": 100.0 * sum(1 for x in sub if x["pos_err_cm"] < 5) / len(sub),
            "pct_pos_lt_25cm": 100.0 * sum(1 for x in sub if x["pos_err_cm"] < 25) / len(sub),
        }
    summary["by_speed"] = by_speed

    # Per vertical regime (above/below/level head)
    by_vertical = {}
    for _, _, name in VERTICAL_BINS:
        sub = [x for x in pa_arr if x["vertical_bin"] == name]
        if not sub:
            by_vertical[name] = {"n": 0}
            continue
        by_vertical[name] = {
            "n": len(sub),
            "pos_err_cm": _summarize([x["pos_err_cm"] for x in sub]),
            "ori_err_deg": _summarize([x["ori_err_deg"] for x in sub]),
            "pct_pos_lt_5cm": 100.0 * sum(1 for x in sub if x["pos_err_cm"] < 5) / len(sub),
        }
    summary["by_vertical"] = by_vertical

    # Per lateral regime
    by_lateral = {}
    for _, _, name in LATERAL_BINS:
        sub = [x for x in pa_arr if x["lateral_bin"] == name]
        if not sub:
            by_lateral[name] = {"n": 0}
            continue
        by_lateral[name] = {
            "n": len(sub),
            "pos_err_cm": _summarize([x["pos_err_cm"] for x in sub]),
            "ori_err_deg": _summarize([x["ori_err_deg"] for x in sub]),
            "pct_pos_lt_5cm": 100.0 * sum(1 for x in sub if x["pos_err_cm"] < 5) / len(sub),
        }
    summary["by_lateral"] = by_lateral

    # Worst attempts (top-20 by pos_err): timestamps for blobviz follow-up
    pa_sorted = sorted(pa_arr, key=lambda x: -x["pos_err_cm"])
    summary["worst_pos_err"] = [{
        "t_ns": x["t_ns"], "cam_id": x["cam_id"],
        "pos_err_cm": x["pos_err_cm"], "ori_err_deg": x["ori_err_deg"],
        "matched": x["matched"], "visible": x["visible"],
        "reproj_err_px": x["reproj_err_px"],
        "speed_m_s": x["speed_m_s"], "dz_m": x["dz_m"], "dxy_m": x["dxy_m"],
        "vertical_bin": x["vertical_bin"], "lateral_bin": x["lateral_bin"],
    } for x in pa_sorted[:20]]

    return summary


def fmt_dist(d):
    if d.get("n", 0) == 0:
        return "n=0"
    return f"n={d['n']} med={d['med']:.2f} p75={d['p75']:.2f} p95={d['p95']:.2f} p99={d['p99']:.2f} max={d['max']:.2f}"


def fmt_summary_text(s, cams_n):
    lines = []
    dev_name = DEVICE_NAMES.get(s["device"], str(s["device"]))
    lines.append(f"\n=== device {dev_name} (dev {s['device']}, n_accepted = {s['n_accepted']}) ===")
    lines.append(f"  overall pos_err_cm: {fmt_dist(s['pos_err_cm'])}")
    lines.append(f"  overall ori_err_deg: {fmt_dist(s['ori_err_deg'])}")
    lines.append("\n  correctness curve (pct of accepted attempts within threshold):")
    for name, c in s["correctness_curve"].items():
        lines.append(f"    {name:>10s}  pos<{c['pos_cm']:>5.1f}cm + ori<{c['ori_deg']:>5.1f}deg  =>  {c['pct_correct']:6.1f}%  ({c['n_correct']}/{s['n_accepted']})")
    lines.append("\n  by cam (controller LED frames per cam):")
    lines.append(f"    {'cam':>4s}  {'n':>5s}  {'<5cm%':>8s}  {'<25cm%':>8s}  {'pos_p50':>8s}  {'pos_p95':>8s}  {'ori_p50':>8s}  {'ori_p95':>8s}  {'med_m':>5s}  {'med_v':>5s}")
    for j in range(cams_n):
        c = s["by_cam"].get(j, {"n": 0})
        if c["n"] == 0:
            lines.append(f"    {j:>4d}  {'0':>5s}  {'-':>8s}  {'-':>8s}  {'-':>8s}  {'-':>8s}  {'-':>8s}  {'-':>8s}  {'-':>5s}  {'-':>5s}")
        else:
            pe = c["pos_err_cm"]
            oe = c["ori_err_deg"]
            lines.append(f"    {j:>4d}  {c['n']:>5d}  {c['pct_pos_lt_5cm']:>7.1f}%  {c['pct_pos_lt_25cm']:>7.1f}%  {pe['med']:>8.2f}  {pe['p95']:>8.2f}  {oe['med']:>8.2f}  {oe['p95']:>8.2f}  {c['median_matched']:>5d}  {c['median_visible']:>5d}")
    lines.append("\n  by speed regime (cleaned-GT velocity):")
    lines.append(f"    {'bin':>8s}  {'n':>5s}  {'<5cm%':>8s}  {'pos_p50':>8s}  {'pos_p95':>8s}  {'ori_p50':>8s}  {'ori_p95':>8s}")
    for _, _, name in SPEED_BINS_M_S:
        c = s["by_speed"].get(name, {"n": 0})
        if c["n"] == 0:
            lines.append(f"    {name:>8s}  {'0':>5s}")
        else:
            pe = c["pos_err_cm"]; oe = c["ori_err_deg"]
            lines.append(f"    {name:>8s}  {c['n']:>5d}  {c['pct_pos_lt_5cm']:>7.1f}%  {pe['med']:>8.2f}  {pe['p95']:>8.2f}  {oe['med']:>8.2f}  {oe['p95']:>8.2f}")
    lines.append("\n  by vertical regime (controller-Z relative to head-Z, world frame):")
    lines.append(f"    {'bin':>8s}  {'n':>5s}  {'<5cm%':>8s}  {'pos_p50':>8s}  {'pos_p95':>8s}  {'ori_p50':>8s}  {'ori_p95':>8s}")
    for _, _, name in VERTICAL_BINS:
        c = s["by_vertical"].get(name, {"n": 0})
        if c["n"] == 0:
            lines.append(f"    {name:>8s}  {'0':>5s}")
        else:
            pe = c["pos_err_cm"]; oe = c["ori_err_deg"]
            lines.append(f"    {name:>8s}  {c['n']:>5d}  {c['pct_pos_lt_5cm']:>7.1f}%  {pe['med']:>8.2f}  {pe['p95']:>8.2f}  {oe['med']:>8.2f}  {oe['p95']:>8.2f}")
    lines.append("\n  by lateral regime (controller XY distance from head, world frame):")
    for _, _, name in LATERAL_BINS:
        c = s["by_lateral"].get(name, {"n": 0})
        if c["n"] == 0:
            lines.append(f"    {name:>8s}  {'0':>5s}")
        else:
            pe = c["pos_err_cm"]; oe = c["ori_err_deg"]
            lines.append(f"    {name:>8s}  {c['n']:>5d}  {c['pct_pos_lt_5cm']:>7.1f}%  {pe['med']:>8.2f}  {pe['p95']:>8.2f}  {oe['med']:>8.2f}  {oe['p95']:>8.2f}")
    lines.append("\n  WORST 20 by pos_err (for blobviz follow-up):")
    lines.append(f"    {'pos_cm':>7s}  {'ori_deg':>7s}  {'cam':>4s}  {'spd_ms':>6s}  {'dz_m':>6s}  {'dxy':>6s}  {'matched':>7s}  {'visible':>7s}  {'reproj':>6s}  {'t_ns':>16s}  {'vert':>6s}  {'lat':>8s}")
    for x in s["worst_pos_err"]:
        lines.append(f"    {x['pos_err_cm']:>7.2f}  {x['ori_err_deg']:>7.2f}  {x['cam_id']:>4d}  {x['speed_m_s']:>6.2f}  {x['dz_m']:>6.2f}  {x['dxy_m']:>6.2f}  {x['matched']:>7d}  {x['visible']:>7d}  {x['reproj_err_px']:>6.2f}  {x['t_ns']:>16d}  {x['vertical_bin']:>6s}  {x['lateral_bin']:>8s}")
    return "\n".join(lines)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--cams", default=DEFAULT_CAMS)
    ap.add_argument("--ctrl-left", default=DEFAULT_CTRL_LEFT)
    ap.add_argument("--ctrl-right", default=DEFAULT_CTRL_RIGHT)
    ap.add_argument("--match-ms", type=float, default=25.0)
    ap.add_argument("--max-ref-gap-ms", type=float, default=150.0)
    ap.add_argument("--out-prefix", default="/tmp/analyze")
    args = ap.parse_args()

    capture = Path(args.capture)
    telem = capture / "telemetry"
    cams = load_cameras(args.cams)
    print(f"Loaded {len(cams)} cameras\n")

    hp = _load_head_pose(telem)
    if hp is None:
        print("no head_pose.bin", file=sys.stderr)
        return 1
    hp_t, hp_pos, hp_q = hp

    out_txt_lines = []
    out_obj = {"capture": str(capture), "verification": [], "devices": []}

    # ---- composition verification ----
    out_txt_lines.append("=== POSE COMPOSITION VERIFICATION ===")
    out_txt_lines.append("Pick the cleanest-looking accepted attempt; reconstruct world pose; compare to cleaned-GT.")
    out_txt_lines.append("If pos<1cm AND ori<2deg, composition is correct.")
    for dev, ctrl in ((1, args.ctrl_left), (2, args.ctrl_right)):
        v = verify_composition(telem, dev, cams, ctrl, hp_t, hp_pos, hp_q)
        if v is None:
            out_txt_lines.append(f"  dev {dev}: no verifiable attempt")
            continue
        out_obj["verification"].append(v)
        out_txt_lines.append(
            f"  dev {dev} cam {v['cam']} matched={v['matched_blobs']} reproj={v['reproj_err_px']:.2f}px "
            f"=> pos_err={v['pos_err_cm']:.2f}cm  ori_err={v['ori_err_deg']:.2f}deg")

    # ---- per-device distributions ----
    for dev, ctrl in ((1, args.ctrl_left), (2, args.ctrl_right)):
        s = analyse_device(telem, dev, cams, ctrl,
                           hp_t, hp_pos, hp_q,
                           args.match_ms, args.max_ref_gap_ms)
        if s is None:
            continue
        out_obj["devices"].append(s)
        out_txt_lines.append(fmt_summary_text(s, len(cams)))

    out_txt = "\n".join(out_txt_lines)
    print(out_txt)

    cap_tag = capture.name
    Path(f"{args.out_prefix}_{cap_tag}.txt").write_text(out_txt)
    with open(f"{args.out_prefix}_{cap_tag}.json", "w") as f:
        json.dump(_json_safe(out_obj), f, indent=2)
    print(f"\nwrote {args.out_prefix}_{cap_tag}.txt and .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
