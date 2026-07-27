#!/usr/bin/env python3
"""run_ab.py -- one-command A/B of the G2 controller tracker on a capture.

Runs the offline_vio_replay harness on BOTH controllers (and, optionally, a baseline binary vs
a candidate binary) over one or more captures, scores each run with mse_eval against the
self-consistent cleaned-GT reference AND with headpose_anchor against the INDEPENDENT,
non-drifting head-pose anchor, and prints ONE comparison table covering every failure mode:

    pos RMSE/median | ori RMS/wrong-branch | consecutive flip-rate
    fly-off (max/p99 jump, beyond-arm-reach) | jitter (RMS) | yield/latency
    independent anchor: tilt-flip% + PURE-YAW-flip% (vs the non-drifting head yaw)

Because the G2_NO_FLIP_VETO toggle has been removed from the tree (the veto is now the single
unified path), the default A/B is column-wise: the optical front-end stream (`opt`) vs the ESKF
fused prediction (`pred`) of the SAME current binary. Pass --baseline-bin to additionally A/B a
second (e.g. previously-built) binary against --candidate-bin.

Usage:
  run_ab.py --capture DIR [--capture DIR ...] --bin PATH
  run_ab.py --capture DIR [--capture DIR ...] --baseline-bin PATH --candidate-bin PATH
            [--cams JSON] [--ctrl-left JSON] [--ctrl-right JSON]
            [--cols opt,pred] [--out DIR]

Conda: PYTHONNOUSERSITE=1 ~/miniconda3/envs/g2vr/bin/python run_ab.py ...
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from smooth_ref import build_reference
from mse_eval import load_candidate_csv, compute_metrics, csv_row_count
import headpose_anchor as HA
from frame_view import frame_source
from manifest import DEVICE_NAMES
from replay_contract import cams_for_capture, imu_cal_dir_for_capture, replay_env


def _find_controller_jsons(explicit_left, explicit_right):
    """Resolve the per-hand controller jsons (serials differ per unit); prefer explicit paths."""
    wmr = Path(os.path.expanduser("~/.config/monado/wmr"))
    left = explicit_left or next((str(p) for p in sorted(wmr.glob("controller_*L.json"))), None)
    right = explicit_right or next((str(p) for p in sorted(wmr.glob("controller_*R.json"))), None)
    return left, right


def run_replay_dual(binary, frames, cams, telem, ctrl_left, ctrl_right, out_dir, capture) -> bool:
    """Invoke the dual-controller harness once; both controllers run on the same tracker
    instance so the multi-device code paths (predictive-ROI per-device gate, blob labelling
    across devices, per-device pose attempts) all exercise. Produces out_dir/dev{1,2}.csv.
    Returns True iff both per-device CSVs are produced.

    Harness CLI: <frames> <hmd_cams.json> <telem_dir> <ctrl_left> <ctrl_right> <out_dir>
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cmd = [binary, frames, cams, telem, ctrl_left, ctrl_right, out_dir]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       env=replay_env(imu_cal_dir=imu_cal_dir_for_capture(capture)))
    if r.returncode != 0:
        sys.stderr.write(f"replay FAILED: {r.stderr.strip()[:400]}\n")
        return False
    # Surface the harness's diagnostic stderr lines even on rc 0: capture_output swallowed
    # WARN/ERROR lines (e.g. the completion-barrier timeout before it became fatal), making a
    # degraded run invisible to every consumer of this table.
    for line in (r.stderr or "").splitlines():
        if any(key in line for key in ("WARN", "ERROR", "FATAL")):
            sys.stderr.write(f"replay [{Path(out_dir).name}] {line}\n")
    ok = True
    for dev in (1, 2):
        csv_path = Path(out_dir) / f"dev{dev}.csv"
        if not csv_path.is_file():
            sys.stderr.write(f"replay missing CSV for dev{dev}: {csv_path}\n")
            ok = False
    return ok


def score_run(telem, dev, csv_path, col):
    """All metrics for one (csv, col) candidate: cleaned-GT metrics + independent anchor."""
    ref = build_reference(telem, dev)
    if ref is None:
        return None
    t_c, pos_c, quat_c = load_candidate_csv(csv_path, col)
    if t_c.shape[0] == 0:
        return None
    n_total = csv_row_count(csv_path)
    mt = compute_metrics(ref, t_c, pos_c, quat_c, n_total, match_ms=25.0, valid_only=True)
    if mt is None:
        return None
    anc = HA.score_candidate_against_anchor(telem, dev, t_c, quat_c)
    mt["anchor_tilt_flip_pct"] = anc["tilt_flip_rate_pct"] if anc else float("nan")
    mt["anchor_yaw_flip_pct"] = anc["yaw_flip_rate_pct"] if anc else float("nan")
    mt["anchor_judged"] = anc["n_judged"] if anc else 0
    return mt


COLS = [
    ("posRMSE", "pos_rmse_cm", "{:.1f}"),
    ("posMed", "pos_med_cm", "{:.2f}"),
    ("oriRMS", "ori_rms", "{:.1f}"),
    ("wrongBr%", "wrong_branch_pct", "{:.1f}"),
    ("flip%", "flip_rate_pct", "{:.2f}"),
    ("flyMax", "fly_max_jump_m", "{:.2f}"),
    ("fly>reach%", "fly_frac_beyond_reach", "{:.2%}"),
    ("jitRMS", "jitter_rms_cm", "{:.2f}"),
    ("yield%", "yield_pct", "{:.0f}"),
    ("tiltFlip%", "anchor_tilt_flip_pct", "{:.2f}"),
    ("YAWflip%", "anchor_yaw_flip_pct", "{:.2f}"),
]


def _fmt(v, fmt):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        if fmt.endswith("%}"):
            return fmt.format(v)
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


def print_table(rows):
    """rows: list of (label, metrics dict). Prints one aligned comparison table."""
    hdr = ["run"] + [c[0] for c in COLS]
    widths = [max(len(hdr[0]), max((len(r[0]) for r in rows), default=0))]
    cells = []
    for label, mt in rows:
        row = [label] + [_fmt(mt.get(key), fmt) for _, key, fmt in COLS]
        cells.append(row)
    for j in range(1, len(hdr)):
        widths.append(max(len(hdr[j]), max((len(c[j]) for c in cells), default=0)))
    line = "  ".join(h.rjust(w) if j else h.ljust(w) for j, (h, w) in enumerate(zip(hdr, widths)))
    print(line)
    print("  ".join("-" * w for w in widths))
    for c in cells:
        print("  ".join(v.rjust(w) if j else v.ljust(w) for j, (v, w) in enumerate(zip(c, widths))))


def main() -> int:
    ap = argparse.ArgumentParser(description="one-command A/B of the G2 controller tracker")
    ap.add_argument("--capture", action="append", required=True, help="capture dir (repeatable)")
    ap.add_argument("--bin", help="harness binary, single-binary mode (REQUIRED unless "
                                  "--baseline-bin/--candidate-bin are given; no default so the "
                                  "binary under test is always an explicit, pinnable choice)")
    ap.add_argument("--baseline-bin", help="A/B: baseline harness binary")
    ap.add_argument("--candidate-bin", help="A/B: candidate harness binary")
    ap.add_argument("--cams", default=None,
                    help="override the camera config (default: each capture's own provenance "
                         "snapshot, else the pinned pre-provenance config)")
    ap.add_argument("--ctrl-left")
    ap.add_argument("--ctrl-right")
    ap.add_argument("--cols", default="opt,pred")
    ap.add_argument("--out", default="/tmp/run_ab")
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    left, right = _find_controller_jsons(args.ctrl_left, args.ctrl_right)
    if not left or not right:
        sys.stderr.write("could not resolve controller jsons; pass --ctrl-left/--ctrl-right\n")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # binary set: single (one binary, opt-vs-pred) or A/B (two binaries).
    if args.baseline_bin and args.candidate_bin:
        binaries = [("base", args.baseline_bin), ("cand", args.candidate_bin)]
    elif args.bin:
        binaries = [("", args.bin)]
    else:
        ap.error("pass --bin, or both --baseline-bin and --candidate-bin")
    for _, binary in binaries:
        if not Path(binary).is_file():
            ap.error(f"harness binary not found: {binary} — build it with "
                     f"`ninja -C <worktree>/build-cmake offline_vio_replay`")

    rows = []
    for cap in args.capture:
        capture = Path(os.path.expanduser(cap))
        telem = capture / "telemetry"
        frames = str(frame_source(capture))
        cap_tag = capture.name
        cams = str(args.cams or cams_for_capture(capture))
        for btag, binary in binaries:
            # Dual-controller replay: harness exercises the multi-device tracker code paths
            # (predictive-ROI per-device gate, cross-device blob labelling) in one invocation.
            run_dir = str(out / f"{cap_tag}_{btag or 'cur'}")
            if not run_replay_dual(binary, frames, cams, str(telem), left, right, run_dir, capture):
                continue
            for dev in (1, 2):
                csv_path = str(Path(run_dir) / f"dev{dev}.csv")
                for col in cols:
                    mt = score_run(str(telem), dev, csv_path, col)
                    if mt is None:
                        continue
                    parts = [cap_tag, DEVICE_NAMES[dev], col]
                    if btag:
                        parts.insert(1, btag)
                    rows.append(("/".join(parts), mt))

    if not rows:
        sys.stderr.write("no runs scored\n")
        return 1
    print(f"\n=== G2 controller tracker A/B  ({len(rows)} runs) ===")
    print("cleaned-GT metrics (self-consistent) + INDEPENDENT head-pose anchor "
          "(tiltFlip/YAWflip, non-drifting)\n")
    print_table(rows)
    print("\nlegend: posRMSE/posMed cm | oriRMS deg | wrongBr%/flip% vs cleaned-GT | "
          "flyMax m, fly>reach% | jitRMS cm (self) | yield% | tiltFlip%/YAWflip% vs INDEPENDENT anchor")
    return 0


if __name__ == "__main__":
    sys.exit(main())
