#!/usr/bin/env python3
"""regression_check.py -- standing guardrail for the G2 controller tracker (C4).

Replays the faithful capture through the offline_vio_replay harness for BOTH controllers, scores
BOTH streams with the SAME metrics the A/B tooling uses (mse_eval cleaned-GT + the INDEPENDENT
head-pose anchor), and compares against a checked-in baseline (tests/regression_baseline.json). FAILS
if a guarded metric regresses beyond a small tolerance -- so a change that re-introduces mirror flips,
drops yield, or worsens the out-of-view re-entry accuracy / coast snap cannot land unnoticed.

Two streams are guarded. opt = the constellation front-end's own output, where a matching /
disambiguation regression shows up directly. pred = the ESKF FUSED prediction, the controller's actual
reported pose (out-of-view coast / freeze / re-entry snap) -- the user-facing stream. A metric that
worsens on EITHER stream fails the gate. Guarded metrics: yield% (must not DROP), pos median (cm),
wrong-branch%, flip%, fly-max-jump (m), tiltFlip%, YAWflip% (must not RISE), plus the fused re-entry
snap (m, pred only) and the re-entry GT accuracy floor (pred only) -- each within an absolute tolerance.

OUT-OF-VIEW ACCURACY (pred only): instead of fly>reach (distance-from-session-median, a freeze-biased
proxy the GT re-entry test proved does NOT track actual prediction accuracy), the gate uses a
re-entry GT floor on TRUE forced-drop windows when the replay CSV marks them with drop_optical. Natural
opt_valid gaps in a normal replay are not necessarily no-optical intervals because per-LED optical folds
may still feed fusion, so they stay covered by the regular trajectory metrics instead of this floor.
On scored forced-drop re-entries, body-follow prediction must be at least as accurate as a synthetic
freeze counterfactual (last-seen position) on the SAME events.

Gating: every skip path (missing scoring deps, capture dir, harness binary, controller jsons or
baseline) exits 77 with a clear SKIP message -- the ctest registration's SKIP_RETURN_CODE 77 turns it
into a LOUD ctest SKIP (mirroring coast_regression), never a silent pass.

Usage:
  regression_check.py [--capture DIR] [--bin PATH] [--baseline JSON]
                      [--cams JSON] [--ctrl-left JSON] [--ctrl-right JSON]
                      [--cols opt,pred] [--tol-pct N] [--yield-tol-pct N] [--update] [--out DIR]

Conda: PYTHONNOUSERSITE=1 ~/miniconda3/envs/g2vr/bin/python regression_check.py ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# The scoring stack needs numpy + the telemetry tools on sys.path. A bare interpreter (e.g. a build host
# without the analysis env) cannot run the guardrail; SKIP cleanly rather than fail the build.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from run_ab import (
        _find_controller_jsons,
        _frames_dir,
        run_replay_dual,
        score_run,
    )
    from replay_contract import cams_for_capture
    from mse_eval import reentry_accuracy_from_csv, reentry_snap_m_from_csv
    from manifest import DEVICE_NAMES
except ImportError as e:
    print(f"SKIP: scoring deps unavailable ({e}) -- regression guardrail not run")
    sys.exit(77)

# The faithful capture this guardrail is calibrated against (the in-headset capture used for tuning).
DEFAULT_CAPTURE = os.path.expanduser("~/g2-linux-research/captures/20260524-200416-headpose")
# The live work front. `src/monado-thaytan` is the dormant integration checkout, whose baseline still
# carries the pre-corrected-harness bars (dev1/opt flip-rate 9.209 % vs the live 0.0, wrong-branch
# 19.226 % vs 5.0, fly-max 1.093 m vs 0.139) — defaulting there made a direct run validate against
# bars two orders of magnitude too loose. CTest was unaffected because CMake passes --baseline
# explicitly; a hand invocation was not.
DEFAULT_WORKTREE = Path(__file__).resolve().parents[2] / "worktrees/g2-sota-stack"
DEFAULT_BIN = str(DEFAULT_WORKTREE / "build-cmake/tests/offline_vio_replay")
DEFAULT_BASELINE = str(DEFAULT_WORKTREE / "tests/regression_baseline.json")

# A reported metric. key = JSON/score key; label = table header; bad = the regressing direction ("up"
# => higher is worse, "down" => lower is worse); cols = which scored columns it applies to; tol =
# absolute tolerance in the metric's own unit (None => use the shared flip/yield tolerance below).
# floor_key (optional): when set, the gate compares cur to mt[floor_key] (a LIVE counterfactual computed
# from the SAME CSV) instead of to the stored JSON baseline -- used for the re-entry GT accuracy floor
# (BF must be no worse than the synthetic freeze counterfactual on the same re-entry events).
# informational (optional): printed in the table but NEVER gated -- for signals that are real-but-imperfect
# (e.g. fly>reach, GT-proven freeze-biased) where transparency matters more than failing on them.
#
# Two streams are reported. opt = the constellation front-end's own output, where a matching /
# disambiguation regression shows up directly. pred = the ESKF FUSED prediction, the controller's
# actual reported pose (out-of-view coast, freeze, re-entry snap) -- the user-facing stream. A GATED
# metric that worsens on EITHER column fails the gate.
class M:
    def __init__(self, key, label, bad, cols, tol=None, scale=1.0, floor_key=None, informational=False):
        self.key, self.label, self.bad, self.cols = key, label, bad, cols
        self.tol, self.scale, self.floor_key, self.informational = tol, scale, floor_key, informational


BOTH = ("opt", "pred")
GUARDED = [
    # quality of the matched trajectory (both streams)
    M("yield_pct", "yield%", "down", BOTH),
    M("pos_med_cm", "posMed_cm", "up", BOTH, tol=1.5),       # median pos err vs cleaned-GT may rise <=1.5cm
    M("wrong_branch_pct", "wrongBr%", "up", BOTH),
    M("flip_rate_pct", "flip%", "up", BOTH),
    # out-of-view runaway catcher (degenerate solves teleporting the report)
    M("fly_max_jump_m", "flyMax_m", "up", BOTH, tol=0.15),   # worst consecutive jump may rise <=15cm
    # out-of-view accuracy FLOOR vs freeze counterfactual (pred only; forced drop_optical re-entries only
    # on modern replay CSVs; natural opt_valid gaps may still include per-LED optical folds)
    M("reentry_bf_err_cm_median", "reentryErr", "up", ("pred",), tol=5.0,
      floor_key="reentry_freeze_err_cm_median"),
    # informational mean alongside the gated median -- the mean is dominated by a few large-displacement
    # outliers per the GT analysis; both views together prevent a one-statistic story.
    M("reentry_bf_err_cm_mean", "reentryErr_mean", "up", ("pred",),
      floor_key="reentry_freeze_err_cm_mean", informational=True),
    # the user-visible coast SNAP back into view -- only the fused stream has a tracked/coast notion
    M("reentry_snap_m", "snap_m", "up", ("pred",), tol=0.15),
    # informational: fly>reach (% beyond arm-reach of the trajectory's OWN session median). The GT
    # re-entry test proved this is freeze-biased -- a body-following controller's report naturally
    # excurses far from its in-view median when the head moves, even when GT-accuracy is preserved.
    # Kept for transparency: viewers can see the trade-off (BF raises distance-from-median) without
    # the gate firing on a metric that does not reflect real accuracy.
    M("fly_frac_beyond_reach", "fly>reach", "up", BOTH, scale=100.0, informational=True),
    # independent (non-drifting) head-pose anchor flips -- front-end disambiguation signal
    M("anchor_tilt_flip_pct", "tiltFlip%", "up", BOTH),
    M("anchor_yaw_flip_pct", "YAWflip%", "up", BOTH),
]


def _scaled(m, value):
    return None if value is None else value * m.scale


def _missing(value):
    """True when a metric value carries no measurement: absent, None, or non-finite.

    NaN comparisons are all False, so without this a vanished metric (e.g. the head-pose anchor
    stream silently absent -> anchor_*_flip_pct = NaN) sails through the gate as 'ok'."""
    if value is None:
        return True
    try:
        return not math.isfinite(value)
    except TypeError:
        return True


# Shared-tolerance percentage metrics ride on tiny baselines (pinned flip rates are
# 0.017-0.035%, anchor flips 0-0.02%): a flat 2.0-point absolute tolerance would let a ~100x
# flip-rate regression pass. Gate them at max(floor, 5x baseline), never LOOSER than the
# shared absolute tolerance (so percent-scale metrics like wrong-branch keep their old gate).
PCT_TOL_FLOOR = 0.1
PCT_TOL_REL = 5.0


def _tol_for(m, tol_pct, yield_tol_pct, ref=None):
    """The tolerance for one metric: its own absolute tol if set, else the shared flip/yield
    scale, rescaled to the baseline for the small-baseline percentage metrics."""
    if m.tol is not None:
        return m.tol
    if m.key == "yield_pct":
        return yield_tol_pct
    if ref is not None:
        return min(tol_pct, max(PCT_TOL_FLOOR, PCT_TOL_REL * ref))
    return tol_pct


def measure(capture, binary, cams, left, right, out_dir, columns):
    """Replay + score the given columns for both controllers. Returns {dev: {col: metrics}} or None.

    reentry_snap_m is computed from the CSV's pred_tracked transitions (not a matched-frame metric) and
    folded into each column's dict so the gate can read it uniformly."""
    telem = Path(capture) / "telemetry"
    frames = _frames_dir(Path(capture))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "regression_dual"
    if not run_replay_dual(binary, frames, cams, str(telem), left, right, str(run_dir), Path(capture)):
        return None

    result = {}
    for dev in (1, 2):
        csv_path = str(run_dir / f"dev{dev}.csv")
        snap = reentry_snap_m_from_csv(csv_path)
        racc = reentry_accuracy_from_csv(csv_path)  # pred-only floor; ungated on opt
        per_col = {}
        for col in columns:
            mt = score_run(str(telem), dev, csv_path, col)
            if mt is None:
                return None
            mt["reentry_snap_m"] = snap
            mt["reentry_bf_err_cm_median"] = racc["bf_err_cm_median"]
            mt["reentry_freeze_err_cm_median"] = racc["freeze_err_cm_median"]
            mt["reentry_bf_err_cm_mean"] = racc["bf_err_cm_mean"]
            mt["reentry_freeze_err_cm_mean"] = racc["freeze_err_cm_mean"]
            mt["reentry_n_events"] = racc["n_events"]
            per_col[col] = mt
        result[dev] = per_col
    return result


def to_baseline(measured):
    """Reduce full metrics dicts to the small per-column guarded-metric baseline written to JSON.
    Informational metrics and floor-keyed metrics are excluded -- the former are not gated, the latter
    use a live counterfactual, neither needs a stored baseline."""
    return {
        str(dev): {
            col: {m.key: _scaled(m, mt[m.key]) for m in GUARDED
                  if col in m.cols and not m.informational and m.floor_key is None}
            for col, mt in per_col.items()
        }
        for dev, per_col in measured.items()
    }


def check(measured, baseline, tol_pct, yield_tol_pct):
    """Compare measured vs baseline across columns. Returns (ok, rows) for the printed table."""
    ok = True
    rows = []
    for dev in sorted(measured):
        base_dev = baseline.get(str(dev), {})
        for col in sorted(measured[dev]):
            mt = measured[dev][col]
            base = base_dev.get(col, {})
            for m in GUARDED:
                if col not in m.cols:
                    continue
                cur_raw = mt.get(m.key)
                cur = None if _missing(cur_raw) else _scaled(m, cur_raw)
                # A floor_key metric is gated against a live counterfactual computed from the SAME CSV
                # (apples-to-apples), not against the stored JSON baseline.
                if m.floor_key is not None:
                    ref_raw = mt.get(m.floor_key)
                    ref = None if _missing(ref_raw) else _scaled(m, ref_raw)
                else:
                    ref_raw = base.get(m.key)
                    ref = None if _missing(ref_raw) else ref_raw
                if m.informational:
                    # Printed for transparency, never gated; n/a renders as n/a.
                    delta = cur - ref if (cur is not None and ref is not None) else float("nan")
                    rows.append((DEVICE_NAMES[dev], col, m.label, cur, ref, delta, "info"))
                    continue
                if cur is None and ref is None:
                    # The failure-mode class is absent from BOTH sides of this run (e.g. a capture
                    # with zero forced-drop re-entry events): render n/a loudly, nothing to gate.
                    rows.append((DEVICE_NAMES[dev], col, m.label, cur, ref, float("nan"), "n/a"))
                    continue
                if cur is None:
                    # The current run failed to produce a measurement the reference has: that is a
                    # measurement regression, never an 'ok'. FAIL-safe.
                    ok = False
                    rows.append((DEVICE_NAMES[dev], col, m.label, cur, ref, float("nan"), "FAIL(n/a)"))
                    continue
                if ref is None:
                    # Stored baseline predates this metric: visible, not gated (the committed
                    # baseline makes this state loud in review).
                    rows.append((DEVICE_NAMES[dev], col, m.label, cur, ref, float("nan"), "no-base"))
                    continue
                tol = _tol_for(m, tol_pct, yield_tol_pct, ref=ref)
                delta = cur - ref
                regressed = (delta > tol) if m.bad == "up" else (delta < -tol)
                if regressed:
                    ok = False
                rows.append((DEVICE_NAMES[dev], col, m.label, cur, ref, delta,
                             "FAIL" if regressed else "ok"))
    return ok, rows


def print_table(rows, tol_pct, yield_tol_pct):
    hdr = ("device", "col", "metric", "current", "baseline", "delta", "status")
    print(f"{hdr[0]:<7} {hdr[1]:<5} {hdr[2]:<10} {hdr[3]:>9} {hdr[4]:>9} {hdr[5]:>9}  {hdr[6]}")
    print("-" * 62)
    for dev, col, label, cur, ref, delta, status in rows:
        cs = "n/a" if (cur is None or cur != cur) else f"{cur:9.2f}"
        rs = "n/a" if (ref is None or ref != ref) else f"{ref:9.2f}"
        ds = "n/a" if delta != delta else f"{delta:+9.2f}"
        mark = " <== REGRESSION" if status.startswith("FAIL") else ""
        print(f"{dev:<7} {col:<5} {label:<10} {cs:>9} {rs:>9} {ds:>9}  {status}{mark}")
    print(f"\ntolerance: flip/branch/anchor min({tol_pct:.2f}, max({PCT_TOL_FLOOR}, {PCT_TOL_REL:.0f}x baseline)) pts, "
          f"yield drop > {yield_tol_pct:.2f}%; "
          f"posMed +1.5cm, flyMax/snap +0.15m; forced-drop reentryErr +5cm vs freeze counterfactual")


def main() -> int:
    ap = argparse.ArgumentParser(description="standing regression guardrail for the G2 controller tracker")
    ap.add_argument("--capture", default=DEFAULT_CAPTURE)
    ap.add_argument("--bin", default=DEFAULT_BIN)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE)
    ap.add_argument("--cams", default=None,
                    help="override the camera config (default: the capture's own provenance "
                         "snapshot, else the pinned pre-provenance config)")
    ap.add_argument("--ctrl-left")
    ap.add_argument("--ctrl-right")
    ap.add_argument("--tol-pct", type=float, default=2.0, help="abs %% a flip/branch/anchor metric may rise")
    ap.add_argument("--yield-tol-pct", type=float, default=3.0, help="abs %% yield may drop")
    ap.add_argument("--cols", default="opt,pred", help="scored streams to gate (default both)")
    ap.add_argument("--update", action="store_true", help="(re)write the baseline from the current tree")
    ap.add_argument("--out", default="/tmp/regression_check")
    args = ap.parse_args()
    columns = [c.strip() for c in args.cols.split(",") if c.strip()]

    # Gating: exit 77 (ctest SKIP_RETURN_CODE) so a capture-less / harness-less build reports a
    # visible SKIP, never a fake green.
    if not Path(args.capture).is_dir():
        print(f"SKIP: capture not present ({args.capture}) -- regression guardrail not run")
        return 77
    if not Path(args.bin).is_file():
        print(f"SKIP: harness binary not present ({args.bin}) -- build offline_vio_replay first")
        return 77

    left, right = _find_controller_jsons(args.ctrl_left, args.ctrl_right)
    if not left or not right:
        print(f"SKIP: could not resolve controller jsons (pass --ctrl-left/--ctrl-right)")
        return 77

    cams = args.cams or str(cams_for_capture(args.capture))
    measured = measure(args.capture, args.bin, cams, left, right, args.out, columns)
    if measured is None:
        sys.stderr.write("ERROR: replay/scoring failed; cannot evaluate the guardrail\n")
        return 2

    if args.update:
        payload = {
            "capture": Path(args.capture).name,
            "columns": columns,
            "_note": "edited & maintained by Claude, presented as-is. Regenerate: regression_check.py --update",
            "metrics": to_baseline(measured),
        }
        Path(args.baseline).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote baseline -> {args.baseline}")
        return 0

    if not Path(args.baseline).is_file():
        print(f"SKIP: no baseline at {args.baseline} -- create it with --update")
        return 77
    baseline = json.loads(Path(args.baseline).read_text()).get("metrics", {})

    ok, rows = check(measured, baseline, args.tol_pct, args.yield_tol_pct)
    print(f"=== G2 controller tracker regression check (capture {Path(args.capture).name}, "
          f"cols={'+'.join(columns)}) ===\n")
    print_table(rows, args.tol_pct, args.yield_tol_pct)
    if not ok:
        print("\nRESULT: FAIL -- a guarded metric regressed beyond tolerance")
        return 1
    print("\nRESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
