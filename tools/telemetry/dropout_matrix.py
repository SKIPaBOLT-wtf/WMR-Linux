#!/usr/bin/env python3
"""Run controlled optical-dropout replays and summarize OOV/fusion quality."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_contract import (  # noqa: E402
    cams_ctrl_gain,
    cams_for_capture,
    imu_cal_dir_for_capture,
    replay_env,
)

ROOT = Path("/home/mrwhite0racle/g2-linux-research")
DEFAULT_LEFT = Path("/home/mrwhite0racle/.config/monado/wmr/controller_A85K1111630014L.json")
DEFAULT_RIGHT = Path("/home/mrwhite0racle/.config/monado/wmr/controller_A85K5091930012R.json")
DEFAULT_BASELINE = Path(__file__).resolve().parent / "data/dropout-matrix-baseline.json"


CAPTURES = {
    "xv1": {
        "reference": ROOT / "captures/20260528-080421-xv-session1",
        "frames": ROOT / "captures/20260528-080421-xv-session1-framebin/frames",
        "telemetry": ROOT / "captures/20260528-080421-xv-session1/telemetry",
    },
    "clean2": {
        "reference": ROOT / "captures/20260526-175615-clean-session2",
        "frames": ROOT / "captures/20260526-175615-clean-session2/frames",
        "telemetry": ROOT / "captures/20260526-175615-clean-session2/telemetry",
    },
}


REGIMES = {
    "normal": {},
    "periodic-100": {
        "G2_REPLAY_DROP_OPTICAL_PERIOD_MS": "1000",
        "G2_REPLAY_DROP_OPTICAL_DURATION_MS": "100",
    },
    "periodic-300": {
        "G2_REPLAY_DROP_OPTICAL_PERIOD_MS": "1000",
        "G2_REPLAY_DROP_OPTICAL_DURATION_MS": "300",
    },
    "after-2000": {
        "G2_REPLAY_DROP_OPTICAL_AFTER_MS": "2000",
    },
    "random-p05": {
        "G2_REPLAY_DROP_OPTICAL_RANDOM_P": "0.05",
        "G2_REPLAY_DROP_OPTICAL_RANDOM_SEED": "475205",
    },
    "bursts-short": {
        "G2_REPLAY_DROP_OPTICAL_BURST_COUNT": "10",
        "G2_REPLAY_DROP_OPTICAL_BURST_MIN_MS": "100",
        "G2_REPLAY_DROP_OPTICAL_BURST_MAX_MS": "250",
        "G2_REPLAY_DROP_OPTICAL_BURST_SEED": "475210",
    },
    "bursts-long": {
        "G2_REPLAY_DROP_OPTICAL_BURST_COUNT": "8",
        "G2_REPLAY_DROP_OPTICAL_BURST_MIN_MS": "300",
        "G2_REPLAY_DROP_OPTICAL_BURST_MAX_MS": "700",
        "G2_REPLAY_DROP_OPTICAL_BURST_SEED": "475208",
    },
    # Realistic-OOV: black out ONE device's LEDs inside the windows (frames/clutter/other
    # controller keep flowing). Needs <capture>/led_mask_dev<N>.csv from make_led_mask.py;
    # frames without mask coverage fall back to blackout inside the window.
    "mask1-periodic-300": {
        "G2_REPLAY_DROP_OPTICAL_PERIOD_MS": "1000",
        "G2_REPLAY_DROP_OPTICAL_DURATION_MS": "300",
        "G2_REPLAY_DROP_DEVICE": "1",
    },
    "mask2-periodic-300": {
        "G2_REPLAY_DROP_OPTICAL_PERIOD_MS": "1000",
        "G2_REPLAY_DROP_OPTICAL_DURATION_MS": "300",
        "G2_REPLAY_DROP_DEVICE": "2",
    },
    "mask1-bursts-long": {
        "G2_REPLAY_DROP_OPTICAL_BURST_COUNT": "8",
        "G2_REPLAY_DROP_OPTICAL_BURST_MIN_MS": "300",
        "G2_REPLAY_DROP_OPTICAL_BURST_MAX_MS": "700",
        "G2_REPLAY_DROP_OPTICAL_BURST_SEED": "475208",
        "G2_REPLAY_DROP_DEVICE": "1",
    },
    "mask2-bursts-long": {
        "G2_REPLAY_DROP_OPTICAL_BURST_COUNT": "8",
        "G2_REPLAY_DROP_OPTICAL_BURST_MIN_MS": "300",
        "G2_REPLAY_DROP_OPTICAL_BURST_MAX_MS": "700",
        "G2_REPLAY_DROP_OPTICAL_BURST_SEED": "475208",
        "G2_REPLAY_DROP_DEVICE": "2",
    },
}


def capture_cams(args: argparse.Namespace, capture_name: str) -> Path:
    """The camera config for one cell: an explicit --cams wins, else the capture's own."""
    return args.cams or cams_for_capture(CAPTURES[capture_name]["reference"])


def regime_drop_target(regime_name: str) -> int:
    return int(REGIMES.get(regime_name, {}).get("G2_REPLAY_DROP_DEVICE", 0) or 0)


#: Hard-fail reasons that mean the MEASUREMENT itself was invalid (as opposed to a
#: genuine quality failure like an identity swap). Any such row makes the process
#: exit nonzero so a benchmark can never silently publish non-compliant numbers.
COMPLIANCE_FAIL_PREFIXES = ("drop_noncompliance", "schema_incomplete", "identity_gate_disabled")

#: Row keys the objective depends on. None/missing here means an old harness or old
#: scorer schema produced the row — that is a measurement failure, never a free pass.
REQUIRED_ROW_KEYS = (
    "total_frame_groups",
    "scoreable_frames",
    "scoreable_pct_total",
    "all_scoreable_position_tracked_correct_pct",
    "all_scoreable_reported_position_correct_pct",
    "all_scoreable_position_tracked_accuracy_pct",
    "pred_recall",
    "pred_position_recall",
    "pred_position_precision",
    "pred_position_rmse_cm",
    "pred_position_p95_cm",
    "all_pos_p95_cm",
    "wrong_branch_pct_scored",
    # Shrinkage/evidence denominators: shrunk_pct leaves a raw value untouched and
    # evidence_weight ramps a term to zero when its count is missing, so a schema that
    # silently drops these counts would silently disengage shrinkage (and conditional
    # terms) while still printing green. Police them like the metrics they weight.
    "all_scoreable_frames",
    "all_scoreable_position_tracked",
    "pred_scored_frames",
    "pred_tp",
    "pred_fn",
    "pred_position_tp",
    "pred_position_fn",
    "pred_position_fp",
    "forced_drop_visible_frames",
    "forced_drop_visible_reported",
    "forced_drop_visible_position_tracked",
    "stale_visible_frames",
    "stale_position_tracked",
)
REQUIRED_IDENTITY_KEYS = (
    "identity_other_report_explains_ref",
    "identity_strict_two_way_swap_frames",
)


def regime_expects_drops(regime_name: str) -> bool:
    return bool(REGIMES.get(regime_name))


def verify_drop_compliance(run_dir: Path | str, regime_name: str) -> dict[str, Any]:
    """Verify from the replay's own output that the forced-drop schedule engaged.

    The schedule is requested via env, but a harness that predates the machinery
    silently replays normal — and used to score ~9x better for it. The replay CSV's
    drop_optical column is the authoritative record, so compliance is checked there,
    not in what the scorer was told.
    """
    out_dir = Path(run_dir) / "out"
    expects = regime_expects_drops(regime_name)
    info: dict[str, Any] = {
        "expects_drops": expects,
        "column_present": False,
        "csv_rows": 0,
        "csv_dropped": 0,
        "status": "unverifiable",
    }
    for csv_path in sorted(out_dir.glob("dev*.csv")):
        try:
            with csv_path.open() as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "drop_optical" not in reader.fieldnames:
                    continue
                info["column_present"] = True
                for rec in reader:
                    info["csv_rows"] += 1
                    if rec.get("drop_optical") not in (None, "", "0"):
                        info["csv_dropped"] += 1
        except OSError:
            continue
    if not info["column_present"]:
        info["status"] = "missing_column" if expects else "ok_no_column"
    elif expects and info["csv_dropped"] == 0:
        info["status"] = "no_drops_engaged"
    elif not expects and info["csv_dropped"] > 0:
        info["status"] = "unexpected_drops"
    else:
        info["status"] = "ok"
    return info


def metric(result: dict[str, Any], stream: str, key: str) -> Any:
    return result.get("scores", {}).get(stream, {}).get(key)


def stat(result: dict[str, Any], group: str, field: str, stat_key: str) -> Any:
    block = result.get("oov", {}).get(group, {}).get(field, {})
    return block.get(stat_key) if isinstance(block, dict) else None


def oov_metric(result: dict[str, Any], group: str, key: str) -> Any:
    return result.get("oov", {}).get(group, {}).get(key)


def ref_metric(result: dict[str, Any], key: str) -> Any:
    return result.get("reference", {}).get(key)


def score_metric(result: dict[str, Any], stream: str, key: str) -> Any:
    return result.get("scores", {}).get(stream, {}).get(key)


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def pct_fraction(value: Any) -> float | None:
    out = as_float(value)
    return out * 100.0 if out is not None else None


#: Sample count at which a conditional (stale/forced-drop) objective term carries full
#: weight. Below it the term's weight ramps smoothly to zero instead of the old hard
#: `n >= 10` block gate, which flipped whole term blocks in/out on 8-vs-10 stale frames
#: and let 2-frame numerators score 100%-or-excluded at full weight.
EVIDENCE_FULL_N = 10.0


def evidence_weight(count: Any) -> float:
    """C1-smooth ramp 0 -> 1 over [0, EVIDENCE_FULL_N] (smoothstep).

    Exactly 0.0 at n=0 (no evidence: the term is undefined and excluded) and exactly
    1.0 at n >= EVIDENCE_FULL_N, so well-evidenced rows score identically to the
    pre-ramp scorer; in between, evidence weight grows continuously with sample count
    so no single frame can flip a block in or out.
    """
    n = as_float(count) or 0.0
    t = min(max(n, 0.0), EVIDENCE_FULL_N) / EVIDENCE_FULL_N
    return t * t * (3.0 - 2.0 * t)


#: Denominator count at which a percentage term's raw value carries no small-sample
#: granularity worth damping. A raw percentage on n frames moves in 100/n-point steps
#: from a SINGLE frame (n=12: 100 -> 91.67), while its own binomial standard error is
#: 100*sqrt(q(1-q)/n); quantization dominates sampling noise for n < 1/(q(1-q)), which
#: at the scorer's strictest target (q=0.98) is 51.02. Below ESTIMATE_FULL_N the
#: estimate is therefore shrunk toward its loss-neutral prior; at or above it the raw
#: value is used EXACTLY, so well-evidenced cells are bit-identical to the pre-shrink
#: scorer. One constant, no per-term tuning.
ESTIMATE_FULL_N = 51.0


def shrunk_pct(value: Any, count: Any, neutral: float) -> float | None:
    """Uncertainty-aware small-sample percentage estimate.

    Posterior mean under an evidence-budget prior: for n < ESTIMATE_FULL_N the missing
    (N - n) frames are scored at the term's loss-neutral rate, i.e. the Beta-posterior
    mean with prior strength max(0, N - n) centered on `neutral`:

        p_hat = neutral + (n / N) * (p - neutral)          (= (k + (N - n) * q0) / N)

    Single-frame sensitivity is thereby bounded at 100/N points for every n (the raw
    estimate steps 100/n, unbounded as n shrinks), the shrink weight decays linearly as
    evidence accumulates (a posterior mean is linear in counts; the smoothstep above is
    the WEIGHT gate, not an estimator), and the shrink is exactly zero at n >= N —
    unlike constant-strength Laplace/Wilson forms, which would perturb every large-n
    cell and silently re-calibrate the recorded history. Values at or above a shortfall
    target (or at an excess limit) stay loss-free at any n, since the shrink pulls
    toward the neutral point, never past it. A missing denominator leaves the raw value
    untouched (schema gaps are policed by the hard-fail keys, never a free pass here).
    """
    pct = as_float(value)
    if pct is None:
        return None
    n = as_float(count)
    if n is None or n >= ESTIMATE_FULL_N:
        return pct
    return neutral + (max(n, 0.0) / ESTIMATE_FULL_N) * (pct - neutral)


def count_sum(*values: Any) -> float | None:
    parts = [as_float(value) for value in values]
    if any(part is None for part in parts):
        return None
    return sum(parts)


def shortfall_loss(value: float | None, target: float, scale: float, weight: float) -> float:
    if value is None:
        return weight * math.log1p(16.0)
    miss = max(0.0, target - value) / scale
    return weight * math.log1p(miss * miss)


def excess_loss(value: float | None, limit: float, scale: float, weight: float) -> float:
    if value is None:
        return 0.0
    excess = max(0.0, value - limit) / scale
    return weight * math.log1p(excess * excess)


def shortfall_pct_loss(value: Any, count: Any, target: float, scale: float, weight: float) -> float:
    """shortfall_loss on a percentage whose denominator is `count`, small-n shrunk.

    The shrink prior is the loss-neutral point itself (the target), so the target is
    named once and the two cannot drift.
    """
    return shortfall_loss(shrunk_pct(value, count, target), target, scale, weight)


def excess_pct_loss(value: Any, count: Any, limit: float, scale: float, weight: float) -> float:
    """excess_loss on a percentage whose denominator is `count`, small-n shrunk."""
    return excess_loss(shrunk_pct(value, count, limit), limit, scale, weight)


def objective_for_row(row: dict[str, Any]) -> dict[str, Any]:
    """Nonlinear row objective.

    This is a ranking score, not a replacement for the detailed metrics. It uses
    hard gates for identity failures and an exponential loss so severe OOV
    failures dominate small normal-path wins instead of being averaged away.
    Conditional stale/forced-drop terms carry per-term continuous evidence
    weights (evidence_weight of each term's own denominator count) instead of a
    hard n>=10 block gate, and every percentage term is scored through the
    small-sample shrinkage estimator (shrunk_pct of its own denominator) so one
    frame in a small denominator cannot step a cell by whole points.
    """
    hard_fail: list[str] = []
    if (as_float(row.get("identity_strict_two_way_swap_frames")) or 0.0) > 0.0:
        hard_fail.append("strict_two_way_swap")
    if (as_float(row.get("identity_other_report_explains_ref")) or 0.0) > 0.0:
        hard_fail.append("cross_device_identity")

    forced_frames = as_float(row.get("forced_drop_visible_frames")) or 0.0
    stale_frames = as_float(row.get("stale_visible_frames")) or 0.0

    # Measurement-compliance hard fails: a row whose drops never engaged, whose schema
    # predates the required metrics, or whose identity gate was silently disabled is
    # not a quality datapoint at all.
    regime = str(row.get("regime") or "")
    drop_status = str(row.get("drop_compliance") or "unverifiable")
    target = regime_drop_target(regime)
    device = int(as_float(row.get("device")) or 0)
    expects = regime_expects_drops(regime) and (target == 0 or device == target)
    if expects and drop_status != "ok":
        hard_fail.append(f"drop_noncompliance:{drop_status}")
    elif expects and forced_frames <= 0.0:
        hard_fail.append("schema_incomplete:forced_drop_visible_frames")
    missing = [key for key in REQUIRED_ROW_KEYS if as_float(row.get(key)) is None]
    if missing:
        hard_fail.append("schema_incomplete:" + "+".join(missing))
    if any(as_float(row.get(key)) is None for key in REQUIRED_IDENTITY_KEYS):
        hard_fail.append("identity_gate_disabled")

    loss = 0.0
    weight_sum = 0.0
    components: dict[str, float] = {}
    weights: dict[str, float] = {}

    def add(name: str, value: float, weight: float) -> None:
        nonlocal loss, weight_sum
        components[name] = value
        weights[name] = weight
        loss += value
        weight_sum += weight

    # Position dominates the objective. Targets are in percentages or cm. Every
    # percentage term is shrunk against ITS OWN denominator (shrunk_pct).
    all_scoreable_frames = row.get("all_scoreable_frames")
    all_scoreable_position_tracked = row.get("all_scoreable_position_tracked")
    add(
        "position_tracked_correct_yield",
        shortfall_pct_loss(row.get("all_scoreable_position_tracked_correct_pct"),
                           all_scoreable_frames, 95.0, 5.0, 8.0),
        8.0,
    )
    add(
        "position_reported_correct_yield",
        shortfall_pct_loss(row.get("all_scoreable_reported_position_correct_pct"),
                           all_scoreable_frames, 95.0, 5.0, 3.0),
        3.0,
    )
    add(
        "visible_position_recall",
        shortfall_pct_loss(pct_fraction(row.get("pred_position_recall")),
                           count_sum(row.get("pred_position_tp"), row.get("pred_position_fn")),
                           95.0, 3.0, 4.0),
        4.0,
    )
    add("position_rmse", excess_loss(row.get("pred_position_rmse_cm"), 1.0, 1.5, 3.0), 3.0)
    add("position_p95", excess_loss(row.get("pred_position_p95_cm"), 3.0, 2.0, 4.0), 4.0)
    add("all_position_tail", excess_loss(row.get("all_pos_p95_cm"), 8.0, 6.0, 2.0), 2.0)

    # Accuracy of what we CLAIM: a tracked/accepted pose must be a correct pose.
    # Without these terms, inflating yield with mediocre poses GAINS objective points
    # (measured: +3 points while tracked-accuracy craters 91% -> 73%).
    add(
        "position_tracked_accuracy",
        shortfall_pct_loss(row.get("all_scoreable_position_tracked_accuracy_pct"),
                           all_scoreable_position_tracked, 98.0, 3.0, 6.0),
        6.0,
    )
    add(
        "position_precision",
        shortfall_pct_loss(pct_fraction(row.get("pred_position_precision")),
                           count_sum(row.get("pred_position_tp"), row.get("pred_position_fp")),
                           97.0, 2.0, 4.0),
        4.0,
    )

    # Orientation matters, but after position.
    add(
        "full_pose_true_recall",
        shortfall_pct_loss(pct_fraction(row.get("pred_recall")),
                           count_sum(row.get("pred_tp"), row.get("pred_fn")),
                           95.0, 5.0, 1.5),
        1.5,
    )
    add(
        "wrong_branch",
        excess_pct_loss(row.get("wrong_branch_pct_scored"),
                        row.get("pred_scored_frames"), 0.0, 0.25, 8.0),
        8.0,
    )

    # OOV/dead-reckon terms are only active when the regime actually creates those
    # denominators; each term's weight ramps with ITS OWN denominator count so scarce
    # evidence (2-frame accuracies, single-digit tail p95s) is damped, never cliffed.
    def add_evidenced(name: str, count: Any, weight: float, loss_fn: Any, value: Any, *loss_args: Any) -> None:
        w = weight * evidence_weight(count)
        if w > 0.0:
            add(name, loss_fn(value, *loss_args, w), w)

    forced_reported = row.get("forced_drop_visible_reported")
    forced_position_tracked = row.get("forced_drop_visible_position_tracked")
    add_evidenced("forced_drop_true_position_yield", forced_frames, 8.0, shortfall_pct_loss,
                  row.get("forced_drop_visible_position_tracked_correct_pct"), forced_frames, 95.0, 5.0)
    add_evidenced("forced_drop_reported_position_yield", forced_frames, 3.0, shortfall_pct_loss,
                  row.get("forced_drop_visible_reported_position_correct_pct"), forced_frames, 95.0, 5.0)
    add_evidenced("forced_drop_tracked_tail", forced_position_tracked, 4.0, excess_loss,
                  row.get("forced_drop_visible_position_tracked_p95_cm"), 5.0, 3.0)
    add_evidenced("forced_drop_all_tail", forced_reported, 2.0, excess_loss,
                  row.get("forced_drop_visible_p95_cm"), 10.0, 5.0)
    add_evidenced("forced_drop_position_tracked_accuracy", forced_position_tracked, 4.0, shortfall_pct_loss,
                  row.get("forced_drop_visible_position_tracked_accuracy_pct"), forced_position_tracked, 98.0, 3.0)

    stale_position_tracked = row.get("stale_position_tracked")
    add_evidenced("stale_true_position_yield", stale_frames, 5.0, shortfall_pct_loss,
                  row.get("stale_position_tracked_correct_pct"), stale_frames, 95.0, 5.0)
    add_evidenced("stale_reported_position_yield", stale_frames, 2.0, shortfall_pct_loss,
                  row.get("stale_reported_position_correct_pct"), stale_frames, 95.0, 5.0)
    add_evidenced("stale_tracked_tail", stale_position_tracked, 3.0, excess_loss,
                  row.get("stale_position_tracked_pos_p95_cm"), 5.0, 3.0)
    add_evidenced("stale_position_tracked_accuracy", stale_position_tracked, 3.0, shortfall_pct_loss,
                  row.get("stale_position_tracked_accuracy_pct"), stale_position_tracked, 98.0, 3.0)

    score = 0.0 if hard_fail else 100.0 * math.exp(-loss / max(weight_sum, 1.0))
    return {
        "objective_score": score,
        "objective_loss": loss,
        "objective_weight": weight_sum,
        "objective_hard_fail": ",".join(hard_fail),
        "objective_components": components,
        "objective_weights": weights,
    }


def run_checked(cmd: list[str], env: dict[str, str], cwd: Path, stdout: Path, stderr: Path) -> None:
    with stdout.open("w") as out, stderr.open("w") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}; stderr={stderr}")


def run_one(args: argparse.Namespace, capture_name: str, regime_name: str, runroot: Path) -> list[dict[str, Any]]:
    cap = CAPTURES[capture_name]
    regime = REGIMES[regime_name]
    out = runroot / capture_name / regime_name
    replay_out = out / "out"
    out.mkdir(parents=True, exist_ok=True)
    replay_out.mkdir(parents=True, exist_ok=True)

    settings: dict[str, str] = {**regime, "G2_REPLAY_TELEMETRY": str(out / "telemetry")}
    target = regime_drop_target(regime_name)
    if target:
        mask = cap["reference"] / f"led_mask_dev{target}.csv"
        if not mask.exists():
            raise RuntimeError(f"{capture_name}/{regime_name}: missing {mask} — "
                               f"generate it with tools/telemetry/make_led_mask.py")
        settings["G2_REPLAY_LED_MASK_FILE"] = str(mask)
    env = replay_env(settings, imu_cal_dir=imu_cal_dir_for_capture(cap["reference"]))

    replay_cmd = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(out / "time.txt"),
        str(args.binary),
        str(cap["frames"]),
        str(capture_cams(args, capture_name)),
        str(cap["telemetry"]),
        str(args.ctrl_left),
        str(args.ctrl_right),
        str(replay_out),
    ]
    run_checked(replay_cmd, env, out, out / "replay.stdout", out / "replay.stderr")

    compliance = verify_drop_compliance(out, regime_name)
    if regime_expects_drops(regime_name) and compliance["status"] != "ok":
        raise RuntimeError(
            f"{capture_name}/{regime_name}: forced drops did not engage "
            f"(status={compliance['status']}, csv_dropped={compliance['csv_dropped']}); "
            f"the harness lacks the drop machinery or the env never reached it — refusing to continue")

    metrics_json = out / "tracking_metrics.json"
    metrics_cmd = [
        sys.executable,
        str(ROOT / "tools/telemetry/tracking_metrics.py"),
        str(cap["reference"]),
        str(replay_out),
        "--candidate-name",
        f"{capture_name}-{regime_name}",
        "--reference-qc",
        args.reference_qc,
        "--out",
        str(metrics_json),
    ]
    run_checked(metrics_cmd, os.environ.copy(), ROOT, out / "tracking_metrics.stdout", out / "tracking_metrics.stderr")
    data = json.loads(metrics_json.read_text())
    for result in data:
        result["_capture"] = capture_name
        result["_regime"] = regime_name
        result["_run_dir"] = str(out)
    return data


def flatten(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        stale = result.get("oov", {}).get("detect_visible_but_stale", {})
        all_scoreable = result.get("oov", {}).get("all_scoreable", {})
        unknown_blob_rich = result.get("oov", {}).get("unknown_blob_rich", {})
        unknown_adjudication = result.get("unknown_adjudication", {}) or {}
        unknown_classes = unknown_adjudication.get("by_class", {}) or {}
        bng_reasons = unknown_adjudication.get("bng_reason_counts", {}) or {}
        identity = result.get("cross_device_identity", {}) or {}
        paired_identity = identity.get("paired", {}) or {}
        row = {
            "capture": result["_capture"],
            "regime": result["_regime"],
            "reference_qc": result.get("_reference_qc"),
            "device": result["device"],
            "total_frame_groups": ref_metric(result, "total_frame_groups"),
            "scoreable_frames": ref_metric(result, "scoreable_frames"),
            "scoreable_pct_total": ref_metric(result, "scoreable_pct_total"),
            "unknown_frames": ref_metric(result, "unknown_frames"),
            "unknown_pct_total": ref_metric(result, "unknown_pct_total"),
            "unknown_blob_ge4": ref_metric(result, "unknown_blob_total_ge4"),
            "unknown_blob_ge4_pct_total": ref_metric(result, "unknown_blob_ge4_pct_total"),
            "raw_observable_blob_ge4_scoreable_pct": ref_metric(result, "raw_observable_blob_ge4_scoreable_pct"),
            "unknown_reference_gap_frames": unknown_adjudication.get("explained_or_reference_gap"),
            "unknown_tracker_unresolved_blob_rich": unknown_adjudication.get("tracker_unresolved_blob_rich"),
            "unknown_tracker_unresolved_blob_rich_pct": unknown_adjudication.get(
                "tracker_unresolved_blob_rich_pct_blob_rich_unknown"),
            "unknown_unobservable_zero_blob": unknown_classes.get("unobservable_zero_blob"),
            "unknown_sparse_raw_blobs_1_3": unknown_classes.get("sparse_raw_blobs_1_3"),
            "unknown_ref_gap_tracker_pose": unknown_classes.get("reference_gap_tracker_pose"),
            "unknown_ref_gap_tracker_position": unknown_classes.get("reference_gap_tracker_position"),
            "unknown_search_success_no_commit": unknown_classes.get("reference_gap_search_success_no_commit"),
            "unknown_search_best_not_good": unknown_classes.get("blob_rich_search_best_not_good"),
            "unknown_bng_matched_lt3": bng_reasons.get("matched_lt3"),
            "unknown_bng_prior_position_fail": bng_reasons.get("prior_position_fail"),
            "unknown_bng_prior_orient_fail": bng_reasons.get("prior_orient_fail"),
            "unknown_bng_led_ids_fail": bng_reasons.get("led_ids_fail"),
            "unknown_bng_reproj_fail": bng_reasons.get("reproj_fail"),
            "unknown_bng_clean_cluster_fail": bng_reasons.get("clean_cluster_fail"),
            "unknown_bng_visible_cover_fail": bng_reasons.get("visible_cover_fail"),
            "unknown_bng_minimal_prior_fail": bng_reasons.get("minimal_prior_fail"),
            "unknown_bng_priorless_large_fail": bng_reasons.get("priorless_large_fail"),
            "unknown_bng_reason_unset": bng_reasons.get("reason_unset"),
            "unknown_no_success_other": unknown_classes.get("blob_rich_no_success_other"),
            "unknown_forced_drop_no_search": unknown_classes.get("blob_rich_forced_drop_no_search"),
            "unknown_no_search": unknown_classes.get("blob_rich_no_search"),
            "identity_other_report_explains_ref": identity.get("other_report_explains_ref"),
            "identity_other_report_explains_ref_pct_visible": identity.get("other_report_explains_ref_pct_visible"),
            "identity_other_report_explains_ref_pct_own_reports": identity.get(
                "other_report_explains_ref_pct_own_reports"),
            "identity_strict_two_way_swap_frames": paired_identity.get("strict_two_way_swap_frames"),
            "identity_strict_two_way_swap_pct_both_visible": paired_identity.get("strict_two_way_swap_pct_both_visible"),
            "identity_strict_two_way_swap_pct_both_reports": paired_identity.get("strict_two_way_swap_pct_both_reports"),
            "opt_accepted_blob_rich_unknown": score_metric(result, "opt", "accepted_blob_rich_unknown_frames"),
            "opt_accepted_blob_rich_unknown_pct": score_metric(result, "opt",
                                                               "accepted_blob_rich_unknown_pct_blob_rich_unknown"),
            "pred_accepted_blob_rich_unknown": score_metric(result, "pred", "accepted_blob_rich_unknown_frames"),
            "pred_accepted_blob_rich_unknown_pct": score_metric(result, "pred",
                                                                "accepted_blob_rich_unknown_pct_blob_rich_unknown"),
            "unknown_blob_rich_reported": unknown_blob_rich.get("reported"),
            "unknown_blob_rich_tracked": unknown_blob_rich.get("tracked"),
            "opt_f1": metric(result, "opt", "f1"),
            "opt_precision": metric(result, "opt", "precision"),
            "opt_recall": metric(result, "opt", "recall"),
            "pred_f1": metric(result, "pred", "f1"),
            "pred_precision": metric(result, "pred", "precision"),
            "pred_recall": metric(result, "pred", "recall"),
            "pred_tp": metric(result, "pred", "TP"),
            "pred_fn": metric(result, "pred", "FN"),
            "pred_position_tp": metric(result, "pred", "position_TP"),
            "pred_position_fn": metric(result, "pred", "position_FN"),
            "pred_position_fp": metric(result, "pred", "position_FP"),
            "pred_scored_frames": metric(result, "pred", "n_scoreable_frames"),
            "pred_rmse_cm": metric(result, "pred", "pos_rmse_cm"),
            "pred_p95_cm": metric(result, "pred", "pos_p95_cm"),
            "pred_position_f1": metric(result, "pred", "position_f1"),
            "pred_position_precision": metric(result, "pred", "position_precision"),
            "pred_position_recall": metric(result, "pred", "position_recall"),
            "pred_position_rmse_cm": metric(result, "pred", "position_rmse_cm"),
            "pred_position_p95_cm": metric(result, "pred", "position_p95_cm"),
            "pred_full_pose_yield_pct": metric(result, "pred", "yield_pct_frame_rows"),
            "pred_position_yield_pct": metric(result, "pred", "position_yield_pct_frame_rows"),
            "wrong_branch_pct_scored": metric(result, "pred", "wrong_branch_pct_scored"),
            "stale_visible_frames": stale.get("frames"),
            "stale_reported": stale.get("reported"),
            "stale_tracked": stale.get("tracked"),
            "stale_position_tracked": stale.get("position_tracked"),
            "stale_reported_position_correct_pct": oov_metric(result, "detect_visible_but_stale",
                                                              "reported_position_correct_pct"),
            "stale_position_tracked_correct_pct": oov_metric(result, "detect_visible_but_stale",
                                                             "position_tracked_correct_pct"),
            "stale_position_tracked_accuracy_pct": oov_metric(result, "detect_visible_but_stale",
                                                              "position_tracked_accuracy_pct"),
            "stale_pos_median_cm": stat(result, "detect_visible_but_stale", "pos_err_cm", "median"),
            "stale_pos_p95_cm": stat(result, "detect_visible_but_stale", "pos_err_cm", "p95"),
            "stale_tracked_pos_p95_cm": stat(result, "detect_visible_but_stale", "pos_err_tracked_cm", "p95"),
            "stale_position_tracked_pos_p95_cm": stat(
                result, "detect_visible_but_stale", "pos_err_position_tracked_cm", "p95"),
            "all_scoreable_frames": all_scoreable.get("frames"),
            "all_scoreable_reported": all_scoreable.get("reported"),
            "all_scoreable_position_tracked": all_scoreable.get("position_tracked"),
            "all_scoreable_reported_position_correct_pct": oov_metric(result, "all_scoreable",
                                                                       "reported_position_correct_pct"),
            "all_scoreable_position_tracked_correct_pct": oov_metric(result, "all_scoreable",
                                                                      "position_tracked_correct_pct"),
            "all_scoreable_position_tracked_accuracy_pct": oov_metric(result, "all_scoreable",
                                                                       "position_tracked_accuracy_pct"),
            "stale_freeze_p95_cm": stat(result, "detect_visible_but_stale", "freeze_err_cm", "p95"),
            "forced_drop_visible_frames": result.get("oov", {}).get("forced_drop_visible", {}).get("frames"),
            "forced_drop_visible_reported": result.get("oov", {}).get("forced_drop_visible", {}).get("reported"),
            "forced_drop_visible_position_tracked": result.get("oov", {}).get(
                "forced_drop_visible", {}).get("position_tracked"),
            "forced_drop_visible_reported_position_correct_pct": oov_metric(
                result, "forced_drop_visible", "reported_position_correct_pct"),
            "forced_drop_visible_position_tracked_correct_pct": oov_metric(
                result, "forced_drop_visible", "position_tracked_correct_pct"),
            "forced_drop_visible_position_tracked_accuracy_pct": oov_metric(
                result, "forced_drop_visible", "position_tracked_accuracy_pct"),
            "forced_drop_visible_p95_cm": stat(result, "forced_drop_visible", "pos_err_cm", "p95"),
            "forced_drop_visible_tracked_p95_cm": stat(result, "forced_drop_visible", "pos_err_tracked_cm", "p95"),
            "forced_drop_visible_position_tracked_p95_cm": stat(
                result, "forced_drop_visible", "pos_err_position_tracked_cm", "p95"),
            "forced_drop_visible_freeze_p95_cm": stat(result, "forced_drop_visible", "freeze_err_cm", "p95"),
            "forced_drop_visible_anchor_100ms_frames": result.get("oov", {}).get(
                "forced_drop_visible_anchor_100ms", {}).get("frames"),
            "forced_drop_visible_anchor_100ms_tracked_p95_cm": stat(
                result, "forced_drop_visible_anchor_100ms", "pos_err_tracked_cm", "p95"),
            "forced_drop_visible_anchor_250ms_frames": result.get("oov", {}).get(
                "forced_drop_visible_anchor_250ms", {}).get("frames"),
            "forced_drop_visible_anchor_250ms_tracked_p95_cm": stat(
                result, "forced_drop_visible_anchor_250ms", "pos_err_tracked_cm", "p95"),
            "forced_drop_visible_anchor_500ms_frames": result.get("oov", {}).get(
                "forced_drop_visible_anchor_500ms", {}).get("frames"),
            "forced_drop_visible_anchor_500ms_tracked_p95_cm": stat(
                result, "forced_drop_visible_anchor_500ms", "pos_err_tracked_cm", "p95"),
            "all_pos_p95_cm": stat(result, "all_scoreable", "pos_err_cm", "p95"),
            "run_dir": result["_run_dir"],
        }
        compliance = verify_drop_compliance(row["run_dir"], row["regime"])
        row["drop_compliance"] = compliance["status"]
        row["drop_csv_rows"] = compliance["csv_rows"]
        row["drop_csv_dropped"] = compliance["csv_dropped"]
        objective = objective_for_row(row)
        row.update({k: v for k, v in objective.items() if k not in ("objective_components", "objective_weights")})
        row["objective_components_json"] = json.dumps(objective["objective_components"], sort_keys=True)
        row["objective_weights_json"] = json.dumps(objective["objective_weights"], sort_keys=True)
        rows.append(row)
    return rows


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git(worktree: Path, *argv: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(worktree), *argv],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def scorer_hashes() -> dict[str, str | None]:
    tools = ROOT / "tools/telemetry"
    return {name: _sha256(tools / name) for name in
            ("dropout_matrix.py", "tracking_metrics.py", "gt_blob_fix.py",
             "smooth_ref.py", "deflip.py", "g2_geom.py")}


def build_provenance(args: argparse.Namespace, captures: list[str], regimes: list[str]) -> dict[str, Any]:
    """Pin everything a future reader needs to trust (or distrust) this run."""
    binary = Path(args.binary).resolve()
    worktree = next((p for p in binary.parents if (p / ".git").exists()), None)
    git_info = None
    if worktree is not None:
        dirty = _git(worktree, "status", "--porcelain") or ""
        diff = (_git(worktree, "diff") or "") if dirty else ""
        git_info = {
            "worktree": str(worktree),
            "head": _git(worktree, "rev-parse", "HEAD"),
            "branch": _git(worktree, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(dirty),
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None,
        }
    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "host": os.uname().nodename,
        "loadavg_start": os.getloadavg(),
        "binary": {
            "path": str(binary),
            "sha256": _sha256(binary),
            "mtime": datetime.fromtimestamp(binary.stat().st_mtime).isoformat(),
            "git": git_info,
        },
        "scorer": scorer_hashes(),
        "gate_baseline": {
            "path": str(args.baseline),
            "sha256": _sha256(args.baseline),
        },
        "gt_blobfix_caches": {
            cap: {p.name: _sha256(p)
                  for p in sorted((CAPTURES[cap]["reference"] / "telemetry").glob("gt_blobfix_*.npz"))}
            for cap in captures
        },
        # The replay contract, per capture: which camera config the cell was handed, the
        # commanded gain that config replays at (an absent key means the gain-16 calibration
        # point, so recording the resolved value is what makes a mis-scaled run visible), and
        # the IMU calibration seeded in. All three are chosen by replay_contract, never
        # inherited from the caller's shell.
        "replay_contract": {
            cap: {
                "cams": str(capture_cams(args, cap)),
                "ctrl_gain": cams_ctrl_gain(capture_cams(args, cap)),
                "imu_cal_dir": str(imu_cal_dir_for_capture(CAPTURES[cap]["reference"])),
            }
            for cap in captures
        },
        "captures": captures,
        "regimes": regimes,
        "reference_qc": args.reference_qc,
    }


def summarize_objective(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [as_float(row.get("objective_score")) for row in rows]
    scores = [score for score in scores if score is not None]
    losses = [as_float(row.get("objective_loss")) for row in rows]
    losses = [loss for loss in losses if loss is not None]
    hard_fail_rows = [row for row in rows if row.get("objective_hard_fail")]
    positive_scores = [score for score in scores if score > 0.0]
    return {
        "score_geomean": (
            0.0
            if len(positive_scores) != len(scores) or not positive_scores
            else 100.0 * math.exp(sum(math.log(score / 100.0) for score in positive_scores) / len(positive_scores))
        ),
        "score_mean": sum(scores) / len(scores) if scores else None,
        "score_min": min(scores) if scores else None,
        "loss_mean": sum(losses) / len(losses) if losses else None,
        "hard_fail_rows": len(hard_fail_rows),
    }


#: Objective-score tolerance for the non-regression floors, in score points on the 0-100 scale.
#: The harness is bit-deterministic (a re-run reproduces a reference exactly), so a replicate needs
#: no tolerance at all. This is not for replicates: it is the resolution below which a CHANGED
#: trajectory is not a quality claim. objective_score is 100*exp(-loss/weight), so 0.01 points is a
#: ~1e-4 relative move of the aggregate -- reached by p95 order statistics shifting tens of
#: micrometres, which is far under this tracker's own noise on a 32-LED constellation at ~0.5 m.
#: Without it the floors demand bit-identical trajectories forever and reject strict improvements:
#: recall-reform-20260724 raised opt_recall and lowered RMSE and p95 on clean2/bursts-long/dev1 while
#: a forced-drop p95 moved +44 um, dropping the composite 0.00014 and "failing". Behaviour is still
#: gated exactly: identity hard failures are absolute, and a real regression is orders of magnitude
#: larger than this.
OBJECTIVE_SCORE_TOLERANCE = 0.01


def row_identity(row: dict[str, Any]) -> str:
    return f"{row['capture']}/{row['regime']}/dev{int(row['device'])}"


def evaluate_gate(
    rows: list[dict[str, Any]],
    objective: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Enforce row hard failures and non-regression against an exact matrix profile."""
    identities = {row_identity(row): row for row in rows}
    if len(identities) != len(rows):
        return {"passed": False, "failures": ["duplicate matrix row identity"]}
    expected_keyset = set(identities)
    profile = next(
        (
            item
            for item in baseline.get("profiles", [])
            if set(item.get("row_objective_minima", {})) == expected_keyset
        ),
        None,
    )
    failures = []
    for identity, row in sorted(identities.items()):
        hard_fail = str(row.get("objective_hard_fail") or "")
        if hard_fail:
            failures.append(f"{identity}: objective hard fail: {hard_fail}")
    if profile is None:
        failures.append(
            "no pinned baseline profile for rows: " + ",".join(sorted(expected_keyset))
        )
    else:
        for identity, minimum in sorted(profile["row_objective_minima"].items()):
            actual = as_float(identities[identity].get("objective_score"))
            floor = as_float(minimum)
            if actual is None or floor is None or actual + OBJECTIVE_SCORE_TOLERANCE < floor:
                failures.append(
                    f"{identity}: objective_score {actual!r} below pinned {floor!r}"
                )
        actual_geomean = as_float(objective.get("score_geomean"))
        pinned_geomean = as_float(profile.get("objective_geomean_min"))
        if (
            actual_geomean is None
            or pinned_geomean is None
            or actual_geomean + OBJECTIVE_SCORE_TOLERANCE < pinned_geomean
        ):
            failures.append(
                f"matrix objective geomean {actual_geomean!r} below pinned "
                f"{pinned_geomean!r}"
            )
    return {
        "passed": not failures,
        "profile": profile.get("name") if profile else None,
        "failures": failures,
    }


def load_gate_baseline(path: Path) -> dict[str, Any]:
    baseline = json.loads(path.read_text())
    if baseline.get("schema_version") != 1 or not isinstance(baseline.get("profiles"), list):
        raise ValueError(f"unsupported dropout baseline schema: {path}")
    return baseline


def print_rows(rows: list[dict[str, Any]]) -> None:
    print("capture regime        dev obj  cov% tpAcc% optF1 poseF1 posF1 poseRMSE posRMSE poseY% posY% idX 2way staleN staleTP% staleP95 staleTP95 dropN dropTP% dropP95 dropTP95 a250N a250TP95")
    print("-" * 189)
    for row in rows:
        def fmt(value: Any, width: int = 7, prec: int = 3) -> str:
            if value is None:
                return " " * (width - 3) + "nan"
            if isinstance(value, float):
                return f"{value:{width}.{prec}f}"
            return f"{value:{width}}"

        print(
            f"{row['capture']:7s} {row['regime']:13s} {row['device']:3d} "
            f"{fmt(row['objective_score'], 5, 1)} "
            f"{fmt(as_float(row.get('scoreable_pct_total')), 5, 1)} "
            f"{fmt(as_float(row.get('all_scoreable_position_tracked_accuracy_pct')), 6, 1)} "
            f"{fmt(row['opt_f1'])} {fmt(row['pred_f1'])} {fmt(row['pred_position_f1'])} "
            f"{fmt(row['pred_rmse_cm'], 8, 2)} {fmt(row['pred_position_rmse_cm'], 8, 2)} "
            f"{fmt(row['pred_full_pose_yield_pct'], 6, 1)} {fmt(row['pred_position_yield_pct'], 5, 1)} "
            f"{fmt(row['identity_other_report_explains_ref'], 3, 0)} {fmt(row['identity_strict_two_way_swap_frames'], 4, 0)} "
            f"{fmt(row['stale_visible_frames'], 6, 0)} {fmt(row['stale_position_tracked_correct_pct'], 8, 1)} "
            f"{fmt(row['stale_pos_p95_cm'], 8, 2)} "
            f"{fmt(row['stale_tracked_pos_p95_cm'], 9, 2)} "
            f"{fmt(row['forced_drop_visible_frames'], 5, 0)} "
            f"{fmt(row['forced_drop_visible_position_tracked_correct_pct'], 7, 1)} "
            f"{fmt(row['forced_drop_visible_p95_cm'], 7, 2)} "
            f"{fmt(row['forced_drop_visible_tracked_p95_cm'], 8, 2)} "
            f"{fmt(row['forced_drop_visible_anchor_250ms_frames'], 5, 0)} "
            f"{fmt(row['forced_drop_visible_anchor_250ms_tracked_p95_cm'], 9, 2)}"
        )
    flagged = [row for row in rows if row.get("objective_hard_fail")]
    if flagged:
        print()
        for row in flagged:
            print(f"  HARD FAIL {row['capture']}/{row['regime']}/dev{row['device']}: {row['objective_hard_fail']}")


def report(
    rows: list[dict[str, Any]],
    objective: dict[str, Any],
    gate: dict[str, Any],
) -> int:
    print_rows(rows)
    coverages = [c for c in (as_float(row.get("scoreable_pct_total")) for row in rows) if c is not None]
    coverage = (f" | scoreable coverage min={min(coverages):.1f}% max={max(coverages):.1f}% of frames"
                if coverages else " | scoreable coverage UNKNOWN")
    print(f"\nobjective: geomean={objective['score_geomean']:.3f} "
          f"mean={objective['score_mean']:.3f} min={objective['score_min']:.3f} "
          f"hard_fail_rows={objective['hard_fail_rows']}{coverage}")
    noncompliant = [
        row for row in rows
        if any(reason.startswith(COMPLIANCE_FAIL_PREFIXES)
               for reason in str(row.get("objective_hard_fail") or "").split(",") if reason)
    ]
    if noncompliant:
        print(f"\nMEASUREMENT NON-COMPLIANT rows: {len(noncompliant)} — these numbers are NOT publishable",
              file=sys.stderr)
        return 2
    if not gate["passed"]:
        print("\nMATRIX GATE FAILED:", file=sys.stderr)
        for failure in gate["failures"]:
            print(f"  {failure}", file=sys.stderr)
        return 3
    print(f"matrix gate: PASS ({gate['profile']})")
    return 0


def rescore(args: argparse.Namespace) -> int:
    """Re-flatten + re-score an existing run dir with the CURRENT scorer (no replays)."""
    runroot = args.rescore.resolve()
    summary_path = runroot / "summary.json"
    data = json.loads(summary_path.read_text())
    results = data.get("results")
    if not results:
        print(f"no results in {summary_path}", file=sys.stderr)
        return 2
    rows = flatten(results)
    objective = summarize_objective(rows)
    gate = evaluate_gate(rows, objective, load_gate_baseline(args.baseline))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = runroot / f"summary-rescored-{stamp}.json"
    out_path.write_text(json.dumps({
        "rescored_from": str(summary_path),
        "rescored_at": datetime.now().astimezone().isoformat(),
        "scorer": scorer_hashes(),
        "rows": rows,
        "objective": objective,
        "gate": gate,
    }, indent=2))
    code = report(rows, objective, gate)
    print(f"\nwrote {out_path}")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=None,
                        help="offline_vio_replay binary under test (REQUIRED; no default so the "
                             "binary is always an explicit, pinnable choice)")
    parser.add_argument("--cams", type=Path, default=None,
                        help="override the camera config for every cell (default: each capture's own "
                             "provenance snapshot, else the pinned pre-provenance config)")
    parser.add_argument("--ctrl-left", type=Path, default=DEFAULT_LEFT)
    parser.add_argument("--ctrl-right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="pinned corrected-harness matrix objective baseline")
    parser.add_argument("--capture", action="append", choices=sorted(CAPTURES), help="default: xv1 and clean2")
    parser.add_argument("--regime", action="append", choices=sorted(REGIMES), help="default compact OOV matrix")
    parser.add_argument("--reference-qc", choices=("none", "corrupt", "confirmed"), default="confirmed")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rescore", type=Path, default=None,
                        help="re-score an existing run dir's summary.json with the current scorer; no replays")
    args = parser.parse_args()

    if args.rescore is not None:
        if not args.baseline.exists():
            parser.error(f"--baseline file not found: {args.baseline}")
        args.baseline = args.baseline.resolve()
        return rescore(args)
    if args.binary is None:
        parser.error("--binary is required (no default — pin the binary under test explicitly)")
    if not args.binary.exists():
        parser.error(f"binary not found: {args.binary}")
    args.binary = args.binary.resolve()
    # Resolve against the invoker's CWD now: replays run with a per-cell cwd, which would
    # silently re-anchor relative paths (the H2-era dev2refit probe mis-run class).
    for name in ("cams", "ctrl_left", "ctrl_right", "baseline"):
        path = getattr(args, name)
        if path is None:
            continue
        if not path.exists():
            parser.error(f"--{name.replace('_', '-')} file not found: {path}")
        setattr(args, name, path.resolve())

    captures = args.capture or ["xv1", "clean2"]
    regimes = args.regime or ["normal", "periodic-300", "bursts-short", "bursts-long"]
    if args.out is not None:
        runroot = args.out if args.out.is_absolute() else ROOT / args.out
    else:
        runroot = ROOT / "results" / f"dropout-matrix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    runroot = runroot.resolve()
    runroot.mkdir(parents=True, exist_ok=True)

    provenance = build_provenance(args, captures, regimes)
    (runroot / "provenance.json").write_text(json.dumps(provenance, indent=2))

    all_results: list[dict[str, Any]] = []
    for capture_name in captures:
        for regime_name in regimes:
            print(f"running {capture_name}/{regime_name}", flush=True)
            results = run_one(args, capture_name, regime_name, runroot)
            for result in results:
                result["_reference_qc"] = args.reference_qc
            all_results.extend(results)

    rows = flatten(all_results)
    objective = summarize_objective(rows)
    gate = evaluate_gate(rows, objective, load_gate_baseline(args.baseline))
    provenance["loadavg_end"] = os.getloadavg()
    (runroot / "summary.json").write_text(json.dumps(
        {"provenance": provenance, "results": all_results, "rows": rows,
         "objective": objective, "gate": gate}, indent=2))
    with (runroot / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    code = report(rows, objective, gate)
    print(f"\nwrote {runroot}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
