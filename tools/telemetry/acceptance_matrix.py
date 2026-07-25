#!/usr/bin/env python3
"""Run the G2 golden-vs-current acceptance matrix.

This is the reusable version of the ad hoc acceptance folders:
  * canonical captures: xv1 and clean2
  * modes: frame cadence and render90
  * artifacts: replay CSVs, tracking_metrics JSON, live_health JSON/text
  * gates: current must match or beat the golden binary's quality and not lose runtime

Both binaries are explicit, pinnable arguments — there is no default, because a default
outlives the worktree it names and a benchmark against a silently-substituted binary is
worse than no benchmark. The golden binary must carry the replay harness's metric-only
instrumentation (G2_REPLAY_RENDER_HZ): do not compare a non-render-capable golden binary
against render90.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_contract import cams_for_capture, imu_cal_dir_for_capture, replay_env  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEFT = Path.home() / ".config/monado/wmr/controller_A85K1111630014L.json"
DEFAULT_RIGHT = Path.home() / ".config/monado/wmr/controller_A85K5091930012R.json"


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

QUALITY_METRICS = {
    "precision": "up",
    "recall": "up",
    "f1": "up",
    "position_f1": "up",
    "yield_pct_frame_rows": "up",
    "pos_rmse_cm": "down",
    "pos_p95_cm": "down",
    "ori_rms_deg": "down",
    "ori_p95_deg": "down",
    "wrong_branch_pct_scored": "down",
}

RENDER_METRICS = {
    "pred_tracked_pct": "up",
    "first_pred_tracked_ms": "down",
    "untracked_run_ms.p95": "down",
    "untracked_run_ms.max": "down",
    "optical_age_ms.p95": "down",
    "optical_age_ms.max": "down",
    "pos_step_cm.p95": "down",
    "pos_step_cm.p99": "down",
    "pos_step_cm.max": "down",
    "rot_step_deg.p95": "down",
    "rot_step_deg.p99": "down",
    "rot_step_deg.max": "down",
    "jitter_cm.rms_cm": "down",
    "jitter_cm.p95_cm": "down",
    "pred_to_hmd_cm.p95": "down",
}

RENDER_TOLERANCE = {
    # Step percentiles are denominator-sensitive: if current tracks more frames, the p95/p99 set
    # can include extra motion samples while the actual user-facing continuity improves. Keep a
    # small physical tolerance and still gate hard jumps through max plus jitter/untracked runs.
    "pos_step_cm.p95": 0.25,
    "pos_step_cm.p99": 0.15,
    "rot_step_deg.p95": 0.5,
    "rot_step_deg.p99": 0.5,
}


def _run_checked(cmd: list[str], env: dict[str, str], cwd: Path, stdout: Path, stderr: Path) -> float:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with stdout.open("w") as out, stderr.open("w") as err:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=out, stderr=err, text=True)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}; stderr={stderr}")
    return elapsed


def _replay_env(capture_name: str, mode: str, render_hz: float) -> dict[str, str]:
    settings = {"G2_REPLAY_RENDER_HZ": str(render_hz)} if mode.startswith("render") else {}
    return replay_env(settings, imu_cal_dir=imu_cal_dir_for_capture(CAPTURES[capture_name]["reference"]))


def _stat_median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return math.nan
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _nested(block: dict[str, Any], dotted: str) -> Any:
    cur: Any = block
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _run_one(args: argparse.Namespace,
             branch: str,
             binary: Path,
             capture_name: str,
             mode: str,
             runroot: Path) -> dict[str, Any]:
    cap = CAPTURES[capture_name]
    cell = runroot / branch / capture_name / mode
    cell.mkdir(parents=True, exist_ok=True)

    times: list[float] = []
    for rep in range(1, args.repeats + 1):
        rep_dir = cell / f"rep{rep}"
        out_dir = rep_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(binary),
            str(cap["frames"]),
            str(args.cams or cams_for_capture(cap["reference"])),
            str(cap["telemetry"]),
            str(args.ctrl_left),
            str(args.ctrl_right),
            str(out_dir),
        ]
        elapsed = _run_checked(
            cmd,
            _replay_env(capture_name, mode, args.render_hz),
            binary.parents[1],
            rep_dir / "replay.stdout",
            rep_dir / "replay.stderr",
        )
        times.append(elapsed)
        if mode.startswith("render"):
            for dev in (1, 2):
                if not (out_dir / f"dev{dev}_render.csv").is_file():
                    raise RuntimeError(
                        f"{branch}/{capture_name}/{mode}: missing dev{dev}_render.csv; "
                        f"use a render-capable audit harness for this mode"
                    )

    scored_out = cell / "rep1/out"
    metrics_json = cell / "tracking_metrics.json"
    _run_checked(
        [
            sys.executable,
            str(ROOT / "tools/telemetry/tracking_metrics.py"),
            str(cap["reference"]),
            str(scored_out),
            "--candidate-name",
            f"{branch}-{capture_name}-{mode}",
            "--out",
            str(metrics_json),
        ],
        os.environ.copy(),
        ROOT,
        cell / "tracking_metrics.stdout",
        cell / "tracking_metrics.stderr",
    )

    live_json = cell / "live_health.json"
    live_txt = cell / "live_health.txt"
    _run_checked(
        [
            sys.executable,
            str(ROOT / "tools/telemetry/live_health_metrics.py"),
            str(cap["reference"]),
            "--replay-dir",
            str(scored_out),
            "--json",
            str(live_json),
        ],
        os.environ.copy(),
        ROOT,
        live_txt,
        cell / "live_health.stderr",
    )

    return {
        "branch": branch,
        "binary": str(binary),
        "capture": capture_name,
        "mode": mode,
        "cell": str(cell),
        "scored_out": str(scored_out),
        "replay_times_s": times,
        "replay_time_median_s": _stat_median(times),
        "tracking_metrics_json": str(metrics_json),
        "live_health_json": str(live_json),
    }


def _metric_rows(run: dict[str, Any], streams: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = json.loads(Path(run["tracking_metrics_json"]).read_text())
    for result in results:
        for stream in streams:
            score = result.get("scores", {}).get(stream)
            if not isinstance(score, dict):
                continue
            row = {
                "branch": run["branch"],
                "capture": run["capture"],
                "mode": run["mode"],
                "device": int(result["device"]),
                "stream": stream,
            }
            for name in QUALITY_METRICS:
                row[name] = score.get(name)
            rows.append(row)
    return rows


def _render_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    live = json.loads(Path(run["live_health_json"]).read_text())
    render = live.get("render_replay", {})
    for dev_key, block in render.items():
        row = {
            "branch": run["branch"],
            "capture": run["capture"],
            "mode": run["mode"],
            "device": int(dev_key),
        }
        for name in RENDER_METRICS:
            row[name] = _nested(block, name)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _compare_scalar(cur: float | None, ref: float | None, direction: str, eps: float) -> tuple[bool, float | None]:
    if cur is None or ref is None:
        return False, None
    delta = cur - ref
    if direction == "up":
        return cur + eps >= ref, delta
    return cur <= ref + eps, delta


def _quality_comparisons(rows: list[dict[str, Any]], eps: float) -> tuple[list[dict[str, Any]], list[str]]:
    by_key = {
        (r["branch"], r["capture"], r["mode"], r["device"], r["stream"]): r
        for r in rows
    }
    out: list[dict[str, Any]] = []
    failures: list[str] = []
    keys = sorted(k for k in by_key if k[0] == "current")
    for _, capture, mode, device, stream in keys:
        cur = by_key[("current", capture, mode, device, stream)]
        ref = by_key.get(("golden", capture, mode, device, stream))
        if ref is None:
            failures.append(f"missing golden quality row for {capture}/{mode}/dev{device}/{stream}")
            continue
        for metric, direction in QUALITY_METRICS.items():
            ok, delta = _compare_scalar(_as_float(cur.get(metric)), _as_float(ref.get(metric)), direction, eps)
            item = {
                "group": "quality",
                "capture": capture,
                "mode": mode,
                "device": device,
                "stream": stream,
                "metric": metric,
                "golden": ref.get(metric),
                "current": cur.get(metric),
                "delta": delta,
                "direction": direction,
                "status": "ok" if ok else "FAIL",
            }
            out.append(item)
            if not ok:
                failures.append(f"quality regression: {capture}/{mode}/dev{device}/{stream}/{metric}")
    return out, failures


def _render_comparisons(rows: list[dict[str, Any]], eps: float) -> tuple[list[dict[str, Any]], list[str]]:
    by_key = {(r["branch"], r["capture"], r["mode"], r["device"]): r for r in rows}
    out: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, capture, mode, device in sorted(k for k in by_key if k[0] == "current"):
        cur = by_key[("current", capture, mode, device)]
        ref = by_key.get(("golden", capture, mode, device))
        if ref is None:
            failures.append(f"missing golden render row for {capture}/{mode}/dev{device}")
            continue
        for metric, direction in RENDER_METRICS.items():
            tol = max(eps, RENDER_TOLERANCE.get(metric, eps))
            ok, delta = _compare_scalar(_as_float(cur.get(metric)), _as_float(ref.get(metric)), direction, tol)
            item = {
                "group": "render",
                "capture": capture,
                "mode": mode,
                "device": device,
                "metric": metric,
                "golden": ref.get(metric),
                "current": cur.get(metric),
                "delta": delta,
                "direction": direction,
                "tolerance": tol,
                "status": "ok" if ok else "FAIL",
            }
            out.append(item)
            if not ok:
                failures.append(f"render regression: {capture}/{mode}/dev{device}/{metric}")
    return out, failures


def _runtime_comparisons(runs: list[dict[str, Any]], ratio: float, slop_s: float) -> tuple[list[dict[str, Any]], list[str]]:
    by_key = {(r["branch"], r["capture"], r["mode"]): r for r in runs}
    out: list[dict[str, Any]] = []
    failures: list[str] = []
    for _, capture, mode in sorted(k for k in by_key if k[0] == "current"):
        cur = by_key[("current", capture, mode)]
        ref = by_key.get(("golden", capture, mode))
        if ref is None:
            failures.append(f"missing golden runtime for {capture}/{mode}")
            continue
        cur_s = _as_float(cur.get("replay_time_median_s"))
        ref_s = _as_float(ref.get("replay_time_median_s"))
        ok = cur_s is not None and ref_s is not None and cur_s <= ref_s * ratio + slop_s
        item = {
            "group": "runtime",
            "capture": capture,
            "mode": mode,
            "metric": "replay_time_median_s",
            "golden": ref_s,
            "current": cur_s,
            "delta": None if cur_s is None or ref_s is None else cur_s - ref_s,
            "direction": "down",
            "status": "ok" if ok else "FAIL",
        }
        out.append(item)
        if not ok:
            failures.append(f"runtime regression: {capture}/{mode}")
    return out, failures


def _print_summary(comparisons: list[dict[str, Any]], failures: list[str], runroot: Path) -> None:
    print(f"wrote {runroot}")
    runtime = [r for r in comparisons if r["group"] == "runtime"]
    if runtime:
        print("\nruntime median seconds (current vs golden):")
        for r in runtime:
            print(
                f"  {r['capture']}/{r['mode']}: "
                f"{r['current']:.2f} vs {r['golden']:.2f} ({r['status']})"
            )
    qual_fail = [r for r in comparisons if r["group"] == "quality" and r["status"] != "ok"]
    rend_fail = [r for r in comparisons if r["group"] == "render" and r["status"] != "ok"]
    print(f"\nquality checks: {sum(1 for r in comparisons if r['group'] == 'quality')} total, {len(qual_fail)} fail")
    print(f"render checks: {sum(1 for r in comparisons if r['group'] == 'render')} total, {len(rend_fail)} fail")
    if failures:
        print("\nFAILURES:")
        for item in failures[:40]:
            print(f"  {item}")
        if len(failures) > 40:
            print(f"  ... {len(failures) - 40} more")
    else:
        print("\nRESULT: PASS")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden-bin", type=Path, required=True,
                    help="reference offline_vio_replay binary (REQUIRED; no default so the "
                         "benchmark is always an explicit, pinnable choice)")
    ap.add_argument("--current-bin", type=Path, required=True,
                    help="offline_vio_replay binary under test (REQUIRED)")
    ap.add_argument("--cams", type=Path, default=None,
                    help="override the camera config for every cell (default: each capture's own "
                         "provenance snapshot, else the pinned pre-provenance config)")
    ap.add_argument("--ctrl-left", type=Path, default=DEFAULT_LEFT)
    ap.add_argument("--ctrl-right", type=Path, default=DEFAULT_RIGHT)
    ap.add_argument("--capture", action="append", choices=sorted(CAPTURES), help="default: xv1 and clean2")
    ap.add_argument("--mode", action="append", choices=("frame", "render90"), help="default: frame and render90")
    ap.add_argument("--stream", action="append", choices=("opt", "pred"), help="default: opt and pred")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--render-hz", type=float, default=90.0)
    ap.add_argument("--quality-eps", type=float, default=1e-9)
    ap.add_argument("--render-eps", type=float, default=1e-9)
    ap.add_argument("--runtime-ratio", type=float, default=1.05)
    ap.add_argument("--runtime-slop-s", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    for path in (args.golden_bin, args.current_bin, args.cams, args.ctrl_left, args.ctrl_right):
        if path is not None and not path.is_file():
            ap.error(f"missing required file: {path}")

    captures = args.capture or ["xv1", "clean2"]
    modes = args.mode or ["frame", "render90"]
    streams = args.stream or ["opt", "pred"]
    runroot = args.out or ROOT / "results" / f"acceptance-matrix-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    runroot = runroot.resolve()
    runroot.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for branch, binary in (("golden", args.golden_bin.resolve()), ("current", args.current_bin.resolve())):
        for capture in captures:
            for mode in modes:
                print(f"running {branch}/{capture}/{mode}", flush=True)
                runs.append(_run_one(args, branch, binary, capture, mode, runroot))

    metric_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    for run in runs:
        metric_rows.extend(_metric_rows(run, streams))
        render_rows.extend(_render_rows(run))

    comparisons: list[dict[str, Any]] = []
    failures: list[str] = []
    for block, fail in (
        _quality_comparisons(metric_rows, args.quality_eps),
        _render_comparisons(render_rows, args.render_eps),
        _runtime_comparisons(runs, args.runtime_ratio, args.runtime_slop_s),
    ):
        comparisons.extend(block)
        failures.extend(fail)

    _write_csv(runroot / "metrics.csv", metric_rows)
    _write_csv(runroot / "render_health.csv", render_rows)
    _write_csv(runroot / "comparisons.csv", comparisons)
    payload = {
        "branches": {
            "golden": str(args.golden_bin.resolve()),
            "current": str(args.current_bin.resolve()),
        },
        "captures": {name: {k: str(v) for k, v in CAPTURES[name].items()} for name in captures},
        "modes": modes,
        "streams": streams,
        "repeats": args.repeats,
        "render_hz": args.render_hz,
        "gates": {
            "quality_eps": args.quality_eps,
            "render_eps": args.render_eps,
            "runtime_ratio": args.runtime_ratio,
            "runtime_slop_s": args.runtime_slop_s,
        },
        "runs": runs,
        "metric_rows": metric_rows,
        "render_rows": render_rows,
        "comparisons": comparisons,
        "failures": failures,
    }
    (runroot / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    _print_summary(comparisons, failures, runroot)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
