#!/usr/bin/env python3
"""Summarize G2 matcher candidate/twin telemetry.

This is intentionally narrower than the cleaned-GT metrics. It answers one
question: what did the front-end rank, select, accept, recover, or reject before
fusion hid the decision? That is the evidence needed before tuning flip gates.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from g2_geom import load_stream
from manifest import DEVICE_NAMES, POSE_OUTCOME, Manifest


# candidate.stage carries enum association_hypothesis_source (association_hypothesis.h).
STAGE_NAMES = {
    1: "prior_pose",
    2: "last_seen",
    3: "labelled_pnp",
    4: "prior_labelled_pnp",
    5: "joint_pnp",
    6: "cold_search",
    7: "partner_ring",
}
CANDIDATE_NAMES = {0: "primary", 1: "twin"}


def _finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else 0.0


def _quantiles(a: np.ndarray) -> dict[str, float | None]:
    a = _finite(np.asarray(a, dtype=float))
    if len(a) == 0:
        return {"p50": None, "p75": None, "p95": None, "max": None}
    return {
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def _row_summary(rows: np.ndarray) -> dict[str, Any]:
    selected = rows["selected"] != 0
    accepted = rows["outcome"] == 1
    recovered = rows["outcome"] == 2
    selected_rows = rows[selected]
    selected_good = selected_rows[selected_rows["outcome"] != 0]
    twin_rows = rows[rows["had_twin"] != 0]
    return {
        "rows": int(len(rows)),
        "selected": int(np.count_nonzero(selected)),
        "accepted": int(np.count_nonzero(accepted)),
        "recovered": int(np.count_nonzero(recovered)),
        "rejected_selected": int(np.count_nonzero(selected & (rows["outcome"] == 0))),
        "had_twin_rows": int(len(twin_rows)),
        "selected_good_with_4_or_fewer_blobs": int(
            np.count_nonzero((selected_rows["outcome"] != 0) & (selected_rows["blobs_matched"] <= 4))
        )
        if len(selected_rows)
        else 0,
        "selected_good_prior_untrusted": int(
            np.count_nonzero((selected_rows["outcome"] != 0) & (selected_rows["prior_tilt_trusted"] == 0))
        )
        if len(selected_rows)
        else 0,
        "selected_good_reproj_px": _quantiles(selected_good["reproj_err_px"]) if len(selected_good) else _quantiles([]),
        "selected_good_prior_cost": _quantiles(selected_good["prior_cost"]) if len(selected_good) else _quantiles([]),
        "selected_good_yaw_sigma_deg": _quantiles(np.degrees(selected_good["yaw_sigma_rad"]))
        if len(selected_good)
        else _quantiles([]),
        "selected_good_tilt_err_deg": _quantiles(np.degrees(selected_good["tilt_err_rad"]))
        if len(selected_good)
        else _quantiles([]),
        "selected_good_yaw_err_deg": _quantiles(np.degrees(selected_good["yaw_err_rad"]))
        if len(selected_good)
        else _quantiles([]),
    }


def _group(rows: np.ndarray, *fields: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if len(rows) == 0:
        return out
    seen = sorted({tuple(int(r[f]) for f in fields) for r in rows})
    for key in seen:
        mask = np.ones(len(rows), dtype=bool)
        for field, value in zip(fields, key):
            mask &= rows[field] == value
        item: dict[str, Any] = {}
        for field, value in zip(fields, key):
            if field == "device_id":
                item[field] = DEVICE_NAMES.get(value, str(value))
            elif field == "stage":
                item[field] = STAGE_NAMES.get(value, str(value))
            elif field == "candidate":
                item[field] = CANDIDATE_NAMES.get(value, str(value))
            elif field == "outcome":
                item[field] = POSE_OUTCOME.get(value, str(value))
            else:
                item[field] = value
        item.update(_row_summary(rows[mask]))
        out.append(item)
    return out


def _pairwise_twins(rows: np.ndarray) -> dict[str, Any]:
    twin = rows[rows["had_twin"] != 0]
    if len(twin) == 0:
        return {"pairs": 0}
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for i, r in enumerate(twin):
        key = (int(r["device_id"]), int(r["cam_id"]), int(r["stage"]), int(r["t_mono_ns"]))
        groups.setdefault(key, []).append(i)

    pairs = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        g = twin[idxs]
        prim = g[g["candidate"] == 0]
        tw = g[g["candidate"] == 1]
        if len(prim) == 0 or len(tw) == 0:
            continue
        p = prim[0]
        t = tw[0]
        selected = g[g["selected"] != 0]
        if len(selected) == 0:
            selected_candidate = -1
            selected_outcome = 0
        else:
            selected_candidate = int(selected[0]["candidate"])
            selected_outcome = int(selected[0]["outcome"])
        pairs.append(
            {
                "device_id": int(p["device_id"]),
                "stage": int(p["stage"]),
                "selected_candidate": selected_candidate,
                "selected_outcome": selected_outcome,
                "primary_total": float(p["total_cost"]),
                "twin_total": float(t["total_cost"]),
                "primary_prior": float(p["prior_cost"]),
                "twin_prior": float(t["prior_cost"]),
                "primary_reproj": float(p["reproj_err_px"]),
                "twin_reproj": float(t["reproj_err_px"]),
                "primary_tilt_deg": float(np.degrees(p["tilt_err_rad"])),
                "twin_tilt_deg": float(np.degrees(t["tilt_err_rad"])),
                "primary_yaw_deg": float(np.degrees(p["yaw_err_rad"])),
                "twin_yaw_deg": float(np.degrees(t["yaw_err_rad"])),
                "blobs": int(p["blobs_matched"]),
                "visible": int(p["leds_visible"]),
            }
        )

    if not pairs:
        return {"pairs": 0}
    selected_twin = sum(1 for p in pairs if p["selected_candidate"] == 1)
    selected_primary = sum(1 for p in pairs if p["selected_candidate"] == 0)
    selected_good = [p for p in pairs if p["selected_outcome"] != 0]
    risky_4_blob = [p for p in selected_good if p["blobs"] <= 4]
    winner_margin = np.array(
        [
            (p["primary_total"] - p["twin_total"]) if p["selected_candidate"] == 1 else (p["twin_total"] - p["primary_total"])
            for p in pairs
            if p["selected_candidate"] in (0, 1)
        ],
        dtype=float,
    )
    return {
        "pairs": len(pairs),
        "selected_primary": selected_primary,
        "selected_twin": selected_twin,
        "selected_good": len(selected_good),
        "selected_good_4_or_fewer_blobs": len(risky_4_blob),
        "winner_margin_total_cost": _quantiles(winner_margin),
        "selected_twin_pct": _pct(selected_twin, selected_primary + selected_twin),
        "risky_examples": sorted(risky_4_blob, key=lambda p: min(p["primary_total"], p["twin_total"]))[:20],
    }


def _write_csv(path: Path, rows: np.ndarray) -> None:
    selected = rows[rows["selected"] != 0]
    selected = np.sort(selected, order=["device_id", "t_mono_ns", "cam_id"])
    cols = [
        "t_mono_ns",
        "device_id",
        "cam_id",
        "stage",
        "candidate",
        "outcome",
        "had_twin",
        "blobs_matched",
        "leds_visible",
        "unmatched_blobs",
        "reproj_err_px",
        "prior_cost",
        "total_cost",
        "yaw_sigma_rad",
        "tilt_err_rad",
        "yaw_err_rad",
        "prior_tilt_trusted",
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in selected:
            w.writerow([r[c].item() if hasattr(r[c], "item") else r[c] for c in cols])


def analyze(telemetry_dir: Path) -> dict[str, Any]:
    manifest = Manifest.load(telemetry_dir)
    if "candidate" not in manifest.streams:
        raise KeyError(f"no candidate stream in {telemetry_dir}")
    rows = load_stream(telemetry_dir, manifest, "candidate")
    return {
        "telemetry_dir": str(telemetry_dir),
        "rows": int(len(rows)),
        "manifest_rows_written": manifest.streams["candidate"].rows_written,
        "overflow_total": manifest.streams["candidate"].overflow_total,
        "overall": _row_summary(rows),
        "by_device_stage": _group(rows, "device_id", "stage"),
        "by_device_stage_candidate_outcome": _group(rows, "device_id", "stage", "candidate", "outcome"),
        "twins": _pairwise_twins(rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("telemetry_dir", type=Path, help="directory containing manifest.json and candidate.bin")
    ap.add_argument("--json", type=Path, help="write full JSON summary")
    ap.add_argument("--selected-csv", type=Path, help="write selected-candidate rows for manual inspection")
    args = ap.parse_args()

    result = analyze(args.telemetry_dir)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    if args.selected_csv:
        rows = load_stream(args.telemetry_dir, Manifest.load(args.telemetry_dir), "candidate")
        _write_csv(args.selected_csv, rows)

    print(f"candidate rows: {result['rows']} (manifest {result['manifest_rows_written']}, overflow {result['overflow_total']})")
    print("\nby device/stage:")
    for row in result["by_device_stage"]:
        print(
            f"  {row['device_id']:>5} {row['stage']:<14} rows={row['rows']:5d} "
            f"selected={row['selected']:5d} accepted={row['accepted']:5d} recovered={row['recovered']:4d} "
            f"rej_selected={row['rejected_selected']:4d} <=4blob_good={row['selected_good_with_4_or_fewer_blobs']:4d} "
            f"prior_untrusted_good={row['selected_good_prior_untrusted']:4d}"
        )
    twins = result["twins"]
    print("\ntwins:")
    print(
        f"  pairs={twins.get('pairs', 0)} selected_primary={twins.get('selected_primary', 0)} "
        f"selected_twin={twins.get('selected_twin', 0)} ({twins.get('selected_twin_pct', 0):.1f}%) "
        f"selected_good_<=4blob={twins.get('selected_good_4_or_fewer_blobs', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
