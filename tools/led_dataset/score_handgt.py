#!/usr/bin/env python3
"""Score a handgt replay battery with the pinned matcher_failure classifier.

Thin driver over tools/led_dataset/matcher_failure.py (the scorer is untouched): its SPLITS
mapping is pointed at the replay dirs under --root (root/<xv1|clean2|headpose>/telemetry must
hold each split's candidate.bin), then the module's own classify/report path runs verbatim.
Must run with cwd = tools/led_dataset (the GT pool paths are relative there).

Usage: score_handgt.py --root <battery>/handgt [--json out.json]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matcher_failure as MF  # noqa: E402


def index_by_tag(rows):
    """Index adjudications by physical hand-GT identity; device is an outcome, never identity."""
    indexed = {}
    for row in rows:
        key = (row["split"], row["tag"])
        if key in indexed:
            raise ValueError(f"duplicate hand-GT identity: {key[0]}/{key[1]}")
        indexed[key] = row
    return indexed


def compare_by_tag(reference, candidate):
    old = index_by_tag(reference)
    new = index_by_tag(candidate)
    if old.keys() != new.keys():
        missing = sorted(old.keys() - new.keys())
        added = sorted(new.keys() - old.keys())
        raise ValueError(f"hand-GT identity mismatch: missing={missing[:5]} added={added[:5]}")
    rows = []
    for split, tag in sorted(old):
        before = old[(split, tag)]
        after = new[(split, tag)]
        rows.append({
            "split": split,
            "tag": tag,
            "reference_verdict": before["verdict"],
            "candidate_verdict": after["verdict"],
            "verdict_changed": before["verdict"] != after["verdict"],
            "reference_device": before.get("dev"),
            "candidate_device": after.get("dev"),
            "device_changed": before.get("dev") != after.get("dev"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding <split>/telemetry replay outputs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", help="prior matcher_failure.json to compare by (split, tag)")
    ap.add_argument("--comparison-csv", help="write all tag-level comparison rows")
    args = ap.parse_args()
    if bool(args.compare) != bool(args.comparison_csv):
        ap.error("--compare and --comparison-csv must be supplied together")
    root = Path(args.root).resolve()
    splits = MF.split_paths(root)
    cams = MF.g2cam.load_cams(MF.CAMS_JSON)
    models = {d: MF.g2cam.load_led_model(Path(p)) for d, p in MF.CTRL.items()}
    allrows = []
    for split, replay in splits.items():
        fb = MF.frame_blob_map([Path("dataset/pool") / split])
        rows = MF.classify_split(split, replay, cams, models, fb)
        allrows += rows
        MF.report(split, rows)
    print("\n" + "=" * 78)
    MF.report("ALL", allrows, grand=True)
    if args.json:
        json.dump(allrows, open(args.json, "w"), indent=1)
        print(f"\nwrote {len(allrows)} classified GT frames -> {args.json}")
    if args.compare:
        comparison = compare_by_tag(json.load(open(args.compare)), allrows)
        with open(args.comparison_csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
            writer.writeheader()
            writer.writerows(comparison)
        print(
            f"compared {len(comparison)} unique (split, tag) identities: "
            f"{sum(row['verdict_changed'] for row in comparison)} verdict changes, "
            f"{sum(row['device_changed'] for row in comparison)} device changes -> "
            f"{args.comparison_csv}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
