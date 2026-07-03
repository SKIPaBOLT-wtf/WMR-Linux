#!/usr/bin/env python3
"""N6 metric: max-consecutive-lost-vs-reference between two replay out/dev CSVs.

Given a REFERENCE dev CSV and a CANDIDATE dev CSV from the same capture replay
(offline_vio_replay out dirs), aligns rows on t_ns and reports, over frames where the
reference accepted optical (opt_valid=1), the longest consecutive run the candidate did
not accept — the "how long can the candidate stay lost while the reference tracks"
number the felt layer cares about — plus total lost/gained counts both ways.

Worked example (clutter capture 20260612-153651 band-2, dev1):
  reference = pre-H5 replay (f7e7675ba), candidate = H5 replay (90f7ef62f)
  -> rows 4029 | accepts ref 3247 cand 3098 | lost-vs-ref 176 gained-vs-ref 27
     max consecutive LOST: 80 frames (3.506 s at t+42.218..45.725)   <- the dev1 clutter
     lock-out this metric exists to catch; reverse direction max run 5.

Usage:
  accept_runs.py <reference_dev.csv> <candidate_dev.csv>
"""
from __future__ import annotations

import argparse
import csv
import sys


def load_accepts(path: str) -> dict[int, bool]:
    with open(path) as f:
        return {int(r["t_ns"]): r["opt_valid"] == "1" for r in csv.DictReader(f)}


def max_run(ts: list[int], lost: set[int]) -> tuple[int, int, int]:
    """Longest consecutive run (over the reference-accepted frame sequence `ts`) of frames
    in `lost`. Returns (length, start_ns, end_ns); (0, 0, 0) when empty."""
    best_len, best_start, best_end = 0, 0, 0
    run_len, run_start = 0, 0
    for t in ts:
        if t in lost:
            if run_len == 0:
                run_start = t
            run_len += 1
            if run_len > best_len:
                best_len, best_start, best_end = run_len, run_start, t
        else:
            run_len = 0
    return best_len, best_start, best_end


def compare(ref_path: str, cand_path: str) -> dict:
    ref = load_accepts(ref_path)
    cand = load_accepts(cand_path)
    common = sorted(set(ref) & set(cand))
    if not common:
        raise SystemExit("no common t_ns rows — are these replays of the same capture?")
    t0 = common[0]
    ref_acc = [t for t in common if ref[t]]
    cand_acc = [t for t in common if cand[t]]
    lost = {t for t in ref_acc if not cand[t]}
    gained = {t for t in cand_acc if not ref[t]}
    n_lost, s_lost, e_lost = max_run(ref_acc, lost)
    n_gained, s_gained, e_gained = max_run(cand_acc, gained)
    return dict(
        rows=len(common),
        ref_accepts=len(ref_acc),
        cand_accepts=len(cand_acc),
        lost_vs_ref=len(lost),
        gained_vs_ref=len(gained),
        max_consec_lost=n_lost,
        max_consec_lost_span_s=(e_lost - s_lost) / 1e9 if n_lost else 0.0,
        max_consec_lost_at_s=((s_lost - t0) / 1e9, (e_lost - t0) / 1e9) if n_lost else None,
        max_consec_gained=n_gained,
        max_consec_gained_at_s=((s_gained - t0) / 1e9, (e_gained - t0) / 1e9) if n_gained else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference", help="reference out/devN.csv")
    ap.add_argument("candidate", help="candidate out/devN.csv (same capture, same device)")
    args = ap.parse_args()
    r = compare(args.reference, args.candidate)
    print(f"rows {r['rows']} | accepts ref {r['ref_accepts']} cand {r['cand_accepts']} | "
          f"lost-vs-ref {r['lost_vs_ref']} gained-vs-ref {r['gained_vs_ref']}")
    if r["max_consec_lost"]:
        a, b = r["max_consec_lost_at_s"]
        print(f"max consecutive LOST: {r['max_consec_lost']} frames "
              f"({r['max_consec_lost_span_s']:.3f} s at t+{a:.3f}..{b:.3f})")
    else:
        print("max consecutive LOST: 0")
    if r["max_consec_gained"]:
        a, b = r["max_consec_gained_at_s"]
        print(f"max consecutive GAINED: {r['max_consec_gained']} frames (at t+{a:.3f}..{b:.3f})")
    else:
        print("max consecutive GAINED: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
