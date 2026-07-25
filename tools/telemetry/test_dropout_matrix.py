#!/usr/bin/env python3
"""test_dropout_matrix.py -- behavior tests for the dropout-matrix objective's schema policing.

The shrinkage estimator (shrunk_pct) leaves a raw percentage untouched when its denominator
count is missing, and evidence_weight ramps a conditional term to zero on a missing count --
so a scorer schema that silently drops the counts would silently disengage shrinkage while
still printing green. REQUIRED_ROW_KEYS polices the counts like the metrics they weight; a
count-stripped row must hard-fail, never score. Run with the conda env:
    ~/miniconda3/envs/g2vr/bin/python test_dropout_matrix.py
"""
from __future__ import annotations

import sys

from dropout_matrix import (
    OBJECTIVE_SCORE_TOLERANCE,
    ESTIMATE_FULL_N,
    REQUIRED_IDENTITY_KEYS,
    REQUIRED_ROW_KEYS,
    evaluate_gate,
    objective_for_row,
    shrunk_pct,
    summarize_objective,
)

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


def complete_row():
    """A minimal schema-complete row for a no-drop regime (all counts present)."""
    row = {
        "capture": "synthetic",
        "regime": "normal",
        "device": 1,
        "drop_compliance": "unverifiable",
        # metrics
        "total_frame_groups": 1000,
        "scoreable_frames": 900,
        "scoreable_pct_total": 90.0,
        "all_scoreable_position_tracked_correct_pct": 97.0,
        "all_scoreable_reported_position_correct_pct": 96.0,
        "all_scoreable_position_tracked_accuracy_pct": 99.0,
        "pred_recall": 0.96,
        "pred_position_recall": 0.97,
        "pred_position_precision": 0.99,
        "pred_position_rmse_cm": 0.8,
        "pred_position_p95_cm": 2.0,
        "all_pos_p95_cm": 5.0,
        "wrong_branch_pct_scored": 0.1,
        # counts (the R4#18 additions)
        "all_scoreable_frames": 900,
        "all_scoreable_position_tracked": 850,
        "pred_scored_frames": 800,
        "pred_tp": 770,
        "pred_fn": 30,
        "pred_position_tp": 780,
        "pred_position_fn": 20,
        "pred_position_fp": 8,
        "forced_drop_visible_frames": 0,
        "forced_drop_visible_reported": 0,
        "forced_drop_visible_position_tracked": 0,
        "stale_visible_frames": 0,
        "stale_position_tracked": 0,
        # identity gate
        "identity_other_report_explains_ref": 0,
        "identity_strict_two_way_swap_frames": 0,
    }
    missing = [k for k in REQUIRED_ROW_KEYS + REQUIRED_IDENTITY_KEYS if k not in row]
    assert not missing, f"test row out of date with REQUIRED keys: {missing}"
    return row


def main() -> int:
    print("[1] a schema-complete row scores cleanly")
    obj = objective_for_row(complete_row())
    check("no hard fail", obj["objective_hard_fail"] == "", obj["objective_hard_fail"])
    check("nonzero score", obj["objective_score"] > 0.0, str(obj["objective_score"]))

    print("[2] stripping any policed count key hard-fails the row (never a free pass)")
    for key in ("all_scoreable_frames", "all_scoreable_position_tracked", "pred_scored_frames",
                "pred_tp", "pred_fn", "pred_position_tp", "pred_position_fn", "pred_position_fp",
                "forced_drop_visible_frames", "forced_drop_visible_reported",
                "forced_drop_visible_position_tracked", "stale_visible_frames",
                "stale_position_tracked"):
        row = complete_row()
        row[key] = None
        obj = objective_for_row(row)
        check(f"stripped {key} -> schema_incomplete + score 0",
              key in obj["objective_hard_fail"] and "schema_incomplete" in obj["objective_hard_fail"]
              and obj["objective_score"] == 0.0,
              f"hard_fail={obj['objective_hard_fail']!r} score={obj['objective_score']}")

    print("[3] shrinkage engages against its own denominator (the class the counts protect)")
    check("small-n shortfall is shrunk toward the neutral point",
          shrunk_pct(80.0, 10, neutral=95.0) > 80.0, str(shrunk_pct(80.0, 10, neutral=95.0)))
    check("n >= ESTIMATE_FULL_N returns the raw value bit-identically",
          shrunk_pct(80.0, ESTIMATE_FULL_N, neutral=95.0) == 80.0)

    print("[4] the process gate rejects identity failures and valid-but-worse quality")
    good = complete_row()
    good.update(objective_for_row(good))
    objective = summarize_objective([good])
    baseline = {
        "schema_version": 1,
        "profiles": [{
            "name": "synthetic",
            "row_objective_minima": {"synthetic/normal/dev1": good["objective_score"]},
            "objective_geomean_min": objective["score_geomean"],
        }],
    }
    gate = evaluate_gate([good], objective, baseline)
    check("pinned row passes", gate["passed"], str(gate["failures"]))

    worse = complete_row()
    worse["pred_position_rmse_cm"] = 2.0
    worse.update(objective_for_row(worse))
    worse_gate = evaluate_gate([worse], summarize_objective([worse]), baseline)
    check("schema-valid but objectively worse row fails",
          not worse_gate["passed"] and any("below pinned" in f for f in worse_gate["failures"]),
          str(worse_gate["failures"]))

    inside = complete_row()
    inside.update(objective_for_row(inside))
    inside["objective_score"] = good["objective_score"] - OBJECTIVE_SCORE_TOLERANCE / 2.0
    inside_gate = evaluate_gate([inside], summarize_objective([good]), baseline)
    check("a sub-tolerance dip is not a regression", inside_gate["passed"], str(inside_gate["failures"]))

    outside = complete_row()
    outside.update(objective_for_row(outside))
    outside["objective_score"] = good["objective_score"] - OBJECTIVE_SCORE_TOLERANCE * 2.0
    outside_gate = evaluate_gate([outside], summarize_objective([good]), baseline)
    check("a dip past the tolerance still fails",
          not outside_gate["passed"] and any("below pinned" in f for f in outside_gate["failures"]),
          str(outside_gate["failures"]))

    identity = complete_row()
    identity["identity_other_report_explains_ref"] = 1
    identity.update(objective_for_row(identity))
    identity_gate = evaluate_gate([identity], summarize_objective([identity]), baseline)
    check("identity hard fail affects gate",
          not identity_gate["passed"]
          and any("objective hard fail" in f for f in identity_gate["failures"]),
          str(identity_gate["failures"]))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
