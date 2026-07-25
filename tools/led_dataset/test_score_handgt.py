#!/usr/bin/env python3
"""Behavior tests for hand-GT comparison identity."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("score_handgt.py")
    spec = importlib.util.spec_from_file_location("score_handgt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    score = load_module()
    splits = score.MF.split_paths("/tmp/pinned-handgt")
    assert splits == {
        "xv1": Path("/tmp/pinned-handgt/xv1"),
        "clean": Path("/tmp/pinned-handgt/clean2"),
        "headpose": Path("/tmp/pinned-handgt/headpose"),
    }
    reference = [
        {"split": "clean", "tag": "cam2_1", "dev": 1, "verdict": "GOOD"},
        {"split": "xv1", "tag": "cam0_2", "dev": None, "verdict": "ABSTAIN"},
    ]
    candidate = [
        {"split": "xv1", "tag": "cam0_2", "dev": None, "verdict": "ABSTAIN"},
        {"split": "clean", "tag": "cam2_1", "dev": 2, "verdict": "WRONG"},
    ]
    rows = score.compare_by_tag(reference, candidate)
    assert len(rows) == 2
    changed = rows[0]
    assert changed["verdict_changed"]
    assert changed["device_changed"]
    try:
        score.compare_by_tag(reference, candidate[:1])
    except ValueError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("missing physical tag did not fail")
    try:
        score.index_by_tag(reference + [reference[0]])
    except ValueError as error:
        assert "duplicate hand-GT identity" in str(error)
    else:
        raise AssertionError("duplicate physical tag did not fail")
    print("4 passed, 0 failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
