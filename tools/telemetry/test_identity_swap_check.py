#!/usr/bin/env python3
"""test_identity_swap_check.py -- behavior tests for the standing identity-swap gate.

The gate's value is entirely in what it refuses to do quietly: pass because an input went missing,
pass because a swap fell a frame outside a window, or fail because one device alone lost its track.
So these assert the fixture contract (a window is a frozen property of a capture, not a knob), the
transposition signature, and that every missing input comes out as a LOUD SKIP (77) rather than a
green gate. Run with the conda env:
    ~/miniconda3/envs/g2vr/bin/python test_identity_swap_check.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from identity_swap_check import (
    REGRESSION_FIXTURES,
    episodes,
    mutual_swap_frames,
    run_fixture_replay,
)

PASS = 0
FAIL = 0

#: The class each fixture exists to gate, and the arm that proved the window is live.
EXPECTED_WINDOWS = {
    "xv1-blackout": ((60.9, 62.1),),
    "jul23-hands-close": ((41.9, 45.0), (119.3, 120.4)),
}


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


def swap(t_rel_s: float):
    """A SWAP row as classify_device emits it, reduced to what the signature reads."""
    return {"t_rel_s": t_rel_s, "t_hw_ns": int(t_rel_s * 1e9)}


def main() -> int:
    print("[1] the fixture set is a frozen contract")
    names = [f["name"] for f in REGRESSION_FIXTURES]
    check("fixture names are unique", len(set(names)) == len(names), str(names))
    check("both documented classes are pinned", set(names) == set(EXPECTED_WINDOWS), str(names))
    for fixture in REGRESSION_FIXTURES:
        name = fixture["name"]
        check(f"{name}: windows are the frozen ones",
              tuple(fixture["windows"]) == EXPECTED_WINDOWS[name], str(fixture["windows"]))
        check(f"{name}: every window is a forward interval",
              all(lo < hi for lo, hi in fixture["windows"]), str(fixture["windows"]))
        check(f"{name}: the replay regime is harness settings only",
              all(key.startswith("G2_REPLAY_") for key in fixture["drop_env"]),
              str(sorted(fixture["drop_env"])))
        check(f"{name}: the capture is named, not discovered",
              Path(fixture["capture"]).is_absolute(), str(fixture["capture"]))

    print("\n[2] the two-way transposition signature")
    recs = {1: [swap(41.9), swap(42.5)], 2: [swap(41.9), swap(43.9)]}
    check("simultaneous accepts on each other's reference are mutual",
          mutual_swap_frames(recs, (41.9, 45.0)) == [41.9])
    check("a one-sided steal is not a transposition",
          mutual_swap_frames({1: [swap(42.5)], 2: []}, (41.9, 45.0)) == [])
    check("a mutual pair outside the window does not count",
          mutual_swap_frames({1: [swap(50.0)], 2: [swap(50.0)]}, (41.9, 45.0)) == [])
    check("accepts further apart than the frame tolerance are not simultaneous",
          mutual_swap_frames({1: [swap(42.0)], 2: [swap(42.5)]}, (41.9, 45.0)) == [])
    check("the window edges are inclusive",
          mutual_swap_frames({1: [swap(45.0)], 2: [swap(45.0)]}, (41.9, 45.0)) == [45.0])

    print("\n[3] swap accepts group into episodes by their own gaps")
    eps = episodes([swap(41.9), swap(41.95), swap(60.0)])
    check("a run and a distant accept are separate episodes", len(eps) == 2, str(eps))
    check("an episode reports its extent and accept count",
          eps[0] == {"start_s": 41.9, "end_s": 41.95, "n": 2}, str(eps[0]))

    print("\n[4] a missing input is a LOUD SKIP, never a green gate")
    fixture = dict(REGRESSION_FIXTURES[0])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        real_bin = tmp / "harness"
        real_bin.write_text("")
        left, right = tmp / "left.json", tmp / "right.json"
        left.write_text("{}")
        right.write_text("{}")
        check("absent harness binary skips",
              run_fixture_replay(tmp / "nope", tmp / "out", fixture, str(left), str(right)) == 77)
        check("absent capture skips",
              run_fixture_replay(real_bin, tmp / "out", {**fixture, "capture": tmp / "nope"},
                                 str(left), str(right)) == 77)
        check("absent controller json skips",
              run_fixture_replay(real_bin, tmp / "out", fixture, str(tmp / "nope"),
                                 str(right)) == 77)
        empty = tmp / "capture-without-frames"
        (empty / "telemetry").mkdir(parents=True)
        check("a capture with no replayable frame set skips",
              run_fixture_replay(real_bin, tmp / "out", {**fixture, "capture": empty},
                                 str(left), str(right)) == 77)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
