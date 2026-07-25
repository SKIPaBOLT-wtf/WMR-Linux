#!/usr/bin/env python3
"""test_replay_contract.py -- behavior tests for the offline_vio_replay invocation contract.

Both halves of the contract have silently mis-measured before: a pinned camera config with no
`ctrl_gain` replayed the gain-32 July captures at the gain-16 calibration point (half the
photometric scale), and a partial G2_REPLAY_* strip let G2_REPLAY_IMU_CAL_DIR leak in from the
caller's shell (34 vs 36 swap accepts on one cell, same binary). Both failures were silent, so
these tests assert the properties that make them impossible rather than the implementation.
Run with the conda env:
    ~/miniconda3/envs/g2vr/bin/python test_replay_contract.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from replay_contract import (
    CALIBRATION_CTRL_GAIN,
    PINNED_CAMS,
    PINNED_IMU_CAL_DIR,
    ROOT,
    cams_ctrl_gain,
    cams_for_capture,
    imu_cal_dir_for_capture,
    replay_env,
)

PASS = 0
FAIL = 0

#: Captures the pinned gates replay, and the gain each one's sensor actually ran at.
#: The May/June sessions predate the B3 gain change (DEFAULT_CTRL_GAIN 16); the July ones
#: record `ctrl_gain: 32` in their own provenance snapshot.
CAPTURE_GAINS = {
    "20260524-200416-headpose": 16,
    "20260526-175615-clean-session2": 16,
    "20260528-080421-xv-session1": 16,
    "20260706-183948-s3final-gain32-main": 32,
    "20260713-222356-dev1cal-async": 32,
    "20260723-112531-comprehensive-stack": 32,
}


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


def main() -> int:
    print("[1] the pinned config states its gain instead of leaving it to be inferred")
    pinned = json.loads(PINNED_CAMS.read_text())
    check("pinned config carries an explicit ctrl_gain",
          "ctrl_gain" in pinned, str(sorted(pinned)))
    check("pinned gain is the pre-B3 calibration point",
          pinned.get("ctrl_gain") == CALIBRATION_CTRL_GAIN, str(pinned.get("ctrl_gain")))

    print("[2] a capture's own provenance snapshot outranks the pinned fallback")
    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / "capture"
        (capture / "provenance").mkdir(parents=True)
        check("no snapshot -> the pinned config", cams_for_capture(capture) == PINNED_CAMS,
              str(cams_for_capture(capture)))
        check("no imu cal -> the pinned calibration",
              imu_cal_dir_for_capture(capture) == PINNED_IMU_CAL_DIR,
              str(imu_cal_dir_for_capture(capture)))
        snapshot = capture / "provenance" / "hmd-cameras.json"
        snapshot.write_text(json.dumps({"ctrl_gain": 32, "cameras": []}))
        check("snapshot present -> the snapshot", cams_for_capture(capture) == snapshot,
              str(cams_for_capture(capture)))
        check("snapshot gain is read verbatim", cams_ctrl_gain(snapshot) == 32,
              str(cams_ctrl_gain(snapshot)))
        (capture / "provenance" / "imu-cal-SERIAL.txt").write_text("1 0 0 0 0 0 0 1.0 0\n")
        check("imu cal present -> the capture's own",
              imu_cal_dir_for_capture(capture) == capture / "provenance",
              str(imu_cal_dir_for_capture(capture)))

        print("[3] an absent gain resolves to the tracker's own calibration-point fallback")
        legacy = Path(tmp) / "legacy.json"
        legacy.write_text(json.dumps({"cameras": []}))
        check("absent key -> the calibration point",
              cams_ctrl_gain(legacy) == CALIBRATION_CTRL_GAIN, str(cams_ctrl_gain(legacy)))
        zero = Path(tmp) / "zero.json"
        zero.write_text(json.dumps({"ctrl_gain": 0, "cameras": []}))
        check("explicit 0 ('unknown') -> the calibration point",
              cams_ctrl_gain(zero) == CALIBRATION_CTRL_GAIN, str(cams_ctrl_gain(zero)))

    print("[4] every real capture resolves to the gain its sensor actually ran at")
    for name, gain in CAPTURE_GAINS.items():
        capture = ROOT / "captures" / name
        if not capture.is_dir():
            check(f"{name} present", False, "capture missing")
            continue
        resolved = cams_ctrl_gain(cams_for_capture(capture))
        check(f"{name} replays at gain {gain}", resolved == gain, f"resolved {resolved}")

    print("[5] the replay environment is built, never inherited")
    leaked = {
        "G2_REPLAY_IMU_CAL_DIR": "/nonexistent/ambient",
        "G2_REPLAY_DROP_OPTICAL_PERIOD_MS": "999",
        "G2_REPLAY_RENDER_HZ": "45",
        "G2_REPLAY_ANYTHING_FUTURE": "1",
    }
    saved = {k: os.environ.get(k) for k in leaked}
    os.environ.update(leaked)
    try:
        env = replay_env()
        check("no ambient G2_REPLAY_* survives except what the contract sets",
              set(k for k in env if k.startswith("G2_REPLAY_")) == {"G2_REPLAY_IMU_CAL_DIR"},
              str(sorted(k for k in env if k.startswith("G2_REPLAY_"))))
        check("the pinned calibration replaces the ambient one",
              env["G2_REPLAY_IMU_CAL_DIR"] == str(PINNED_IMU_CAL_DIR), env["G2_REPLAY_IMU_CAL_DIR"])
        check("non-replay environment is preserved", env.get("PATH") == os.environ.get("PATH"))

        env = replay_env({"G2_REPLAY_DROP_OPTICAL_DURATION_MS": "300"}, {"G2_REPLAY_TELEMETRY": "/t"})
        check("declared settings are applied",
              env["G2_REPLAY_DROP_OPTICAL_DURATION_MS"] == "300" and env["G2_REPLAY_TELEMETRY"] == "/t")
        check("a declared setting does not resurrect the leaked siblings",
              "G2_REPLAY_DROP_OPTICAL_PERIOD_MS" not in env)

        env = replay_env({"G2_REPLAY_TELEMETRY": "/a"}, {"G2_REPLAY_TELEMETRY": "/b"})
        check("later settings win over earlier ones", env["G2_REPLAY_TELEMETRY"] == "/b")

        with tempfile.TemporaryDirectory() as tmp:
            cal = Path(tmp)
            try:
                replay_env(imu_cal_dir=cal)
                check("a calibration-less directory is refused", False, "no exception raised")
            except FileNotFoundError as exc:
                check("a calibration-less directory is refused", str(cal) in str(exc), str(exc))
            (cal / "imu-cal-SERIAL.txt").write_text("1 0 0 0 0 0 0 1.0 0\n")
            check("an explicit calibration dir wins over the pinned default",
                  replay_env(imu_cal_dir=cal)["G2_REPLAY_IMU_CAL_DIR"] == str(cal))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
