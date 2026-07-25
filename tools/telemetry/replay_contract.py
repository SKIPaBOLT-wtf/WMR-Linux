#!/usr/bin/env python3
"""How offline_vio_replay must be invoked for a replay to measure the captured system.

Two inputs decide that, and both used to be decided independently by each runner:

  * The CAMERA CONFIG carries `ctrl_gain`, the commanded controller-slot analog gain the
    tracker denominates every DN constant in (blobwatch_gain_multiplier: m = gain/16, and a
    missing/zero key means "pre-gain-plumbing" -> the gain-16 calibration point, m = 1.0).
    B3 moved production to gain 32 (m = 2.0), so the three July 2026 captures recorded
    `ctrl_gain: 32` while the pinned pre-B3 config carried no gain at all — replaying them
    against it silently halves the photometric scale. A capture's own provenance snapshot is
    immutable and capture-era by construction, so it wins whenever it exists; the pinned
    config is the fallback for the captures taken before provenance snapshots existed, and it
    names its gain EXPLICITLY so an absent key can only ever mean a genuinely pre-plumbing
    snapshot. What the pinning doctrine forbids is the LIVE-MUTABLE ~/.config copy the driver
    rewrites at session start (2026-06-11: a mid-evening blob_detect_threshold flip shifted
    every subsequent replay) — never the capture's own frozen copy.

  * The G2_REPLAY_* ENVIRONMENT is the harness's entire configuration surface: forced
    dropouts, LED masks, telemetry sink, IMU calibration. Inheriting any of it from the
    caller's shell makes a run depend on an invisible input — G2_REPLAY_IMU_CAL_DIR present
    vs absent is worth 34 vs 36 swap accepts on one matrix cell at 708388d75, on the same
    binary. `replay_env` therefore builds the environment from scratch: every G2_REPLAY_* the
    caller happened to export is dropped, and only what the run declares survives.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Camera config for captures taken before the driver started snapshotting provenance.
#: Byte-equivalent to the 20260612 provenance snapshot, plus an explicit pre-B3 `ctrl_gain`.
PINNED_CAMS = Path(__file__).resolve().parent / "data/hmd-cameras-replay.json"

#: Persisted per-serial IMU bias/scale calibration the harness seeds from. The live driver
#: always has these caches, so replaying WITH them is the faithful configuration; every pinned
#: matrix floor was produced against this directory.
PINNED_IMU_CAL_DIR = ROOT / "captures/20260723-112531-comprehensive-stack/provenance"

#: Commanded gain the tracker falls back to when a config carries no `ctrl_gain` — the
#: calibration operating point every DN-denominated constant was measured at
#: (blobwatch.h: blobwatch_gain_multiplier maps 0 -> 16 -> m = 1.0).
CALIBRATION_CTRL_GAIN = 16

_REPLAY_ENV_PREFIX = "G2_REPLAY_"


def cams_for_capture(capture: Path | str) -> Path:
    """The camera config that describes the system which recorded `capture`."""
    snapshot = Path(capture) / "provenance" / "hmd-cameras.json"
    return snapshot if snapshot.is_file() else PINNED_CAMS


def imu_cal_dir_for_capture(capture: Path | str) -> Path:
    """The IMU calibration caches the harness should seed from for `capture`."""
    snapshot = Path(capture) / "provenance"
    return snapshot if any(snapshot.glob("imu-cal-*.txt")) else PINNED_IMU_CAL_DIR


def cams_ctrl_gain(cams_json: Path | str) -> int:
    """The commanded controller-slot gain a replay against `cams_json` will run at.

    Resolves the tracker's own absent-key fallback, so a run can RECORD the gain it actually
    replayed at instead of leaving "no key" and "gain 16" indistinguishable in provenance.
    """
    gain = int(json.loads(Path(cams_json).read_text()).get("ctrl_gain", 0) or 0)
    return gain if gain else CALIBRATION_CTRL_GAIN


def replay_env(*settings: Mapping[str, str], imu_cal_dir: Path | str = PINNED_IMU_CAL_DIR) -> dict[str, str]:
    """The environment for one harness invocation: ambient G2_REPLAY_* dropped, then `settings`.

    Later mappings win over earlier ones. Raises when `imu_cal_dir` holds no calibration —
    the harness treats an unreadable directory as "no calibration at all" and replays a
    different system, which is a measurement failure, not a degraded mode.
    """
    cal_dir = Path(imu_cal_dir)
    if not any(cal_dir.glob("imu-cal-*.txt")):
        raise FileNotFoundError(
            f"no imu-cal-*.txt in {cal_dir}: the harness would replay with NO IMU calibration, "
            f"which is a different system from every pinned reference (default: {PINNED_IMU_CAL_DIR})"
        )
    env = {key: value for key, value in os.environ.items() if not key.startswith(_REPLAY_ENV_PREFIX)}
    env["G2_REPLAY_IMU_CAL_DIR"] = str(cal_dir)
    for block in settings:
        env.update({key: str(value) for key, value in block.items()})
    return env
