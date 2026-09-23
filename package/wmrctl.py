#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Read-only WMR Linux development tools. No install or telemetry side effects."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

HASH = re.compile(r"[0-9a-f]{64}\Z")
ENV_KEYS = {"WMR_SLAM", "WMR_MAX_SLAM_CAMS", "WMR_AUTOEXPOSURE", "WMR_HANDTRACKING",
            "WMR_CLOCK_WINDOWED", "SLAM_SUBMIT_FROM_START", "G2_REQUIRE_VISUAL_OBSERVATIONS",
            "G2_PREDICT_WITH_VIT_BIAS"}
STATES = {"untested", "development-reference-not-tracking-accepted"}
REFERENCE_MOLD_VERSION = "2.40.4"


def digest(path):
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_profile(data):
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Expected profile schema_version 1")
    allowed = {"schema_version", "id", "qualification", "headset", "platform", "gpu", "runtime", "environment", "unset", "automatic_system_changes"}
    if set(data) != allowed:
        raise ValueError("Missing or unexpected top-level profile fields")
    if not isinstance(data["id"], str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", data["id"]):
        raise ValueError("Profile id must be a lowercase public identifier")
    if not isinstance(data["qualification"], str) or data["qualification"] not in STATES:
        raise ValueError("This tooling has no qualified release-manifest support yet")
    if data["automatic_system_changes"] is not False:
        raise ValueError("Automatic system changes are not supported")
    shapes = {
        "headset": {"family", "calibration", "panel_mode", "slam_cameras"},
        "platform": {"os", "version", "session", "architecture"},
        "gpu": {"vendor", "reference_model", "reference_driver", "workarounds"},
        "runtime": {"frontend", "backend", "backend_source", "backend_sha256"},
    }
    for key, fields in shapes.items():
        if not isinstance(data[key], dict) or set(data[key]) != fields:
            raise ValueError(f"Unexpected {key} fields; do not add private device data")
    if not isinstance(data["headset"]["family"], str) or not data["headset"]["family"].strip():
        raise ValueError("Headset family must be a nonempty public model name")
    if data["headset"]["calibration"] != "read-own-device-factory":
        raise ValueError("Profiles must read the attached device's own calibration")
    mode = data["headset"]["panel_mode"]
    if not isinstance(mode, list) or len(mode) != 3 or any(type(v) is not int or v <= 0 for v in mode):
        raise ValueError("panel_mode must be [width, height, refresh_hz] positive integers")
    cams = data["headset"]["slam_cameras"]
    if type(cams) is not int or not 1 <= cams <= 4:
        raise ValueError("Invalid camera count")
    for section in ("platform", "gpu", "runtime"):
        if any(not isinstance(v, str) or not v for v in data[section].values()):
            raise ValueError(f"{section} values must be nonempty strings")
    if not HASH.fullmatch(data["runtime"]["backend_sha256"]):
        raise ValueError("backend_sha256 must be an exact SHA-256")
    if not re.fullmatch(r"[0-9a-f]{40}", data["runtime"]["backend_source"]):
        raise ValueError("backend_source must be a pinned commit")
    env = data["environment"]
    if not isinstance(env, dict) or set(env) != ENV_KEYS:
        raise ValueError("Unexpected or missing environment fields")
    if any(not isinstance(v, str) or v not in {"true", "false"}
           for k, v in env.items() if k != "WMR_MAX_SLAM_CAMS"):
        raise ValueError("Boolean policy values must be true or false strings")
    if env["WMR_MAX_SLAM_CAMS"] != str(cams):
        raise ValueError("Camera policy disagrees with headset profile")
    if env["G2_PREDICT_WITH_VIT_BIAS"] != "false":
        raise ValueError("Learned-bias backend is not qualified in this schema")
    if data["unset"] != ["WMR_HT1_EXTRINSICS_OVERRIDE", "SLAM_CONFIG"]:
        raise ValueError("Unqualified calibration/config overrides must remain unset")
    return {"valid": True, "profile": data["id"], "qualification": data["qualification"], "applied": False}


def doctor():
    try:
        osinfo = platform.freedesktop_os_release()
    except OSError:
        osinfo = {}
    report = {"tool_schema": 1, "read_only": True, "uploads": False,
              "os": {k: osinfo[k] for k in ("ID", "VERSION_ID") if k in osinfo},
              "kernel": platform.release(), "architecture": platform.machine(),
              "gpu": None, "runtime_libraries": {}}
    # Fixed allowlist: deliberately no serial, UUID, account, environment or process dump.
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            proc = subprocess.run([executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                  text=True, capture_output=True, timeout=5, check=False)
            if proc.returncode == 0:
                report["gpu"] = proc.stdout.strip().splitlines()
            else:
                report["gpu"] = "NVIDIA query unavailable"
        except (OSError, subprocess.TimeoutExpired):
            report["gpu"] = "NVIDIA query unavailable"
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "steamvr-monado/bin/linux64"
    for name in ("driver_monado.so", "libbasalt.so"):
        try:
            report["runtime_libraries"][name] = {"sha256": digest(base / name)}
        except OSError:
            report["runtime_libraries"][name] = {"sha256": None, "unreadable": True}
    report["tracking_quality"] = "not measured; library presence does not establish accuracy"
    return report


def basalt_linker(library):
    """Gate a staged reference backend on its recorded linker identity."""
    if not library.is_file():
        raise ValueError("Basalt artifact unavailable")
    result = subprocess.run(["readelf", "-p", ".comment", str(library)],
                            text=True, capture_output=True, timeout=10, check=False)
    if result.returncode:
        raise ValueError("Basalt ELF comment unavailable")
    versions = set(re.findall(r"\bmold\s+(\d+(?:\.\d+)+)\b", result.stdout))
    return {"ok": versions == {REFERENCE_MOLD_VERSION}, "sha256": digest(library),
            "linker": "mold" if versions else "unverified",
            "linker_version": next(iter(versions)) if len(versions) == 1 else None,
            "required_linker_version": REFERENCE_MOLD_VERSION,
            "scope": "linker metadata only; matching replay and physical tracking remain separate gates"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="print an allowlisted read-only report; review before sharing")
    profile = sub.add_parser("profile", help="validate a profile without applying it")
    profile.add_argument("file", type=Path)
    linker = sub.add_parser("basalt-linker", help="check staged reference Basalt ELF linker metadata")
    linker.add_argument("library", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor()
        elif args.command == "profile":
            report = validate_profile(json.loads(args.file.read_text()))
        else:
            report = basalt_linker(args.library)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, subprocess.TimeoutExpired) as error:
        # Do not print paths/content from malformed private files.
        print(json.dumps({"ok": False, "error_type": type(error).__name__, "message": "Validation could not be completed"}))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 3


if __name__ == "__main__":
    sys.exit(main())
