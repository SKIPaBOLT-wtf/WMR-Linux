"""Per-session render-performance instrumentation (render-program Phase 0).

Three passive observer channels, started with the SteamVR session and stopped
from restore() on every exit path:
  frametiming.csv  per-frame Compositor_FrameTiming via the OpenVR background
                   sidecar (core.frametiming) — App/Comp CPU+GPU ms, presents,
                   mispresents, drops, pose timestamps (tail distributions)
  dmon.csv         nvidia-smi dmon -s pucvt: power/util/clocks/violations/PCIe
                   (power-cap and memclk-excursion evidence)
  pmon.csv         nvidia-smi pmon -s um: per-process SM%/mem incl. gnome-shell
                   (desktop GPU-contention evidence)

Everything lands under var/perf/<YYYYmmdd-HHMMSS>/ (var/perf/latest symlinks
the newest session); channel PIDs live in var/perf/current.json so a crashed
launcher can still be reaped by the next start(). Nothing here touches the
render path.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "var" / "perf"
STATE = ROOT / "current.json"
VENV_PY = REPO / "var" / "venv" / "bin" / "python"


def _spawn(name, cmd, sess, pids, env=None):
    log = open(sess / f"{name}.log", "w")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                         stdout=log, stderr=subprocess.STDOUT,
                         start_new_session=True)
    log.close()
    pids[name] = {"pid": p.pid, "match": cmd[0].rsplit("/", 1)[-1]}


def start(env=None):
    """Start the session channels; returns the session dir (Path)."""
    if STATE.exists():
        stop()                                   # reap a crashed session's channels
    sess = ROOT / time.strftime("%Y%m%d-%H%M%S")
    sess.mkdir(parents=True)
    pids = {}
    _spawn("dmon", ["nvidia-smi", "dmon", "-s", "pucvt", "-o", "DT",
                    "-f", str(sess / "dmon.csv")], sess, pids)
    _spawn("pmon", ["nvidia-smi", "pmon", "-s", "um", "-o", "DT",
                    "-f", str(sess / "pmon.csv")], sess, pids)
    if VENV_PY.exists():
        _spawn("frametiming", [str(VENV_PY), "-m", "core.frametiming",
                               str(sess / "frametiming.csv")], sess, pids, env=env)
    else:
        print("[perfmon] frametiming sidecar skipped: var/venv missing — "
              "run scripts/setup-perfmon.sh once")
    meta = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "xrt_scale_pct": (env or os.environ).get("XRT_COMPOSITOR_SCALE_PERCENTAGE"),
            "channels": sorted(pids)}
    (sess / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    json.dump({"dir": str(sess), "pids": pids}, open(STATE, "w"))
    latest = ROOT / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(sess.name)
    return sess


def _alive(pid, match):
    """True if `pid` is still the process we spawned (guards PID reuse)."""
    try:
        with open(f"/proc/{pid}/cmdline") as f:
            return match in f.read()
    except OSError:
        return False


def stop():
    """Stop all channels of the current session; returns its dir or None.
    Idempotent — a missing/stale state file is a no-op."""
    if not STATE.exists():
        return None
    st = json.load(open(STATE))
    for name, rec in st["pids"].items():
        pid, match = rec["pid"], rec["match"]
        if not _alive(pid, match):
            continue
        # SIGINT first: nvidia-smi flushes its -f file on ^C; the sidecar exits
        # cleanly on SIGTERM and SIGINT alike.
        os.kill(pid, signal.SIGINT)
        for _ in range(30):
            if not _alive(pid, match):
                break
            time.sleep(0.1)
        else:
            os.kill(pid, signal.SIGKILL)
    STATE.unlink()
    _write_report(Path(st["dir"]))
    return st["dir"]


def _write_report(sess):
    """Spike-attribution report at teardown; must never break restore()."""
    try:
        from core import perfreport
        (sess / "report.txt").write_text(perfreport.report(sess) + "\n")
        print(f"[perfmon] report: {sess / 'report.txt'}")
    except Exception as e:  # noqa: BLE001 — teardown restores desktop state
        print(f"[perfmon] report generation failed: {e}")


if __name__ == "__main__":  # python3 -m core.perfmon [start|stop]
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    print({"start": start, "stop": stop}[cmd]())
