"""OpenVR FrameTiming sidecar (render-program §4.1).

A passive VRApplication_Background app: polls IVRCompositor.GetFrameTimings in
batches (shared-memory reads — zero render-path perturbation), dedupes on frame
index and appends one CSV row per compositor frame: App/Compositor CPU+GPU ms,
present/mispresent/drop counts, reprojection flags, WaitGetPoses/new-pose
timestamps. This is the per-frame p50/p95/p99 instrument the async-less present
path lives and dies on — session-exit summaries only give averages.

Waits for SteamVR to come up, self-exits when it quits. Launched detached by
core.perfmon with the var/venv interpreter (which carries the openvr binding):
    var/venv/bin/python -m core.frametiming <out.csv>
"""
import ctypes
import json
import signal
import sys
import time
from pathlib import Path

import openvr

BATCH = 128            # frames per GetFrameTimings call (compositor history cap)
POLL_S = 0.5           # 128 frames @ 90 Hz = 1.42 s of history; 0.5 s never misses
STARTUP_TIMEOUT_S = 180  # Steam cold start budget

FIELDS = [name for name, _ in openvr.Compositor_FrameTiming._fields_
          if name not in ("m_nSize", "m_HmdPose")]


def _wait_for_steamvr():
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while True:
        try:
            openvr.init(openvr.VRApplication_Background)
            return True
        except openvr.error_code.InitError:
            if time.monotonic() > deadline:
                return False
            time.sleep(2)


def main(out_path):
    for sig in (signal.SIGTERM, signal.SIGINT):   # perfmon.stop sends SIGINT
        signal.signal(sig, lambda *_: sys.exit(0))
    if not _wait_for_steamvr():
        print(f"frametiming: no SteamVR within {STARTUP_TIMEOUT_S}s, giving up",
              flush=True)
        return 1
    system = openvr.VRSystem()
    compositor = openvr.VRCompositor()
    batch = (openvr.Compositor_FrameTiming * BATCH)()
    event = openvr.VREvent_t()
    last_index = -1
    rows = 0
    # Wall-clock bridge for joining frame rows to dmon/pmon (both wall-stamped):
    # each poll, wall_now - newest_frame_system_time over-estimates the true
    # offset by the frame's age at poll time (≤ POLL_S + one frame), so the
    # running minimum converges to the offset within ~one frame. Persisted on
    # every improvement so a killed sidecar still leaves a usable bridge.
    clock_path = Path(out_path).with_name("clock.json")
    wall_minus_vrsys = None
    print(f"frametiming: connected, logging to {out_path}", flush=True)
    with open(out_path, "w") as out:
        out.write(",".join(FIELDS) + "\n")
        try:
            while True:
                while system.pollNextEvent(event):
                    if event.eventType == openvr.VREvent_Quit:
                        system.acknowledgeQuit_Exiting()
                        raise SystemExit(0)
                batch[0].m_nSize = ctypes.sizeof(openvr.Compositor_FrameTiming)
                n, _ = compositor.getFrameTimings(batch)
                for t in batch[:n]:      # ascending, oldest -> newest
                    if t.m_nFrameIndex <= last_index:
                        continue
                    last_index = t.m_nFrameIndex
                    out.write(",".join(
                        f"{getattr(t, f):.4f}" if f.startswith("m_fl")
                        else str(getattr(t, f)) for f in FIELDS) + "\n")
                    rows += 1
                if n:
                    cand = time.time() - batch[n - 1].m_flSystemTimeInSeconds
                    if wall_minus_vrsys is None or cand < wall_minus_vrsys:
                        wall_minus_vrsys = cand
                        clock_path.write_text(json.dumps(
                            {"wall_minus_vrsys_s": wall_minus_vrsys}) + "\n")
                out.flush()
                time.sleep(POLL_S)
        except openvr.error_code.OpenVRError:
            pass                          # SteamVR went away mid-poll
        finally:
            print(f"frametiming: done, {rows} frames", flush=True)
            openvr.shutdown()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m core.frametiming <out.csv>")
    sys.exit(main(sys.argv[1]))
