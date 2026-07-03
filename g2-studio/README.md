# g2-studio

> Parts of this repo (including this README) are edited & maintained by Claude (AI assistant),
> presented as-is. I review what lands here, but verify load-bearing claims against the code.

This is my launcher and session manager for running an HP Reverb G2 on Linux (GNOME Wayland,
NVIDIA, SteamVR with a self-built Monado tracking driver). SteamVR doesn't know how to bring this
headset up on its own here, so this repo owns the whole session lifecycle: wake the panel, free the
DRM lease, dip the desktop for USB bandwidth, pin the platform for frame-time consistency, launch
SteamVR, and put everything back exactly as it was when the session ends.

It's built for my machine first, but everything is username-agnostic and auto-detecting (no
hardcoded connectors or paths beyond Steam's defaults), so it should reproduce on a similar
Ubuntu + NVIDIA setup without surgery.

## Quickstart

```
sudo scripts/setup-system.sh     # one-time: scoped sudoers grants, RT limits,
                                 #           cpupower wrapper repair (re-run after kernel upgrades)
scripts/setup-perfmon.sh         # one-time: venv for the frame-timing sidecar (no sudo)
# log out and back in once (RT-priority limit)

python3 -m core.steamvr start    # full VR session up
python3 -m core.steamvr stop     # session down, every system knob restored
```

## What a session applies (and reverts on stop — every exit path)

- GPU: persistence mode, P0 graphics-clock pin, memory-clock pin (kills pstate-excursion
  frame spikes), power limit.
- CPU: performance governor + a deep C-state ceiling (`cpupower idle-set -D 200`) — between
  11 ms frames the cores otherwise drop into C6/C8/C10 and every 1 kHz pose tick, USB interrupt,
  and compositor wake pays a 220–680 µs exit tax. Both are verified by sysfs readback, because
  Ubuntu's cpupower wrapper exits 0 without acting on custom kernels (setup-system.sh repairs it).
- vrserver gets `cap_sys_nice` re-applied each launch (Steam updates wipe it); the limits.d
  rule is the update-proof fallback.
- Desktop display dip for headset USB bandwidth; SteamVR supersample pinned for run-to-run
  determinism; render-target scale settable via `XRT_COMPOSITOR_SCALE_PERCENTAGE`.

## Instrumentation (render-program Phase 0)

Every session records to `var/perf/<timestamp>/` (symlink `var/perf/latest`), zero render-path
perturbation:

- `frametiming.csv` — per-frame App/Compositor CPU+GPU ms, drops, mispresents, pose timestamps,
  via a passive OpenVR background sidecar; `clock.json` bridges its clock to wall time.
- `dmon.csv` / `pmon.csv` — 1 Hz GPU power/clocks/throttle-violations and per-process SM%
  (desktop contention evidence).
- `report.txt` — written automatically at session stop by `core/perfreport.py`: frame-time
  distributions plus every spike burst attributed to power-cap / thermal / memclk excursion /
  core-clock dip / desktop contention / CPU present-wait.
- `scripts/trace-vr-spikes.sh [seconds]` — on-demand kernel trace (sched_switch, dma_fence,
  drm events) into the session dir, with a per-CRTC vblank-cadence summary, for whatever the
  report can't attribute.

Deeper analysis (motion-to-photon decomposition, tracking telemetry) lives in my research tree's
`tools/telemetry/` — see `m2p_decompose.py` there, which joins a capture's telemetry with a
`--perf-session` dir from here.

## Layout

- `core/` — session lifecycle (`steamvr.py`), GPU/CPU pinning (`gpu.py`), display management
  (`display.py`, `vr_display.py`, `mutter.py`), Monado service handling (`monado.py`),
  instrumentation (`perfmon.py`, `frametiming.py`, `perfreport.py`).
- `scripts/` — one-time setup + on-demand diagnostics.
- `system/` — files installed into the system by `setup-system.sh`.
- `var/` — runtime state and per-session performance data (gitignored).
