"""Launch SteamVR with the G2 on Wayland — the proper #77 fix.

SteamVR loads our Monado (ovrd_driver) in-process inside vrserver as the
DEVICE/TRACKING driver only: it creates the device system but passes NULL for the
compositor and exposes the HMD via IVRDisplayComponent (NOT DirectMode). So
SteamVR's own vrcompositor does the rendering and PRESENTS to the G2 — it acquires
the display itself (wp_drm_lease_device_v1 on Wayland). There is NO standalone
monado-service in this path, and Monado's own compositor is not used.

So vrcompositor must be able to lease the G2: it needs the lease FREE (no
standalone monado-service holding it) AND the desktop dipped for display
ISO-bandwidth (nothing in this path runs our negotiator), otherwise vrcompositor's
"WaitForPresent" watchdog times out and aborts. (The G2 present is carried by the
NVIDIA kernel patches: non_desktop advertise + native-mode-preferred + flip-bridge.)

So SteamVR mode is: free the G2, dip + hold the desktop (re-applying the dip once
the lease lands, since the G2 hotplug makes mutter reconfigure and revert it), pin
GPU/CPU perf, launch SteamVR; restore the desktop + clocks on exit. Mirrors
monado.start()/stop() for the OpenXR path.
"""
import json
import os
import shutil
import subprocess
import time

from . import gpu, mutter, perfmon, vr_display

APPID = "250820"  # SteamVR
SVR_PROCS = ("vrserver", "vrcompositor", "vrmonitor", "vrdashboard",
             "vrwebhelper", "steamtours")

# Render-target scale knob (render-program L1): the driver recommends
# panel x this percentage to SteamVR. 140 is the WMR-correct distortion
# supersample and the driver's own default — exporting it here is a no-op until
# swept. With SteamVR's supersample pinned at 1.0 (L5 below) this is the ONLY
# scale knob. Sweep per-session without code edits (inline env wins):
#   XRT_COMPOSITOR_SCALE_PERCENTAGE=110 python3 -m core.steamvr start
RENDER_SCALE_PCT = 140

# Monado WMR head/controller tracking. SLAM_SUBMIT_FROM_START is load-bearing:
# headless SteamVR has no debug GUI to click "submit", so without it the SLAM
# tracker is created but never fed frames → head pose dead-reckons to infinity.
# Also persisted in ~/.config/environment.d/g2-vr.conf for non-launcher starts.
TRACKING_ENV = {
    "WMR_SLAM": "true",
    "SLAM_SUBMIT_FROM_START": "true",
    # AE on: the SLAM-cam auto-exposure now has a saturation-aware metric (u_autoexpgain.c) so a
    # bright window darkens instead of blowing out (was off to avoid that blow-out). Controller
    # frames use the fixed G2_CTRL_EXPOSURE and are unaffected.
    "WMR_AUTOEXPOSURE": "true",
    "VIT_SYSTEM_LIBRARY_PATH":
        os.path.expanduser("~/.local/share/steamvr-monado/bin/linux64/libbasalt.so"),
    # SLAM_CONFIG intentionally NOT set. Setting it makes t_tracker_slam.cpp:1547 SKIP
    # send_calibration() and use the toml's static reverbg1 (G1) calib instead of the driver's
    # per-unit G2 factory calib — a calibration regression for only a 2-key VioConfig tweak. The
    # head-divergence landmark cull lives in libbasalt.so and is active regardless. Re-enable
    # SLAM_CONFIG only with a G2-derived cam-calib baked into the toml.
}


# SteamVR resolution pin (render-program L5): the GPU-speed autoscale drifts
# 0.98-0.99 between sessions, silently changing the render target and poisoning
# every A/B. Pin the supersample override at 1.0 (= exactly the driver's
# recommendation, so L1 above stays the only scale knob) for the session;
# original values are restored on stop/restore.
VRSETTINGS = os.path.expanduser("~/.local/share/Steam/config/steamvr.vrsettings")
_PIN = {"supersampleManualOverride": True, "supersampleScale": 1.0}
_PIN_SAVED = "/tmp/g2_saved_vrsettings_pin"


def _pin_render_scale(snapshot_dir=None):
    try:
        cfg = json.load(open(VRSETTINGS))
    except (OSError, ValueError) as e:
        print(f"[steamvr] resolution pin skipped ({e})")
        return
    if snapshot_dir:
        shutil.copy2(VRSETTINGS, os.path.join(snapshot_dir, "steamvr.vrsettings.orig"))
    sec = cfg.setdefault("steamvr", {})
    if not os.path.exists(_PIN_SAVED):   # keep the pre-pin originals across crashed sessions
        json.dump({k: sec[k] for k in _PIN if k in sec}, open(_PIN_SAVED, "w"))
    sec.update(_PIN)
    json.dump(cfg, open(VRSETTINGS, "w"), indent=3, sort_keys=True)


def _unpin_render_scale():
    """Put back the pre-pin supersample keys. Idempotent; runs after vrserver has
    exited and rewritten vrsettings, so the restore is what persists."""
    if not os.path.exists(_PIN_SAVED):
        return
    saved = json.load(open(_PIN_SAVED))
    os.unlink(_PIN_SAVED)
    try:
        cfg = json.load(open(VRSETTINGS))
    except (OSError, ValueError) as e:
        print(f"[steamvr] resolution unpin skipped ({e})")
        return
    sec = cfg.setdefault("steamvr", {})
    for k in _PIN:
        sec.pop(k, None)
    sec.update(saved)
    json.dump(cfg, open(VRSETTINGS, "w"), indent=3, sort_keys=True)


def _session_env():
    """Graphical-session env so Steam/SteamVR land on the user's display; pulled
    from gnome-shell if this process didn't inherit it."""
    env = os.environ.copy()
    env.update(TRACKING_ENV)             # ensure SLAM tracking env reaches the driver
    env.setdefault("XRT_COMPOSITOR_SCALE_PERCENTAGE", str(RENDER_SCALE_PCT))
    if env.get("WAYLAND_DISPLAY") and env.get("DISPLAY"):
        return env
    try:
        pid = subprocess.check_output(
            ["pgrep", "-u", str(os.getuid()), "-x", "gnome-shell"]).split()[0].decode()
        for kv in open(f"/proc/{pid}/environ").read().split("\0"):
            k = kv.split("=", 1)[0]
            if k in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR",
                     "DBUS_SESSION_BUS_ADDRESS", "XAUTHORITY"):
                env[k] = kv.split("=", 1)[1]
    except Exception:
        pass
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return env


def _kill(procs):
    for p in procs:
        subprocess.run(["pkill", "-x", p], stderr=subprocess.DEVNULL)


def is_running():
    return subprocess.run(["pgrep", "-x", "vrserver"],
                          stdout=subprocess.DEVNULL).returncode == 0


def _dip_in_effect(config):
    """True if the live desktop layout already matches the dip `config` — every
    connector in config at its target mode and no extra connector active. Used to
    skip a redundant apply_layout (each apply is a visible blackout)."""
    try:
        cur = {c: m for l in mutter.canonical()[1] for c, m in l[5]}
    except Exception:
        return False
    if any(cur.get(c) != m for c, m in config.items()):
        return False
    return all(c in config for c in cur)  # an active connector not in config => not fully dipped


def start():
    """Launch SteamVR with the G2 free, desktop dipped, and perf pinned."""
    # 1. Free the G2 lease so SteamVR's vrcompositor can acquire it (no standalone service).
    subprocess.run(["pkill", "-x", "monado-service"], stderr=subprocess.DEVNULL)
    time.sleep(2)

    # 1b. Wake the G2 panel. After a reboot/idle the HMD's DP link drops to standby and the
    #     connector reads "disconnected", so SteamVR finds no leasable display ("GNOME Wayland
    #     does not support DRM leases"). vrcompositor cannot wake a down DP link — only the WMR
    #     USB driver in monado-service can. Do this BEFORE dipping (the wake hotplug makes mutter
    #     re-apply the full layout, which would undo the dip).
    vr_display._ensure_hmd_awake()

    # 2. Save the desktop layout, then dip it to the negotiated (cached) config so
    #    vrcompositor has the ISO-bandwidth it needs to present to the G2.
    _, lg = mutter.canonical()
    json.dump(lg, open(vr_display.SAVED, "w"))
    config = vr_display.negotiate()        # cached per desktop set (probes only if unknown)
    mutter.apply_layout(config)
    time.sleep(1)

    # 3. Pin GPU P0 clocks + CPU performance governor.
    try:
        gpu.apply_vr_optimizations()
    except Exception as e:
        print(f"[steamvr] gpu.apply skipped: {e}")

    # 3b. Grant vrserver CAP_SYS_NICE so the tracking driver's 1 kHz pose threads can take
    #     SCHED_FIFO (without it the setschedparam calls fail EPERM and pose pushes run
    #     SCHED_OTHER with unbounded scheduling jitter — visible in vrserver.txt). Needs one
    #     sudoers line (scoped, like the monado-service grant):
    #       mrwhite0racle ALL=(root) NOPASSWD: /usr/sbin/setcap cap_sys_nice+ep <steamvr>/bin/linux64/vrserver
    #     Until that line exists this logs and continues (Steam updates also reset the cap,
    #     which is why it re-applies every launch). The update-proof alternative is the
    #     rtprio PAM limit in system/99-g2-vr-rtprio.conf (one-time install, see its header);
    #     with it installed this setcap becomes redundant but stays harmless.
    vrserver_bin = os.path.expanduser(
        "~/.local/share/Steam/steamapps/common/SteamVR/bin/linux64/vrserver")
    if os.path.exists(vrserver_bin):
        r = subprocess.run(["sudo", "-n", "/usr/sbin/setcap", "cap_sys_nice+ep", vrserver_bin],
                           stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            print("[steamvr] vrserver cap_sys_nice not granted (add the scoped sudoers line); "
                  "pose threads will run SCHED_OTHER")

    # 3c. Phase-0 instrumentation: per-session FrameTiming/dmon/pmon observer
    #     channels under var/perf/<ts>/, stopped in restore() on every exit path.
    env = _session_env()
    sess = None
    try:
        sess = perfmon.start(env)
        print(f"[steamvr] perf session: {sess}")
    except Exception as e:
        print(f"[steamvr] perfmon skipped: {e}")

    # 3d. Pin the SteamVR resolution for the session (L5) — before vrserver loads
    #     it — snapshotting the pre-pin settings into the session dir.
    _pin_render_scale(sess)

    # 4. Launch SteamVR; vrcompositor wakes + leases the G2 and presents (our Monado drives tracking).
    subprocess.Popen(["setsid", "-f", "steam", "-applaunch", APPID],
                     env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Detached restore-watchdog: puts the desktop back when vrserver exits ANY way
    # (clean stop, crash, or closed in-headset) — not only via stop(). Replace any
    # stale watcher from a previous session first.
    subprocess.run(["pkill", "-f", "core.steamvr _watchrestore"], stderr=subprocess.DEVNULL)
    subprocess.Popen(["setsid", "-f", "python3", "-m", "core.steamvr", "_watchrestore"],
                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 5. Wait for the stack to come up (Steam can be cold-starting), holding the
    #    dip ONLY when the G2 lease hotplug has actually reverted it. Each
    #    apply_layout is a modeset (a visible blackout), so re-applying on a timer
    #    blackouts the desktop repeatedly; instead re-apply only when the live
    #    layout no longer matches the dip (i.e. a hotplug undid it) — at most one
    #    blackout per real revert.
    deadline = time.time() + 150
    scene = False
    while time.time() < deadline:
        if not _dip_in_effect(config):
            mutter.apply_layout(config)
        if subprocess.run(["pgrep", "-x", "vrcompositor"],
                          stdout=subprocess.DEVNULL).returncode == 0:
            scene = True
            time.sleep(2)
            if not _dip_in_effect(config):       # ride out the lease hotplug, once
                mutter.apply_layout(config)
            break
        time.sleep(2)
    return {"ok": True, "display": config, "running": is_running(), "compositor": scene}


def restore():
    """Restore the desktop layout + GPU clocks + SteamVR settings. Idempotent —
    safe to call twice."""
    vr_display.exit_vr()
    try:
        gpu.revert_optimizations()
    except Exception as e:
        print(f"[steamvr] gpu.revert skipped: {e}")
    try:
        sess = perfmon.stop()
        if sess:
            print(f"[steamvr] perf data: {sess}")
    except Exception as e:
        print(f"[steamvr] perfmon stop skipped: {e}")
    _unpin_render_scale()


def _watch_restore():
    """Detached watchdog: wait until vrserver has come up and then gone away — by a
    clean stop(), a crash, or the user closing SteamVR in-headset — then restore the
    desktop. Guarantees monitors + clocks are put back even when stop() never runs
    (the previous behavior only restored on an explicit stop)."""
    for _ in range(120):              # wait for vrserver to appear (Steam cold start)
        if is_running():
            break
        time.sleep(1)
    while is_running():               # then wait for it to exit, however it ends
        time.sleep(2)
    time.sleep(2)                     # let mutter settle after the lease releases
    restore()


def stop():
    """Stop SteamVR, restore the desktop layout + clocks."""
    subprocess.run(["pkill", "-f", "core.steamvr _watchrestore"], stderr=subprocess.DEVNULL)
    _kill(SVR_PROCS)
    time.sleep(2)
    restore()
    return {"ok": True}


if __name__ == "__main__":  # python3 -m core.steamvr [start|stop|status]
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "start"
    fn = {"start": start, "stop": stop, "restore": restore,
          "_watchrestore": _watch_restore,
          "status": lambda: {"running": is_running()}}.get(cmd)
    print(json.dumps(fn() if fn else {"ok": False, "msg": "usage: start|stop|status"},
                     default=str))
