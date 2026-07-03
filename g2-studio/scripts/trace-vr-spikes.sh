#!/usr/bin/env bash
# On-demand kernel-level spike trace for a live VR session (render program §4.4).
# Records sched_switch + dma_fence + drm_vblank_event for DURATION seconds into
# the current perf session dir; open the .dat with gpuvis or `trace-cmd report`.
# perfreport says WHICH second and what class a spike burst is — this says what
# the kernel was doing during the unattributed ones.
#
# Needs root for tracefs. One-time scoped grant (keeps the no-broad-root posture):
#   echo 'mrwhite0racle ALL=(root) NOPASSWD: /usr/bin/trace-cmd' \
#     | sudo tee /etc/sudoers.d/g2ctl-trace
#
# Usage: trace-vr-spikes.sh [duration-seconds]   (default 10)
set -euo pipefail

DUR=${1:-10}
REPO=$(cd "$(dirname "$0")/.." && pwd)
SESS=$(readlink -f "$REPO/var/perf/latest" 2>/dev/null || echo "$REPO/var/perf")
mkdir -p "$SESS"
OUT="$SESS/trace-$(date +%Y%m%d-%H%M%S).dat"
TRACE_CMD=/usr/bin/trace-cmd   # the sudoers grant is path-exact

sudo -n "$TRACE_CMD" stat >/dev/null 2>&1 || {
    echo "ERR: passwordless sudo for $TRACE_CMD not granted — see the header" >&2
    exit 1
}

EVENTS=(-e sched:sched_switch -e 'dma_fence:*' -e 'drm:*')
# nvidia_modeset:flip_request/flip_occurred exist once the patched 595.84
# modules are live — the ONLY kernel-side timing for NVKMS-ioctl lease flips.
if sudo -n "$TRACE_CMD" list -e 2>/dev/null | grep -q '^nvidia_modeset:'; then
    EVENTS+=(-e 'nvidia_modeset:*')
fi
echo "tracing ${DUR}s (${EVENTS[*]//-e /}) -> $OUT"
sudo -n "$TRACE_CMD" record -o "$OUT" "${EVENTS[@]}" sleep "$DUR" >/dev/null 2>&1

# The .dat is root-owned (only trace-cmd is granted, not chown); read it via
# plain trace-cmd if the mode allows, else through the grant.
report() { trace-cmd report -i "$OUT" 2>/dev/null \
           || sudo -n "$TRACE_CMD" report -i "$OUT" 2>/dev/null; }

# Inline vblank-cadence summary. On nvidia-drm only *_delivered fires (event
# delivery to the flipping client; plain drm_vblank_event never does — measured
# 2026-07-03). At 90 Hz the leased CRTC must tick every 11.111 ms — gaps are
# missed scanouts regardless of what userspace logged. Whether lease flips via
# NVKMS_IOCTL_FLIP surface here at all is an open question the first in-session
# run answers; sched/dma_fence forensics work either way.
report | python3 -c '
import re, sys
from collections import defaultdict
vbl = defaultdict(list)
occ = defaultdict(list)
req = defaultdict(list)
for line in sys.stdin:
    m = re.search(r"([0-9]+\.[0-9]+): drm_vblank_event(?:_delivered)?: "
                  r".*crtc=([0-9]+)", line)
    if m:
        vbl[m.group(2)].append(float(m.group(1)))
        continue
    m = re.search(r"([0-9]+\.[0-9]+): flip_occurred: .*api_head=([0-9]+)", line)
    if m:
        occ[m.group(2)].append(float(m.group(1)))
        continue
    m = re.search(r"([0-9]+\.[0-9]+): flip_request: .*api_head=([0-9]+)", line)
    if m:
        req[m.group(2)].append(float(m.group(1)))

def stats(name, t):
    gaps = sorted((b - a) * 1000 for a, b in zip(t, t[1:]))
    if not gaps:
        return
    p = lambda q: gaps[min(len(gaps) - 1, int(len(gaps) * q))]
    print(f"{name}: {len(t)} events  gap p50={p(.5):.3f} "
          f"p99={p(.99):.3f} max={gaps[-1]:.3f} ms")

if not (vbl or occ):
    print("no vblank/flip events (displays idle, or nvidia_modeset events not yet live)")
for crtc, t in sorted(vbl.items()):
    stats(f"drm crtc {crtc}", t)
for h, t in sorted(occ.items()):
    stats(f"nvkms flip_occurred head {h}", t)
    # request -> next completion on the same head = queue/present latency
    lat, r = [], sorted(req.get(h, []))
    i = 0
    for tc in t:
        while i < len(r) - 1 and r[i + 1] < tc:
            i += 1
        if r and r[i] < tc:
            lat.append((tc - r[i]) * 1000)
    if lat:
        lat.sort()
        p = lambda q: lat[min(len(lat) - 1, int(len(lat) * q))]
        print(f"  flip request->occurred head {h}: p50={p(.5):.3f} "
              f"p99={p(.99):.3f} max={lat[-1]:.3f} ms")
'
echo "full trace: trace-cmd report -i $OUT   (or gpuvis $OUT)"
