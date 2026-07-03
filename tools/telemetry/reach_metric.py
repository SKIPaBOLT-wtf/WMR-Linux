#!/usr/bin/env python3
"""reach_metric.py -- does the user's natural reach exceed BODY_REACH_M=0.9?

Two signals from a capture's cleaned-GT trajectory:
  1. Distribution of |controller - head| over all valid reference frames. If p99 > 0.9 m, the
     0.9 m clamp would have artificially limited natural motion during the capture.
  2. For each fold-gap >= 100 ms (an optical dropout), pos_error at re-acquisition: how far is
     the re-acquired optical pose from the LAST optical pose minus the IMU's predicted motion?
     If this is large, the body-anchor pulled the controller toward a stale offset while it
     was actually moving elsewhere -> "wander off" + "snap back".

Usage: reach_metric.py <capture_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest, DEVICE_NAMES  # noqa: E402
from smooth_ref import build_reference  # noqa: E402
import g2_geom as G  # noqa: E402


def analyse(capture: Path):
    telem = capture / "telemetry"
    m = Manifest.load(telem)
    # Head pose timeline
    hp = G.load_stream(telem, m, "head_pose")
    hp_t = hp["t_mono_ns"].astype(np.int64)
    hp_p = np.stack([hp["px"], hp["py"], hp["pz"]], axis=1).astype(float)

    print(f"\n=== capture: {capture.name} ===")
    for dev in (1, 2):
        ref = build_reference(telem, dev)
        if ref is None:
            print(f"  dev{dev}: no reference")
            continue
        valid = ref.valid
        t_ref = ref.t_ns[valid]
        pos_ref = ref.pos[valid]
        if t_ref.shape[0] == 0:
            continue

        # Per ref-sample: head pose at that time (nearest)
        head_at = np.zeros((t_ref.shape[0], 3))
        for k, t in enumerate(t_ref):
            j = np.searchsorted(hp_t, t)
            best = -1
            for cand in (j - 1, j):
                if 0 <= cand < hp_t.shape[0]:
                    if best < 0 or abs(int(hp_t[cand]) - int(t)) < abs(int(hp_t[best]) - int(t)):
                        best = cand
            if best >= 0:
                head_at[k] = hp_p[best]

        # Controller-to-head distance distribution
        d = np.linalg.norm(pos_ref - head_at, axis=1)

        print(f"\n  dev{dev}  {DEVICE_NAMES.get(dev, '')}  (n={t_ref.shape[0]} valid ref frames)")
        print(f"    controller-to-head distance: "
              f"med={np.median(d):.3f}m  p75={np.percentile(d,75):.3f}m  "
              f"p95={np.percentile(d,95):.3f}m  p99={np.percentile(d,99):.3f}m  max={d.max():.3f}m")
        # How much of the session is beyond 0.9m? (i.e. how often would the clamp have been active)
        beyond_09 = float(np.mean(d > 0.9) * 100)
        beyond_11 = float(np.mean(d > 1.1) * 100)
        print(f"    fraction > 0.9m: {beyond_09:5.1f}%  (current clamp)")
        print(f"    fraction > 1.1m: {beyond_11:5.1f}%  (proposed clamp)")

        # Re-acquisition events: find gaps in t_ref where consecutive valid-ref samples are >= 100ms apart
        if t_ref.shape[0] >= 2:
            dt = np.diff(t_ref) / 1e6  # ms
            gaps = np.where(dt >= 100)[0]
            if len(gaps) > 0:
                # For each gap, "snap" = distance between pos_ref[gap+1] and pos_ref[gap]
                snap_d = np.linalg.norm(pos_ref[gaps + 1] - pos_ref[gaps], axis=1)
                print(f"    optical dropouts (>=100ms gap): n={len(gaps)}  "
                      f"snap dist med={np.median(snap_d):.3f}m  "
                      f"p95={np.percentile(snap_d, 95):.3f}m  max={snap_d.max():.3f}m")
                # Coast-gap distribution
                gap_ms = dt[gaps]
                print(f"    coast-gap distribution: med={np.median(gap_ms):.0f}ms  "
                      f"p95={np.percentile(gap_ms, 95):.0f}ms  max={gap_ms.max():.0f}ms")
            else:
                print(f"    no dropouts >= 100ms")


def main():
    if len(sys.argv) != 2:
        print("usage: reach_metric.py <capture_dir>")
        sys.exit(2)
    analyse(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
