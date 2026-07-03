#!/usr/bin/env python3
"""Motion-to-photon decomposition on the one-true-clock (render program §4.2).

Measures every stage of the pose→photon chain that the recorded streams cover,
per device, from a capture's telemetry dir:

  imu transport   t_mono(emission) − (hw_ts + offset̂), offset̂ = running min —
                  USB/batching transport jitter of the raw samples
  imu cadence     inter-sample gap per device (state freshness ceiling)
  fusion cadence  optical-fold inter-arrival (the optical refresh of the state)
  pose staleness  t_getpose − t_mono(last imu sample of that device) — how old
                  the underlying state is when SteamVR pulls the pose
  pull cadence    getpose inter-arrival per device. The push loops sleep 1 ms
                  on SCHED_OTHER threads (render lever L4): tail >> 1 ms here
                  IS the unbounded scheduler jitter, measured from real data.

With --perf-session (a g2-studio var/perf/<ts>/ dir whose frametiming.csv +
clock.json cover the same wall-clock span) it extends the chain through the
compositor: WaitGetPoses→NewPosesReady and the modeled total
  M2P ≈ staleness_p50 + compositor pipe (submit→present ≈ 1 frame FIFO)
      + vsync_to_photons (15.668 ms, render L3).

The HMD (device 0) is absent from getpose — the tap excludes it upstream
(ovrd_driver.cpp); controller stages are the felt-signal path anyway.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest  # noqa: E402

VSYNC_TO_PHOTONS_MS = 15.668   # render L3 (G2 @ 90 Hz)
PCTS = (50, 90, 95, 99, 99.9)


def load(stream, telemetry_dir):
    dt = stream.structured_dtype()
    raw = np.fromfile(telemetry_dir / stream.file, dtype=np.uint8)
    n = raw.size // stream.row_size
    return np.frombuffer(raw[: n * stream.row_size].tobytes(), dtype=dt)


def dist_ms(deltas_ns):
    if len(deltas_ns) == 0:
        return None
    ms = np.asarray(deltas_ns, dtype=np.float64) / 1e6
    d = {p: float(np.percentile(ms, p)) for p in PCTS}
    d["max"] = float(ms.max())
    return d


def fmt(name, d, unit="ms"):
    if d is None:
        return f"  {name:<34} (no data)"
    body = "  ".join(f"p{p:g}={d[p]:7.3f}" for p in PCTS)
    return f"  {name:<34} {body}  max={d['max']:8.3f} {unit}"


def stage_report(tel_dir):
    m = Manifest.load(tel_dir)
    imu = load(m.streams["imu"], tel_dir)
    fusion = load(m.streams["fusion"], tel_dir)
    getpose = load(m.streams["getpose"], tel_dir)
    lines = [f"m2p decomposition — {tel_dir}  (clock {m.clock})"]
    out = {}
    for dev in sorted(set(getpose["device_id"])):
        gi = np.sort(getpose[getpose["device_id"] == dev], order="t_mono_ns")
        ii = np.sort(imu[imu["device_id"] == dev], order="t_mono_ns")
        fi = np.sort(fusion[fusion["device_id"] == dev], order="t_mono_ns")
        lines.append(f"\n device {dev}  (imu {len(ii)}  fusion {len(fi)}"
                     f"  getpose {len(gi)})")
        # imu transport: sample age at emission = t_mono − hw_ts, referenced
        # to a per-10s-bucket lower envelope (a running min gets stuck on the
        # clock-tracker's settling transient at session start, and a bucketed
        # envelope also cancels any residual hw→mono drift)
        if len(ii) > 1:
            off = (ii["t_mono_ns"].astype(np.int64)
                   - ii["hw_ts_ns"].astype(np.int64))
            bucket = ((ii["t_mono_ns"] - ii["t_mono_ns"][0]) // 10_000_000_000)
            env = {b: off[bucket == b].min() for b in np.unique(bucket)}
            transport = off - np.vectorize(env.get)(bucket)
            lines.append(fmt("imu transport (age vs envelope)",
                             dist_ms(transport)))
            imu_gaps = np.diff(ii["t_mono_ns"].astype(np.int64))
            lines.append(fmt("imu cadence", dist_ms(imu_gaps)))
        if len(fi) > 1:
            lines.append(fmt("fusion fold cadence",
                             dist_ms(np.diff(fi["t_mono_ns"].astype(np.int64)))))
        if len(gi) > 1 and len(ii) > 1:
            idx = np.searchsorted(ii["t_mono_ns"], gi["t_mono_ns"]) - 1
            ok = idx >= 0
            stale = (gi["t_mono_ns"][ok].astype(np.int64)
                     - ii["t_mono_ns"][idx[ok]].astype(np.int64))
            lines.append(fmt("pose staleness at pull", dist_ms(stale)))
            pulls = np.diff(gi["t_mono_ns"].astype(np.int64))
            pd = dist_ms(pulls)
            lines.append(fmt("getpose pull cadence (L4 jitter)", pd))
            out[int(dev)] = {"pull_cadence_ms": pd,
                             "staleness_ms": dist_ms(stale),
                             "n_pulls": int(len(gi))}
    return "\n".join(lines), out


def perf_extension(perf_dir):
    """WaitGetPoses→NewPosesReady + modeled M2P from a Phase-0 session."""
    import csv
    ft = Path(perf_dir) / "frametiming.csv"
    if not ft.exists():
        return f"\n perf session {perf_dir}: no frametiming.csv"
    with open(ft) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return f"\n perf session {perf_dir}: frametiming.csv empty"
    wgp = [float(r["m_flNewPosesReadyMs"]) - float(r["m_flWaitGetPosesCalledMs"])
           for r in rows]
    arr = np.array(wgp)
    d = {p: float(np.percentile(arr, p)) for p in PCTS}
    d["max"] = float(arr.max())
    total = d[50] + 11.111 + VSYNC_TO_PHOTONS_MS
    return "\n".join([
        f"\n perf session {perf_dir} ({len(rows)} frames)",
        fmt("WaitGetPoses -> NewPosesReady", d),
        f"  modeled M2P p50 ≈ poses-ready {d[50]:.2f} + submit->present 11.111"
        f" + photons {VSYNC_TO_PHOTONS_MS} = {total:.2f} ms"
        f"  (+ pose staleness above)",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("telemetry_dir", help="capture telemetry/ dir (manifest.json)")
    ap.add_argument("--perf-session",
                    help="g2-studio var/perf/<ts>/ dir of the same session")
    ap.add_argument("--json", help="also write stage distributions to this path")
    args = ap.parse_args()
    text, data = stage_report(Path(args.telemetry_dir))
    if args.perf_session:
        text += "\n" + perf_extension(args.perf_session)
    print(text)
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
