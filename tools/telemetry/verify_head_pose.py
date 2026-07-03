#!/usr/bin/env python3
"""verify_head_pose.py -- sanity-check head_pose.bin before we rely on it for flip detection.

Three checks:
  1. Yaw track over time vs head IMU gyro_z integrated. The SLAM yaw should track the gyro yaw
     short-term (both see the same head rotation) and not drift relative to it over long spans
     (SLAM closes the loop on room features). Persistent divergence = SLAM is wrong.
  2. Head pose position drift. SLAM in a static room should produce bounded position; large
     drifts mean SLAM is degenerate.
  3. Gaps / NaNs / discontinuities. A SLAM dropout shows as a missing or warm-up segment.

Usage: verify_head_pose.py <capture_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest  # noqa: E402
import g2_geom as G  # noqa: E402


def quat_to_yaw_pitch_roll(q):
    """Returns (yaw, pitch, roll) in radians per row of q (Nx4 x,y,z,w)."""
    q = np.asarray(q, dtype=float)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # ZYX intrinsic (yaw about world-Y for OpenXR — we're after the trend, not the exact axis)
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return yaw, pitch, roll


def unwrap_yaw(yaw_rad):
    return np.unwrap(yaw_rad)


def integrate_gyro(t_ns, gyro_z):
    """Cumulative trapezoidal integral of gyro_z over time, in radians."""
    t_s = t_ns.astype(np.float64) / 1e9
    dt = np.diff(t_s, prepend=t_s[0])
    return np.cumsum(gyro_z * dt)


def main():
    if len(sys.argv) != 2:
        print("usage: verify_head_pose.py <capture_dir>")
        sys.exit(2)
    cap = Path(sys.argv[1])
    telem = cap / "telemetry"
    m = Manifest.load(telem)

    hp = G.load_stream(telem, m, "head_pose")
    t_hp = hp["t_mono_ns"].astype(np.int64)
    pos = np.stack([hp["px"], hp["py"], hp["pz"]], axis=1).astype(float)
    quat = G.quat_normalize(np.stack([hp["qx"], hp["qy"], hp["qz"], hp["qw"]], axis=1))
    print(f"head_pose.bin: {len(hp)} rows over {(t_hp[-1] - t_hp[0])/1e9:.1f}s")

    # Position drift
    pos_range = pos.max(axis=0) - pos.min(axis=0)
    print(f"\n[1] POSITION DRIFT / RANGE (x, y, z meters):")
    print(f"    range:  x={pos_range[0]:.3f}  y={pos_range[1]:.3f}  z={pos_range[2]:.3f}")
    print(f"    p99-p1: x={np.percentile(pos[:,0], 99)-np.percentile(pos[:,0], 1):.3f}  "
          f"y={np.percentile(pos[:,1], 99)-np.percentile(pos[:,1], 1):.3f}  "
          f"z={np.percentile(pos[:,2], 99)-np.percentile(pos[:,2], 1):.3f}")
    # First-vs-last frame difference
    print(f"    start->end position drift: {np.linalg.norm(pos[-1] - pos[0]):.3f} m "
          f"(SLAM should return near origin if user came back to start)")

    # Yaw track
    yaw, pitch, roll = quat_to_yaw_pitch_roll(quat)
    yaw_u = unwrap_yaw(yaw)
    print(f"\n[2] HEAD YAW TRACK (degrees):")
    print(f"    yaw range (unwrapped): {np.degrees(yaw_u.max() - yaw_u.min()):.1f}°")
    print(f"    yaw at session end - start: {np.degrees(yaw_u[-1] - yaw_u[0]):.1f}°")
    print(f"    pitch range: {np.degrees(pitch.max() - pitch.min()):.1f}°")
    print(f"    roll range: {np.degrees(roll.max() - roll.min()):.1f}°")

    # Yaw rate from SLAM (numerical diff)
    t_s = (t_hp - t_hp[0]).astype(np.float64) / 1e9
    dyaw = np.diff(yaw_u, prepend=yaw_u[0])
    dt = np.diff(t_s, prepend=t_s[0])
    yaw_rate_slam = dyaw / np.where(dt > 0, dt, 1.0)
    print(f"    SLAM yaw rate: med={np.degrees(np.median(np.abs(yaw_rate_slam))):.1f}°/s  "
          f"p95={np.degrees(np.percentile(np.abs(yaw_rate_slam), 95)):.1f}°/s  "
          f"max={np.degrees(np.max(np.abs(yaw_rate_slam))):.1f}°/s")

    # Look for jumps (frame-to-frame yaw change > 30°)
    jumps = np.abs(np.diff(yaw_u))
    big = np.degrees(jumps) > 30.0
    print(f"    yaw jumps > 30° in one frame: {big.sum()}  (any indicates SLAM glitch)")

    # Compare to head IMU integrated yaw
    imu = G.load_stream(telem, m, "imu")
    himu = imu[imu["device_id"] == 0]
    if len(himu) > 100:
        t_imu = himu["t_mono_ns"].astype(np.int64)
        gz = himu["gz"].astype(float)  # head-IMU gyro Z (rad/s)
        # Integrate gyro_z (raw, no bias correction) over the SLAM time range
        keep = (t_imu >= t_hp[0]) & (t_imu <= t_hp[-1])
        t_imu_k = t_imu[keep]
        gz_k = gz[keep]
        if len(gz_k) > 100:
            yaw_imu_raw = integrate_gyro(t_imu_k, gz_k)
            # Interpolate SLAM yaw to IMU timestamps for direct comparison
            slam_at_imu = np.interp(t_imu_k.astype(float), t_hp.astype(float), yaw_u)
            # Remove average bias by subtracting first-1s offsets
            warm_mask = (t_imu_k - t_imu_k[0]) < 1e9
            if warm_mask.sum() > 10:
                bias = (slam_at_imu[warm_mask] - yaw_imu_raw[warm_mask]).mean()
            else:
                bias = slam_at_imu[0] - yaw_imu_raw[0]
            yaw_imu_aligned = yaw_imu_raw + bias

            diff = slam_at_imu - yaw_imu_aligned
            print(f"\n[3] SLAM YAW vs HEAD-IMU INTEGRATED GYRO_Z (bias-removed, in degrees):")
            print(f"    end-of-session divergence:  {np.degrees(diff[-1]):+.1f}°  "
                  f"(non-zero = either SLAM corrected drift, or gyro_z's local axis differs from world-up)")
            print(f"    max-abs divergence over session: {np.degrees(np.abs(diff).max()):.1f}°")
            print(f"    NB: SLAM tracks world-frame yaw; gyro_z is body-frame Z. They agree only when ")
            print(f"        the head's Z is roughly aligned with world-up (i.e. user's head is upright). ")
            print(f"        Tilt mismatch can produce a steady bias even with NO drift. The SAFE signal is ")
            print(f"        the SHAPE — both should show the SAME swings, just maybe with constant offset.")
            # Correlation of swings
            try:
                from numpy import corrcoef
                k = min(len(slam_at_imu), len(yaw_imu_aligned))
                if k > 100:
                    corr = corrcoef(slam_at_imu[:k], yaw_imu_aligned[:k])[0, 1]
                    print(f"    Pearson correlation (SLAM yaw vs gyro-integrated yaw): {corr:.4f}")
                    print(f"    (>0.9 = SLAM faithfully tracks gyro short-term; <0.5 = serious divergence)")
            except Exception as e:
                print(f"    correlation calc failed: {e}")

    # Gaps in head_pose
    print(f"\n[4] GAPS IN head_pose.bin (frame-to-frame):")
    dts_ms = np.diff(t_hp) / 1e6
    print(f"    inter-sample dt: med={np.median(dts_ms):.1f}ms  p95={np.percentile(dts_ms, 95):.1f}ms  "
          f"max={dts_ms.max():.1f}ms")
    big_gaps = (dts_ms > 100).sum()
    print(f"    gaps > 100ms: {big_gaps}  (any indicates SLAM stalled briefly)")

    # NaN check
    nan_count = (~np.isfinite(pos).all(axis=1)).sum() + (~np.isfinite(quat).all(axis=1)).sum()
    print(f"\n[5] NaN / non-finite head poses: {nan_count}")


if __name__ == "__main__":
    main()
