#!/usr/bin/env python3
"""detection_f1.py -- precision/recall/F1 for the G2 matcher's detection performance.

What this answers (the user's question):
  When the controller IS truly in a camera's view, does the matcher detect it correctly?
  When the matcher claims a detection, is the detection actually right?

Ground-truth-grade signal we have:
  - The smoothed/de-flipped cleaned-GT reference (smooth_ref.build_reference): "where the
    controller really was, smoothed". We project this into each camera to derive the
    IN-FOV mask (front-facing + in-image-bounds + plausible distance).
  - pose_attempt.bin: the matcher's per-attempt accept/reject outcome with reproj err.
  - The pose_attempt's pose vs cleaned-GT MSE: tells us whether an "accepted" pose was
    actually correct (low position error AND low orientation error vs cleaned-GT).

We then classify each (frame_t, device) into:
  TP: in-view AND accepted AND correct
  FP: accepted but either NOT in-view (shouldn't have found one) OR not correct
  FN: in-view but NOT accepted (missed detection)
  TN: not-in-view AND not-accepted (no opportunity)

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 PR / (P + R)

Also reports per-camera detection coverage so we can see which cameras carry most of
the detections vs misses.

Usage:
    detection_f1.py <capture_dir> [--match-ms 25] [--max-ref-gap-ms 150]
                    [--mse-pos-cm 5.0] [--mse-ori-deg 15.0]
                    [--min-leds 3] [--cams JSON] [--ctrl-left JSON] [--ctrl-right JSON]

Outputs: per-device summary table + per-camera coverage, plus optional --out for a
detailed JSON dump.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import Manifest, DEVICE_NAMES  # noqa: E402
from smooth_ref import build_reference  # noqa: E402
import g2_geom as G  # noqa: E402

DEFAULT_CAMS = str(__import__("pathlib").Path(__file__).resolve().parent / "data/hmd-cameras-replay.json")  # pinned: live driver rewrites the ~/.config copy
DEFAULT_CTRL_LEFT = "/home/mrwhite0racle/.config/monado/wmr/controller_A85K1111630014L.json"
DEFAULT_CTRL_RIGHT = "/home/mrwhite0racle/.config/monado/wmr/controller_A85K5091930012R.json"
LED_ANGLE_DEG = 82.0
P_YZ_FLIP_R = np.diag([1.0, -1.0, -1.0])


# --- quaternion / pose math ---


def quat_to_R(q):
    """q = (x,y,z,w) -> 3x3 rotation. Vectorised over leading dims."""
    q = np.asarray(q, dtype=float)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n = np.where(n > 1e-12, n, 1.0)
    x, y, z, w = x / n, y / n, z / n, w / n
    R = np.empty(q.shape[:-1] + (3, 3), dtype=float)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_geodesic_deg(q1, q2):
    """Geodesic angle (deg) between two unit quaternions, vectorised over leading dims."""
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def R_to_quat(R):
    """3x3 rotation matrix -> quaternion (x,y,z,w)."""
    R = np.asarray(R, dtype=float)
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return G.quat_normalize(np.array([qx, qy, qz, qw]))


def pose_mul(R_ab, t_ab, R_bc, t_bc):
    """Compose column-vector poses: P_ac = P_ab . P_bc."""
    return R_ab @ R_bc, R_ab @ t_bc + t_ab


def pose_inv(R_ab, t_ab):
    """Invert a column-vector pose."""
    R_ba = R_ab.T
    return R_ba, -(R_ba @ t_ab)


def pose_flip_YZ(R, t):
    """Mirror Monado's pose_flip_YZ: P_out = F . P_in . F, t_out = F t."""
    return P_YZ_FLIP_R @ R @ P_YZ_FLIP_R, P_YZ_FLIP_R @ t


# --- camera + LED model loading ---


@dataclass
class Camera:
    id: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray  # WMR/RT8 params: k1,k2,p1,p2,k3,k4,k5,k6,codx,cody,rpmax
    P_imu_cam_R: np.ndarray  # 3x3 rotation IMU<-cam (from quaternion)
    P_imu_cam_t: np.ndarray  # 3 translation IMU<-cam
    roi_x: int
    roi_y: int
    roi_w: int
    roi_h: int


def _quat_xyzw_to_R(qx, qy, qz, qw):
    return quat_to_R(np.array([qx, qy, qz, qw]))


def load_cameras(cams_json: str) -> list[Camera]:
    with open(cams_json) as f:
        d = json.load(f)
    cams = []
    for i, c in enumerate(d["cameras"]):
        K = np.asarray(c["intrinsics"], dtype=float)
        dist = np.asarray(c.get("distortion", [0.0] * 8), dtype=float)
        # P_imu_cam: pose of cam in IMU frame
        P = c["P_imu_cam"]
        pos = np.asarray(P["position"], dtype=float)
        ori = P["orientation"]  # [x,y,z,w]
        R = _quat_xyzw_to_R(ori[0], ori[1], ori[2], ori[3])
        roi = c["roi"]
        cams.append(
            Camera(
                id=i,
                width=int(c["width"]),
                height=int(c["height"]),
                fx=K[0, 0],
                fy=K[1, 1],
                cx=K[0, 2],
                cy=K[1, 2],
                distortion=dist,
                P_imu_cam_R=R,
                P_imu_cam_t=pos,
                roi_x=int(roi["x"]),
                roi_y=int(roi["y"]),
                roi_w=int(roi["w"]),
                roi_h=int(roi["h"]),
            )
        )
    return cams


def load_led_model(ctrl_json: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (positions Nx3 in object frame, normals Nx3)."""
    with open(ctrl_json) as f:
        d = json.load(f)
    # OG WMR JSON layout: "ImagePoints" or similar. Inspect first.
    # We use the "TrackedObjectPoints" / "TrackedFeatures" / "LedPoints" path that wmr_config.c parses.
    # Fall back to scanning keys for an array of {Position, Normal}.
    def find_leds(node):
        if isinstance(node, list):
            if node and isinstance(node[0], dict) and any(k in node[0] for k in ("Position", "position")):
                return node
            for x in node:
                r = find_leds(x)
                if r:
                    return r
        elif isinstance(node, dict):
            for k, v in node.items():
                if k.lower().endswith("ledpoints") or k.lower().endswith("trackedfeatures"):
                    if isinstance(v, list) and v:
                        return v
                r = find_leds(v)
                if r:
                    return r
        return None

    leds_raw = find_leds(d)
    if leds_raw is None:
        raise RuntimeError(f"no LED array found in {ctrl_json}")
    pos = []
    nrm = []
    for led in leds_raw:
        p = led.get("Position") or led.get("position") or led.get("pos")
        n = led.get("Normal") or led.get("normal") or led.get("nrm")
        if p is None or n is None:
            continue
        pos.append([float(p[0]), float(p[1]), float(p[2])])
        nrm.append([float(n[0]), float(n[1]), float(n[2])])
    return np.asarray(pos), np.asarray(nrm)


# --- visibility projection ---


def rt8_project(led_cam: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror t_camera_models_project for WMR interpreted as OpenCV RT8."""
    x = led_cam[..., 0]
    y = led_cam[..., 1]
    z = led_cam[..., 2]
    front = z >= 1e-6
    safe_z = np.where(front, z, 1.0)
    xp = x / safe_z
    yp = y / safe_z
    rp2 = xp * xp + yp * yp
    d = np.pad(cam.distortion.astype(float), (0, max(0, 11 - cam.distortion.shape[0])))
    k1, k2, p1, p2, k3, k4, k5, k6 = d[:8]
    rpmax = d[10]
    cdist = (1.0 + rp2 * (k1 + rp2 * (k2 + rp2 * k3))) / (1.0 + rp2 * (k4 + rp2 * (k5 + rp2 * k6)))
    delta_x = 2.0 * p1 * xp * yp + p2 * (rp2 + 2.0 * xp * xp)
    delta_y = 2.0 * p2 * xp * yp + p1 * (rp2 + 2.0 * yp * yp)
    u = cam.fx * (xp * cdist + delta_x) + cam.cx
    v = cam.fy * (yp * cdist + delta_y) + cam.cy
    injective = True if rpmax == 0.0 else rp2 <= rpmax * rpmax
    return u, v, front & injective


def cam_visible(led_cam: np.ndarray, led_normal_cam: np.ndarray, cam: Camera) -> np.ndarray:
    """Return mask (N,) for LEDs visible from this camera.

    Visible = z > 0, projects within this cropped camera frame, and the LED normal
    passes Monado's LED_ANGLE facing test.
    """
    u, v, valid_project = rt8_project(led_cam, cam)
    in_x = (u >= 0.0) & (u < cam.width)
    in_y = (v >= 0.0) & (v < cam.height)
    view_vec = led_cam / (np.linalg.norm(led_cam, axis=-1, keepdims=True) + 1e-12)
    facing_dot = np.sum(view_vec * led_normal_cam, axis=-1)
    facing = facing_dot <= np.cos(np.deg2rad(180.0 - LED_ANGLE_DEG))
    return valid_project & in_x & in_y & facing


def project_device_leds(
    led_pos: np.ndarray,      # (L,3) LED positions, WMR/OpenCV model frame
    led_nrm: np.ndarray,      # (L,3) LED normals
    R_xr_dev: np.ndarray,     # device pose in OpenXR world
    t_xr_dev: np.ndarray,
    R_xr_hmd: np.ndarray,     # HMD IMU pose in OpenXR world
    t_xr_hmd: np.ndarray,
    cam: Camera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a device's VISIBLE LEDs (facing + in-image, per cam_visible) into one
    camera, from OpenXR-world device+head poses. Returns (u, v) of the visible LEDs."""
    R_cv_hmd, t_cv_hmd = pose_flip_YZ(R_xr_hmd, t_xr_hmd)
    R_cv_mod, t_cv_mod = pose_flip_YZ(R_xr_dev, t_xr_dev)
    R_cv_cam, t_cv_cam = pose_mul(R_cv_hmd, t_cv_hmd, cam.P_imu_cam_R, cam.P_imu_cam_t)
    R_cam_cv, t_cam_cv = pose_inv(R_cv_cam, t_cv_cam)
    R_cam_mod, t_cam_mod = pose_mul(R_cam_cv, t_cam_cv, R_cv_mod, t_cv_mod)
    led_c = led_pos @ R_cam_mod.T + t_cam_mod
    nrm_c = led_nrm @ R_cam_mod.T
    vis = cam_visible(led_c, nrm_c, cam)
    u, v, _ = rt8_project(led_c, cam)
    return u[vis], v[vis]


def in_fov_per_cam(
    pose_xrworld_device_R: np.ndarray,  # (N,3,3)
    pose_xrworld_device_t: np.ndarray,  # (N,3)
    head_xrworld_R: np.ndarray,         # (N,3,3) HMD IMU pose in OpenXR world
    head_xrworld_t: np.ndarray,         # (N,3)
    led_pos: np.ndarray,                # (L,3) in WMR/OpenCV LED-model frame
    led_nrm: np.ndarray,                # (L,3) in WMR/OpenCV LED-model frame
    cams: list[Camera],
) -> np.ndarray:
    """Return mask (N, n_cams) of how many LEDs are visible from each cam at each frame."""
    N = pose_xrworld_device_R.shape[0]
    nC = len(cams)
    led_visible_count = np.zeros((N, nC), dtype=np.int32)
    for i in range(N):
        R_xr_dev, t_xr_dev = pose_xrworld_device_R[i], pose_xrworld_device_t[i]
        R_xr_hmd, t_xr_hmd = head_xrworld_R[i], head_xrworld_t[i]
        R_cv_hmd, t_cv_hmd = pose_flip_YZ(R_xr_hmd, t_xr_hmd)
        R_cv_model, t_cv_model = pose_flip_YZ(R_xr_dev, t_xr_dev)
        for j, c in enumerate(cams):
            R_cv_cam, t_cv_cam = pose_mul(R_cv_hmd, t_cv_hmd, c.P_imu_cam_R, c.P_imu_cam_t)
            R_cam_cv, t_cam_cv = pose_inv(R_cv_cam, t_cv_cam)
            R_cam_model, t_cam_model = pose_mul(R_cam_cv, t_cam_cv, R_cv_model, t_cv_model)
            led_c = led_pos @ R_cam_model.T + t_cam_model
            nrm_c = led_nrm @ R_cam_model.T
            vis = cam_visible(led_c, nrm_c, c)
            led_visible_count[i, j] = int(vis.sum())
    return led_visible_count


def pose_attempt_to_xrworld_device(row, cam: Camera, head_q, head_p):
    """Mirror the live path:
    P_cvworld_hmd = flip(P_xrworld_hmd);
    P_cvworld_model = P_cvworld_cam . P_cam_model;
    P_xrworld_device ~= flip(P_cvworld_model) for WMR controllers (identity model/device offset).
    """
    R_xr_hmd = quat_to_R(head_q)
    t_xr_hmd = np.asarray(head_p, dtype=float)
    R_cv_hmd, t_cv_hmd = pose_flip_YZ(R_xr_hmd, t_xr_hmd)
    R_cv_cam, t_cv_cam = pose_mul(R_cv_hmd, t_cv_hmd, cam.P_imu_cam_R, cam.P_imu_cam_t)
    q_cam_model = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=float)
    R_cam_model = quat_to_R(q_cam_model)
    t_cam_model = np.array([row["px"], row["py"], row["pz"]], dtype=float)
    R_cv_model, t_cv_model = pose_mul(R_cv_cam, t_cv_cam, R_cam_model, t_cam_model)
    return pose_flip_YZ(R_cv_model, t_cv_model)


# --- main analysis ---


def _interp_quat_pos(t_ref, q_ref, p_ref, t_query):
    """Nearest-neighbour pose at each query time (avoids slerp on the unsmoothed deflip output)."""
    out_q = np.full((t_query.shape[0], 4), np.nan)
    out_p = np.full((t_query.shape[0], 3), np.nan)
    j = np.searchsorted(t_ref, t_query)
    for k, t in enumerate(t_query):
        best = -1
        for cand in (j[k] - 1, j[k]):
            if 0 <= cand < t_ref.shape[0]:
                if best < 0 or abs(int(t_ref[cand]) - int(t)) < abs(int(t_ref[best]) - int(t)):
                    best = cand
        if best >= 0:
            out_q[k] = q_ref[best]
            out_p[k] = p_ref[best]
    return out_q, out_p


def _interp_reference_to_times(ref, t_query, max_gap_ms):
    """Interpolate the cleaned-GT reference onto camera-frame times.

    A query is valid only when bracketed by valid reference samples whose gap is
    short enough to be real interpolation, not long-dropout hallucination.
    """
    t_src = ref.t_ns
    p_src = ref.pos
    q_src = ref.quat
    valid_src = ref.valid
    out_q = np.full((t_query.shape[0], 4), np.nan)
    out_p = np.full((t_query.shape[0], 3), np.nan)
    out_valid = np.zeros(t_query.shape[0], dtype=bool)
    max_gap_ns = int(max_gap_ms * 1e6)
    for k, t in enumerate(t_query):
        j = np.searchsorted(t_src, t)
        if j < t_src.shape[0] and int(t_src[j]) == int(t) and valid_src[j]:
            out_q[k] = q_src[j]
            out_p[k] = p_src[j]
            out_valid[k] = True
            continue
        lo = j - 1
        hi = j
        if lo < 0 or hi >= t_src.shape[0]:
            continue
        if not (valid_src[lo] and valid_src[hi]):
            continue
        span = int(t_src[hi]) - int(t_src[lo])
        if span <= 0 or span > max_gap_ns:
            continue
        frac = (int(t) - int(t_src[lo])) / span
        out_p[k] = (1.0 - frac) * p_src[lo] + frac * p_src[hi]
        out_q[k] = G.quat_slerp(q_src[lo], q_src[hi], float(frac))
        out_valid[k] = True
    return out_q, out_p, out_valid


def _nearest_time_index(t_sorted: np.ndarray, t: int, max_dt_ns: int) -> int:
    """Return the single nearest index in t_sorted within max_dt_ns, else -1."""
    j = np.searchsorted(t_sorted, t)
    best = -1
    for cand in (j - 1, j):
        if 0 <= cand < t_sorted.shape[0]:
            if abs(int(t_sorted[cand]) - t) <= max_dt_ns:
                if best < 0 or abs(int(t_sorted[cand]) - t) < abs(int(t_sorted[best]) - t):
                    best = cand
    return best


def _load_head_pose(telem: Path):
    # Trimmed loader: drops warm-up + post-crash-tail rows so an in-FOV query in the dropped tail
    # returns NaN (frame is then excluded from F1 via the existing finite-head-pose mask).
    from headpose_anchor import load_head_pose
    hp = load_head_pose(telem)
    if hp is None or hp.t_ns.size == 0:
        return None
    return hp.t_ns, hp.pos, hp.quat


def analyse_device(
    capture: Path,
    telem: Path,
    dev: int,
    cams: list[Camera],
    ctrl_json: str,
    mse_pos_cm: float,
    mse_ori_deg: float,
    min_leds: int,
    match_ms: float,
    max_ref_gap_ms: float,
    cand_telem: Path | None = None,
    dump_fn: Path | None = None,
) -> dict:
    """If cand_telem is provided, the cleaned-GT reference is built from `telem` (the reference
    capture) and the pose_attempts are read from `cand_telem` (the candidate). Otherwise both
    come from `telem` -- self-referential mode."""
    led_pos, led_nrm = load_led_model(ctrl_json)
    # Cleaned-GT reference (smoothed, de-flipped, world frame). Built from the REFERENCE capture.
    ref = build_reference(telem, dev)
    if ref is None:
        print(f"dev{dev}: no reference (no accepted optical pose)?", file=sys.stderr)
        return {}
    # Head pose timeline
    hp = _load_head_pose(telem)
    if hp is None:
        print("no head_pose.bin -- cannot project into world cams", file=sys.stderr)
        return {}
    hp_t, hp_pos, hp_q = hp

    m = Manifest.load(telem)
    fr = G.load_stream(telem, m, "frame")
    t_query_all = np.unique(fr["hw_ts_ns"].astype(np.int64))
    q_query_all, pos_query_all, ref_valid_all = _interp_reference_to_times(ref, t_query_all, max_ref_gap_ms)
    if not ref_valid_all.any():
        return {}

    # Interpolate head pose to each camera-frame time, then keep only frames where
    # both head pose and cleaned-GT reference are valid.
    hp_q_all, hp_p_all = _interp_quat_pos(hp_t, hp_q, hp_pos, t_query_all)
    valid_head = np.isfinite(hp_p_all[:, 0])
    valid_query = ref_valid_all & valid_head
    t_query = t_query_all[valid_query]
    pos_query = pos_query_all[valid_query]
    q_query = q_query_all[valid_query]
    hp_p_at = hp_p_all[valid_query]
    hp_q_at = hp_q_all[valid_query]
    if t_query.shape[0] == 0:
        return {}
    R_ref = quat_to_R(q_query)
    R_head = quat_to_R(hp_q_at)

    # Per-frame LED visibility counts per cam
    led_count = in_fov_per_cam(R_ref, pos_query, R_head, hp_p_at, led_pos, led_nrm, cams)

    # in_view = any cam has >= min_leds visible LEDs
    per_cam_in_view = led_count >= min_leds
    in_view = per_cam_in_view.any(axis=1)

    # Load pose_attempt for this device, take ACCEPTED attempts (outcome=1 or 2).
    # In external-reference mode, read pose_attempts from the CANDIDATE (cand_telem), not the
    # reference capture's telemetry.
    pa_telem = cand_telem if cand_telem is not None else telem
    m_pa = Manifest.load(pa_telem)
    pa = G.load_stream(pa_telem, m_pa, "pose_attempt")
    pa = pa[pa["device_id"] == dev]
    accepted = pa[(pa["outcome"] == 1) | (pa["outcome"] == 2)]
    t_acc = accepted["hw_ts_ns"].astype(np.int64)
    # For each accepted pose: find the single nearest evaluated camera frame within match_ms.
    correct = np.zeros(len(accepted), dtype=bool)
    accept_frame_idx = np.full(len(accepted), -1, dtype=np.int64)
    match_dt_ns = int(match_ms * 1e6)
    if len(accepted) and t_query.shape[0]:
        for k in range(len(accepted)):
            t = int(t_acc[k])
            best = _nearest_time_index(t_query, t, match_dt_ns)
            if best < 0:
                continue
            accept_frame_idx[k] = best
            cam_id = int(accepted[k]["cam_id"])
            if cam_id >= len(cams):
                continue
            c = cams[cam_id]
            # head pose at t
            hp_q_t, hp_p_t = _interp_quat_pos(hp_t, hp_q, hp_pos, np.array([t]))
            if not np.isfinite(hp_p_t[0, 0]):
                continue
            R_wo, t_wo = pose_attempt_to_xrworld_device(accepted[k], c, hp_q_t[0], hp_p_t[0])
            pos_err_cm = np.linalg.norm(t_wo - pos_query[best]) * 100.0
            q_world = R_to_quat(R_wo)
            ori_err_deg = float(quat_geodesic_deg(q_world, q_query[best]))
            correct[k] = (pos_err_cm < mse_pos_cm) and (ori_err_deg < mse_ori_deg)

    # Map accepted timestamps back to evaluated camera-frame indices. One accept can
    # only score one frame; marking both neighbours inflates recall/F1.
    frame_has_accept = np.zeros(t_query.shape[0], dtype=bool)
    frame_has_correct = np.zeros(t_query.shape[0], dtype=bool)
    for k in range(len(accepted)):
        cand = int(accept_frame_idx[k])
        if cand < 0:
            continue
        frame_has_accept[cand] = True
        if correct[k]:
            frame_has_correct[cand] = True

    # Classification per ref frame
    TP = int(np.sum(in_view & frame_has_correct))
    FN = int(np.sum(in_view & ~frame_has_correct))           # in_view but missed (no accept or wrong)
    FP = int(np.sum(frame_has_accept & ~frame_has_correct))  # accepted but wrong or not actually in-view
    TN = int(np.sum(~in_view & ~frame_has_accept))
    precision = TP / max(TP + FP, 1)
    recall = TP / max(TP + FN, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # Per-cam coverage: of in-view frames, how many had this cam contributing?
    per_cam_coverage = []
    for j in range(len(cams)):
        in_view_this_cam = per_cam_in_view[:, j]
        total = int(in_view_this_cam.sum())
        # How many accepts came from this cam?
        accepts_cam_j = accepted[accepted["cam_id"] == j]
        per_cam_coverage.append({
            "cam": j,
            "in_view_frames": total,
            "accepts_from_cam": int(len(accepts_cam_j)),
            "correct_from_cam": int(sum(correct[i] for i in range(len(accepted)) if accepted[i]["cam_id"] == j)),
        })

    if dump_fn is not None:
        # FN forensics: for every in-view frame without a correct accept, record whether an
        # accept existed (wrong-accept vs no-accept) and how many blobs sit near the projected
        # LED cloud in the best camera — separating "matcher had data and failed" from "nothing
        # detected". Joins on hw_ts_ns + cam_id, the same keys the scorer itself uses.
        bl = G.load_stream(telem, m, "blob")
        bl_t = bl["hw_ts_ns"].astype(np.int64)
        fn_idx = np.where(in_view & ~frame_has_correct)[0]
        recs = []
        for i in fn_idx:
            j = int(np.argmax(led_count[i]))
            u, v = project_device_leds(led_pos, led_nrm, R_ref[i], pos_query[i],
                                       R_head[i], hp_p_at[i], cams[j])
            n_near = 0
            near_bright = float("nan")
            if len(u):
                cu, cv = float(np.mean(u)), float(np.mean(v))
                b = bl[(bl_t == int(t_query[i])) & (bl["cam_id"] == j)]
                if len(b):
                    d = np.hypot(b["x"].astype(float) - cu, b["y"].astype(float) - cv)
                    nb = b[d < 120.0]
                    n_near = int(len(nb))
                    if n_near:
                        near_bright = float(np.median(nb["brightness"]))
            recs.append({
                "t_hw_ns": int(t_query[i]),
                "best_cam": j,
                "leds_in_view": int(led_count[i][j]),
                "had_accept": bool(frame_has_accept[i]),
                "blobs_near_120px": n_near,
                "near_brightness_med": near_bright,
            })
        json.dump(recs, open(dump_fn, "w"))
        print(f"dev{dev}: dumped {len(recs)} FN records -> {dump_fn}", file=sys.stderr)

    return {
        "device": dev,
        "n_ref_frames": int(t_query.shape[0]),
        "n_unknown_frames": int(t_query_all.shape[0] - t_query.shape[0]),
        "n_in_view": int(in_view.sum()),
        "n_accepts": int(len(accepted)),
        "n_accepts_correct": int(correct.sum()),
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse_pos_cm_threshold": mse_pos_cm,
        "mse_ori_deg_threshold": mse_ori_deg,
        "min_leds": min_leds,
        "match_ms": match_ms,
        "max_ref_gap_ms": max_ref_gap_ms,
        "per_cam": per_cam_coverage,
    }


def main():
    ap = argparse.ArgumentParser(description=(
        "Detection F1 for the G2 matcher. Two modes:\n"
        "  default        — score the capture's pose_attempts vs its OWN cleaned-GT\n"
        "                    (self-referential; for INTERNAL consistency only)\n"
        "  --reference X  — score the capture's pose_attempts vs X's cleaned-GT\n"
        "                    (fixed external reference; the correct mode for A/B comparison)"
    ))
    ap.add_argument("capture", help="capture dir whose pose_attempts are scored")
    ap.add_argument("--reference",
                    help="external capture dir whose cleaned-GT becomes the fixed reference. "
                         "OMIT for self-referential scoring (internal consistency); "
                         "SET when comparing different binary/config replays.")
    ap.add_argument("--match-ms", type=float, default=25.0)
    ap.add_argument("--max-ref-gap-ms", type=float, default=150.0,
                    help="max bracketing cleaned-GT gap for scoring camera frames")
    ap.add_argument("--mse-pos-cm", type=float, default=5.0, help="max position error for correctness (cm)")
    ap.add_argument("--mse-ori-deg", type=float, default=15.0, help="max orientation error for correctness (deg)")
    ap.add_argument("--min-leds", type=int, default=3, help="min visible LEDs per cam to count as 'in FOV'")
    ap.add_argument("--cams", default=DEFAULT_CAMS)
    ap.add_argument("--ctrl-left", default=DEFAULT_CTRL_LEFT)
    ap.add_argument("--ctrl-right", default=DEFAULT_CTRL_RIGHT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-fn", default=None,
                    help="dump per-FN-frame forensic records (JSON prefix; -dev{N}.json appended)")
    args = ap.parse_args()

    capture = Path(args.capture)
    cand_telem_dir = capture / "telemetry"
    cams = load_cameras(args.cams)
    if args.reference:
        ref_capture = Path(args.reference)
        ref_telem_dir = ref_capture / "telemetry"
        print(f"reference: {ref_capture} (cleaned-GT built ONCE from this capture's telemetry)")
        print(f"candidate: {capture}     (pose_attempts to be scored)")
    else:
        ref_capture = capture
        ref_telem_dir = cand_telem_dir
        print(f"capture (self-referential): {capture}")
    print(f"Loaded {len(cams)} cameras from {args.cams}")

    results = []
    for dev, ctrl in ((1, args.ctrl_left), (2, args.ctrl_right)):
        r = analyse_device(ref_capture, ref_telem_dir, dev, cams, ctrl,
                           args.mse_pos_cm, args.mse_ori_deg, args.min_leds,
                           args.match_ms, args.max_ref_gap_ms,
                           cand_telem=(cand_telem_dir if args.reference else None),
                           dump_fn=(Path(f"{args.dump_fn}-dev{dev}.json") if args.dump_fn else None))
        if r:
            results.append(r)

    print("\n=== Detection F1 ===")
    print(f"thresholds: pos<{args.mse_pos_cm}cm AND ori<{args.mse_ori_deg}deg => CORRECT")
    print(f"           in-view = ANY cam has >= {args.min_leds} visible LEDs (front-facing + in frame)")
    print(f"           scored camera frames require cleaned-GT bracket gap <= {args.max_ref_gap_ms:.0f}ms")
    print()
    fmt = "{:>10s}  {:>8s}  {:>8s}  {:>8s}  {:>6s}  {:>6s}  {:>6s}  {:>8s}  {:>8s}  {:>8s}"
    print(fmt.format("device", "ref_fr", "unknown", "in_view", "TP", "FP", "FN", "precision", "recall", "F1"))
    print("-" * 95)
    for r in results:
        total_frames = r["n_ref_frames"] + r["n_unknown_frames"]
        print(fmt.format(
            DEVICE_NAMES.get(r["device"], str(r["device"])),
            f"{r['n_ref_frames']}",
            f"{r['n_unknown_frames']} ({100*r['n_unknown_frames']/max(total_frames,1):.0f}%)",
            f"{r['n_in_view']} ({100*r['n_in_view']/max(r['n_ref_frames'],1):.0f}%)",
            f"{r['TP']}", f"{r['FP']}", f"{r['FN']}",
            f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
        ))

    print("\n=== Per-cam coverage ===")
    fmt2 = "{:>10s}  {:>4s}  {:>14s}  {:>16s}  {:>16s}"
    print(fmt2.format("device", "cam", "in_view_frames", "accepts_from_cam", "correct_from_cam"))
    print("-" * 70)
    for r in results:
        for c in r["per_cam"]:
            print(fmt2.format(
                DEVICE_NAMES.get(r["device"], str(r["device"])),
                str(c["cam"]),
                str(c["in_view_frames"]),
                str(c["accepts_from_cam"]),
                str(c["correct_from_cam"]),
            ))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
