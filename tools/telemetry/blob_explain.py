#!/usr/bin/env python3
"""blob_explain.py -- the INDEPENDENT multi-camera blob-explain arbiter for the cleaned-GT.

WHY
---
The cleaned-GT (smooth_ref/deflip) is built only from the controller's OWN recorded optical
poses, so it is self-referential: where the recorded stream lost lock and sat on the wrong
yaw/position branch, the cleaned-GT inherits that wrong branch (the metric-audit fixed the TILT
branch; the YAW/POSITION branch is the residual corruption -- see RECALL-PUSH-STRATEGY.md). To
correct the GT without making it validate the tracker by construction, we need an arbiter that is
INDEPENDENT of any pose decision. The raw observation is the DETECTED LED BLOBS in each camera
frame -- they exist regardless of which pose the tracker committed. A pose is ground-truth-
consistent only if it reprojects ONTO those blobs.

THE CIRCULARITY TRAP (and how this avoids it)
---------------------------------------------
A single camera CANNOT disambiguate a mirror-twin: the near-planar LED ring's flipped twin
reprojects onto the SAME real blobs at sub-pixel error (measured: genuine flip poses match ~7
real cluster blobs at 0.7px median -- /tmp/flip_blob_investigation.json). So single-image reproj
is NOT a sufficient arbiter. This module therefore requires MULTI-CAMERA 3D consistency: a pose
is "blob-confirmed" only if it lands on the detected blobs in >=2 co-visible cameras. A twin that
fits one camera's image sits at the wrong 3D depth/heading and FAILS the second camera. Where only
one camera sees the controller (true sparse-depth ambiguity), the arbiter returns AMBIGUOUS and the
GT is left untouched (or marked ABSTAIN) rather than guessed. The head-pose anchor (independent
SLAM) is a second arbiter for the orientation branch.

WHAT IT PROVIDES
----------------
  BlobCache(frames_dir)              -- per-camera frame index + memoized blob detection (cv2).
  explain_pose(...)                  -- multi-cam blob-explain score for ONE world pose.
  compare_branches(...)              -- which of two world poses the blobs prefer + margin/verdict.

Geometry is the source-of-truth path: world<-camera via detection_f1.pose_flip_YZ/pose_mul/pose_inv
(the exact chain the live tracker + scorer use), LED projection via g2cam.project_model
(byte-faithful rt8 + the pose_metrics visibility/facing test). Blob detection via the SAME
prep.detect_candidates that produced the hand-adjudicated GT.

cv2 lives in the BASE conda (~/miniconda3/bin/python), not g2vr. Run analyses that need this
module with that interpreter.
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import detection_f1 as DF
import g2_geom as G

# g2cam + prep live under tools/led_dataset; the caller is responsible for putting them on the path
# (they pull in cv2). Imported lazily so this module loads under g2vr for type/contract use too.
_G2CAM = None
_PREP = None


def _lazy_imports():
    global _G2CAM, _PREP
    if _G2CAM is None:
        # g2cam + prep live under tools/led_dataset (alongside the hand-GT pipeline that produced
        # the blob detector). Put them on the path here so callers need not.
        import sys
        led = Path(__file__).resolve().parent.parent / "led_dataset"
        for p in (str(led / "research"), str(led)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import g2cam  # noqa: E402
        from prep import detect_candidates, cluster_and_flag  # noqa: E402
        _G2CAM = g2cam
        _PREP = (detect_candidates, cluster_and_flag)
    return _G2CAM, _PREP


# --- arbiter tuning (few knobs; each justified) ---
GATE_PX = 6.0          # LED->blob match acceptance radius (px); ~ the matcher's led_radius scale
MIN_VIS_LEDS = 4       # a camera "co-sees" the controller when >=4 model LEDs project in-frame
                       # (the pose_metrics 4-LED PnP floor; the depth-resolving unit)
CONFIRM_CAMS = 2       # multi-cam 3D-confirm threshold: a pose is blob-confirmed on >=2 cameras
                       # ( rules out a single-image depth/heading-wrong twin fit)
MATCH_FRAC_OK = 0.45   # pooled fraction of projected-visible LEDs landing on a blob for a pose to
                       # be blob-confirmed. Data-calibrated (probe_calib): a CORRECT pose pools
                       # ~0.52 median (p10 ~0.37), a WRONG branch ~0.15 (p90 ~0.37); 0.45 admits
                       # ~75-82% of correct poses while rejecting ~95-98% of wrong branches.
                       # NOT 1.0: dim-LED detection misses ~40% and the facing cone over-counts
                       # geometrically-visible-but-occluded LEDs, so a correct pose is never 100%.
MATCH_FRAC_GATE = 0.30 # below this a camera does not contribute a confirmation vote
REPROJ_OK_PX = 2.0     # a camera confirms only if its matched LEDs reproject this tightly.
                       # Data-calibrated (probe_reproj): among >=2-cam-confirmed poses, CORRECT
                       # poses pool to <=1.5px (p90) while a WRONG branch pools to >=3.1px (p25) --
                       # a clean gap at ~2px with NO overlap. Reprojection tightness is the genuine
                       # multi-cam 3D-consistency signal: a depth/heading-wrong twin cannot reproject
                       # tightly across two co-visible cameras at once (it fits one image only).
MARGIN_FRAC = 0.12     # branch-comparison margin on pooled match-fraction when reproj is a tie
REPROJ_MARGIN_PX = 1.0 # branch-comparison margin on pooled reproj: the tighter pose wins only if it
                       # beats the other by this many px (else AMBIGUOUS -- honest abstention)
FRAME_MATCH_MS = 8.0   # pose-time <-> short-exposure frame join window (one frame is ~33ms; the
                       # short-exp LED frames are interleaved, nearest within 8ms is the same instant)

# branch-comparison verdicts
B_A = 1        # pose A (e.g. cleaned-GT) explains the blobs better
B_B = -1       # pose B (the alternative branch) explains the blobs better
B_TIE = 0      # both explain equally well within the margin (true sparse ambiguity) -> ABSTAIN
B_NEITHER = 2  # neither pose lands on the blobs (no usable evidence) -> ABSTAIN


@dataclass
class CamExplain:
    cam_id: int
    n_visible: int          # model LEDs projecting in-frame (front-facing)
    n_matched: int          # of those, how many land on a detected blob within the gate
    match_frac: float       # n_matched / n_visible
    median_reproj_px: float # median LED->blob distance over the matched set (NaN if none)
    n_blobs: int            # detected blobs in this frame
    frame_dt_ms: float      # how far the joined frame was from the pose time


@dataclass
class ExplainResult:
    n_cams_covisible: int   # cameras seeing >=MIN_VIS_LEDS model LEDs in-frame
    n_cams_confirm: int     # cameras where match_frac >= gate AND median reproj <= REPROJ_OK_PX
    pooled_visible: int
    pooled_matched: int
    pooled_match_frac: float
    pooled_reproj_px: float # mean over confirming cameras of their median LED->blob distance (NaN if none)
    per_cam: list = field(default_factory=list)

    @property
    def blob_confirmed(self) -> bool:
        """Multi-cam 3D-consistent: lands TIGHTLY on the detected blobs in >=CONFIRM_CAMS cameras.
        Requires both a strong pooled match-fraction AND tight pooled reprojection. This is what
        rules out a single-image depth/heading-wrong twin: such a twin can match a similar number
        of blobs in one image but cannot reproject tightly across two co-visible cameras at once."""
        return (self.n_cams_confirm >= CONFIRM_CAMS and self.pooled_match_frac >= MATCH_FRAC_OK
                and np.isfinite(self.pooled_reproj_px) and self.pooled_reproj_px <= REPROJ_OK_PX)


@dataclass
class BranchVerdict:
    verdict: int            # B_A / B_B / B_TIE / B_NEITHER
    a: ExplainResult
    b: ExplainResult
    margin: float           # pooled_match_frac(winner) - pooled_match_frac(loser)
    note: str = ""


class BlobCache:
    """Per-capture short-exposure frame index + memoized blob detection.

    Detecting blobs on a PGM is the cv2 cost; we memoize per file path so re-querying the same
    frame across many poses / both branches is free. Detection is the SAME prep.detect_candidates +
    cluster_and_flag that produced the hand-adjudicated GT, so the blobs this arbiter uses are
    exactly the raw observation the annotators judged."""

    def __init__(self, frames_dir, max_exp: int = 50, disk_cache: "Path | str | None" = None):
        self.frames_dir = Path(frames_dir)
        self._index = self._build_index(self.frames_dir, max_exp)
        self._blob_cache: dict[str, np.ndarray] = {}
        self._cluster_cache: dict[str, np.ndarray] = {}
        # optional on-disk persistence (blob detection over thousands of PGMs is the cost; reuse it
        # across the GT-fix pass and the render pass). Keyed by frame basename.
        self._disk = Path(disk_cache) if disk_cache else None
        if self._disk and self._disk.is_file():
            d = np.load(self._disk, allow_pickle=True)
            self._blob_cache = {k: v for k, v in d["blobs"].item().items()}
            self._cluster_cache = {k: v for k, v in d["cluster"].item().items()}

    def save_disk(self):
        if self._disk:
            self._disk.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self._disk, blobs=self._blob_cache, cluster=self._cluster_cache)

    @staticmethod
    def _build_index(frames_dir: Path, max_exp: int):
        idx: dict[int, list[tuple[int, str]]] = {}
        for p in glob.glob(str(frames_dir / "cam*_*.pgm")):
            m = re.match(r"cam(\d+)_t0*(\d+)_e0*(\d+)_", Path(p).name)
            if m and int(m.group(3)) <= max_exp:
                idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
        for c in idx:
            idx[c].sort()
        return {c: (np.array([t for t, _ in arr]), [p for _, p in arr]) for c, arr in idx.items()}

    def nearest_frame(self, cam: int, ts_ns: int, max_dt_ms: float = FRAME_MATCH_MS):
        """(path, dt_ms) of the short-exposure frame nearest ts_ns on `cam`, else (None, inf)."""
        if cam not in self._index:
            return None, float("inf")
        ks, paths = self._index[cam]
        if ks.size == 0:
            return None, float("inf")
        j = int(np.searchsorted(ks, ts_ns))
        best_i, best_d = -1, None
        for c in (j - 1, j, j + 1):
            if 0 <= c < ks.size:
                d = abs(int(ks[c]) - int(ts_ns))
                if best_d is None or d < best_d:
                    best_d, best_i = d, c
        if best_i < 0 or best_d > max_dt_ms * 1e6:
            return None, float("inf")
        return paths[best_i], best_d / 1e6

    def blobs(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """(blobs_xy (M,2), in_cluster (M,)) for a frame path, memoized."""
        if path not in self._blob_cache:
            import cv2
            detect_candidates, cluster_and_flag = _lazy_imports()[1]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._blob_cache[path] = np.zeros((0, 2))
                self._cluster_cache[path] = np.zeros((0,), bool)
            else:
                kept, _flags = cluster_and_flag(detect_candidates(img))
                self._blob_cache[path] = np.array([[c["cx"], c["cy"]] for c in kept],
                                                   float).reshape(-1, 2)
                self._cluster_cache[path] = np.array(
                    [bool(c.get("in_cluster", False)) for c in kept], bool)
        return self._blob_cache[path], self._cluster_cache[path]


def _world_to_cam(R_xr_dev, t_xr_dev, R_xr_hmd, t_xr_hmd, cam: DF.Camera):
    """World(OpenXR)<-device + world<-hmd  ->  camera<-device (R_cam_model, t_cam_model).
    The exact inverse of detection_f1.pose_attempt_to_xrworld_device; uses the same flip_YZ chain
    the live tracker uses, so the projection matches what the camera actually saw."""
    R_cv_hmd, t_cv_hmd = DF.pose_flip_YZ(R_xr_hmd, t_xr_hmd)
    R_cv_model, t_cv_model = DF.pose_flip_YZ(R_xr_dev, t_xr_dev)
    R_cv_cam, t_cv_cam = DF.pose_mul(R_cv_hmd, t_cv_hmd, cam.P_imu_cam_R, cam.P_imu_cam_t)
    R_cam_cv, t_cam_cv = DF.pose_inv(R_cv_cam, t_cv_cam)
    return DF.pose_mul(R_cam_cv, t_cam_cv, R_cv_model, t_cv_model)


def _gnn_match(uv: np.ndarray, blobs: np.ndarray, gate: float) -> tuple[int, list[float]]:
    """Greedy one-to-one nearest-blob assignment of projected LEDs to blobs within `gate`.
    Returns (n_matched, [reproj_px...]). Mirrors flip_blob_investigate.gnn_match."""
    if uv.shape[0] == 0 or blobs.shape[0] == 0:
        return 0, []
    cands = []
    for li in range(uv.shape[0]):
        d = np.hypot(blobs[:, 0] - uv[li, 0], blobs[:, 1] - uv[li, 1])
        for bi in range(blobs.shape[0]):
            if d[bi] <= gate:
                cands.append((d[bi], li, bi))
    cands.sort()
    led_used, blob_used, ds = set(), set(), []
    for dd, li, bi in cands:
        if li in led_used or bi in blob_used:
            continue
        led_used.add(li)
        blob_used.add(bi)
        ds.append(dd)
    return len(ds), ds


def explain_pose(R_xr_dev, t_xr_dev, R_xr_hmd, t_xr_hmd, ts_ns: int,
                 cams: list[DF.Camera], g2cams, led_model, cache: BlobCache,
                 gate: float = GATE_PX) -> ExplainResult:
    """Multi-camera blob-explain for ONE world-frame controller pose at time ts_ns.

    For every camera in which the pose projects >=MIN_VIS_LEDS model LEDs in-frame, detect the
    blobs on that camera's nearest short-exposure frame and GNN-match the projected-visible LEDs
    onto them. Pool the matched/visible counts across cameras. The pose is blob-confirmed iff it
    lands on the blobs in >=CONFIRM_CAMS cameras (multi-cam 3D consistency)."""
    g2cam, _ = _lazy_imports()
    per_cam: list[CamExplain] = []
    pooled_vis = pooled_match = 0
    n_covis = n_confirm = 0
    confirm_reproj: list[float] = []
    for cid, cam in enumerate(cams):
        Rcm, tcm = _world_to_cam(R_xr_dev, t_xr_dev, R_xr_hmd, t_xr_hmd, cam)
        pm = g2cam.project_model(g2cams[cid], Rcm, tcm, led_model)
        vis = pm["visible"]
        nv = int(vis.sum())
        if nv < MIN_VIS_LEDS:
            continue
        n_covis += 1
        path, dt_ms = cache.nearest_frame(cid, ts_ns)
        if path is None:
            continue
        blobs, _cluster = cache.blobs(path)
        uv = pm["uv"][vis]
        nm, ds = _gnn_match(uv, blobs, gate)
        frac = nm / max(nv, 1)
        med_reproj = float(np.median(ds)) if ds else float("nan")
        per_cam.append(CamExplain(cam_id=cid, n_visible=nv, n_matched=nm, match_frac=frac,
                                  median_reproj_px=med_reproj,
                                  n_blobs=int(blobs.shape[0]), frame_dt_ms=dt_ms))
        pooled_vis += nv
        pooled_match += nm
        # a camera confirms only with a strong match AND tight reprojection (3D-consistent)
        if frac >= MATCH_FRAC_GATE and np.isfinite(med_reproj) and med_reproj <= REPROJ_OK_PX:
            n_confirm += 1
            confirm_reproj.append(med_reproj)
    return ExplainResult(
        n_cams_covisible=n_covis,
        n_cams_confirm=n_confirm,
        pooled_visible=pooled_vis,
        pooled_matched=pooled_match,
        pooled_match_frac=(pooled_match / pooled_vis) if pooled_vis else 0.0,
        pooled_reproj_px=float(np.mean(confirm_reproj)) if confirm_reproj else float("nan"),
        per_cam=per_cam,
    )


def blob_refine_pose(q_seed, p_seed, R_xr_hmd, t_xr_hmd, ts_ns: int,
                     cams: list[DF.Camera], g2cams, led_model, cache: BlobCache,
                     gate: float = GATE_PX):
    """INDEPENDENT blob-driven pose correction: refine a world-frame seed pose onto the DETECTED LED
    blobs via per-camera PnP, using ONLY the blobs + the LED model (NO tracker/candidate pose). The
    seed (typically the cleaned-GT) sets the branch neighbourhood; solvePnP then snaps the pose onto
    the actual blobs. Returns (q_world, p_world, n_corr) for the camera with the most LED<->blob
    correspondences, else (None, None, 0).

    This is the arbiter rule (a)'s constructive form: it asks "what rigid controller pose do the raw
    blobs actually support, near here?" -- the answer is chosen by the blobs, not by any tracker, so
    using it to witness GT corruption is non-circular. The caller MUST re-verify the refined pose
    with explain_pose (multi-cam blob_confirmed) before trusting it -- a single-camera PnP alone can
    still latch a mirror twin; the >=2-camera confirmation is what rejects that."""
    import cv2
    g2cam, _ = _lazy_imports()
    R_seed = DF.quat_to_R(q_seed)
    best = None
    for cid, cam in enumerate(cams):
        Rcm, tcm = _world_to_cam(R_seed, p_seed, R_xr_hmd, t_xr_hmd, cam)
        pm = g2cam.project_model(g2cams[cid], Rcm, tcm, led_model)
        vis = pm["visible"]
        if int(vis.sum()) < MIN_VIS_LEDS:
            continue
        path, _dt = cache.nearest_frame(cid, ts_ns)
        if path is None:
            continue
        blobs, _cluster = cache.blobs(path)
        if blobs.shape[0] < MIN_VIS_LEDS:
            continue
        uv = pm["uv"]
        cands = []
        for li in np.where(vis)[0]:
            d = np.hypot(blobs[:, 0] - uv[li, 0], blobs[:, 1] - uv[li, 1])
            k = int(np.argmin(d))
            if d[k] <= gate:
                cands.append((d[k], int(li), k))
        cands.sort()
        led_used, blob_used, obj, img = set(), set(), [], []
        for _d, li, k in cands:
            if li in led_used or k in blob_used:
                continue
            led_used.add(li)
            blob_used.add(k)
            obj.append(led_model.pos[li])
            img.append(blobs[k])
        if len(obj) < MIN_VIS_LEDS:
            continue
        K, D = g2cam.K_D_for_opencv(g2cams[cid])
        rvec, _ = cv2.Rodrigues(Rcm)
        tvec = tcm.reshape(3, 1)
        ok, rvec, tvec = cv2.solvePnP(np.asarray(obj, np.float64), np.asarray(img, np.float64),
                                      K, D, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        Rref, _ = cv2.Rodrigues(rvec)
        tref = tvec.reshape(3)
        # cam<-model refined -> world, via the exact scorer chain (so it lives in the same world frame)
        qcm = DF.R_to_quat(Rref)
        row = {"qx": qcm[0], "qy": qcm[1], "qz": qcm[2], "qw": qcm[3],
               "px": tref[0], "py": tref[1], "pz": tref[2]}
        R_wo, t_wo = DF.pose_attempt_to_xrworld_device(row, cams[cid], DF.R_to_quat(R_xr_hmd), t_xr_hmd)
        if best is None or len(obj) > best[2]:
            best = (DF.R_to_quat(R_wo), t_wo, len(obj))
    return best if best else (None, None, 0)


def compare_branches(R_a, t_a, R_b, t_b, R_xr_hmd, t_xr_hmd, ts_ns: int,
                     cams, g2cams, led_model, cache: BlobCache,
                     gate: float = GATE_PX, margin: float = MARGIN_FRAC) -> BranchVerdict:
    """Which world pose (A or B) the DETECTED blobs prefer, across all co-visible cameras.

    Decision (anti-circular -- uses only the raw blobs, pose-independent):
      * if neither pose is blob-confirmed (multi-cam)            -> B_NEITHER (abstain).
      * if exactly one is blob-confirmed                          -> that one (A or B).
      * if both are confirmed: prefer the higher pooled match-frac if it beats the other by
        >= margin; otherwise B_TIE (genuine sparse-depth ambiguity -> abstain).
    Multi-cam confirmation is what makes this safe: a mirror twin that fits one image fails the
    co-visible second camera, so it does not become "confirmed"."""
    a = explain_pose(R_a, t_a, R_xr_hmd, t_xr_hmd, ts_ns, cams, g2cams, led_model, cache, gate)
    b = explain_pose(R_b, t_b, R_xr_hmd, t_xr_hmd, ts_ns, cams, g2cams, led_model, cache, gate)
    ca, cb = a.blob_confirmed, b.blob_confirmed
    if not ca and not cb:
        return BranchVerdict(B_NEITHER, a, b, 0.0, "neither_confirmed")
    if ca and not cb:
        return BranchVerdict(B_A, a, b, a.pooled_match_frac - b.pooled_match_frac, "only_A_confirmed")
    if cb and not ca:
        return BranchVerdict(B_B, a, b, b.pooled_match_frac - a.pooled_match_frac, "only_B_confirmed")
    # both blob-confirmed -> the cleaner 3D fit wins. Reprojection tightness is the decisive
    # multi-cam signal (a wrong branch matches a similar blob COUNT but reprojects loosely), so
    # test it FIRST; fall back to the match-fraction margin only when reproj genuinely ties.
    dr = b.pooled_reproj_px - a.pooled_reproj_px   # >0 => A is tighter (better)
    if abs(dr) >= REPROJ_MARGIN_PX:
        return (BranchVerdict(B_A, a, b, dr, "both_confirmed_A_tighter") if dr > 0
                else BranchVerdict(B_B, a, b, -dr, "both_confirmed_B_tighter"))
    df = a.pooled_match_frac - b.pooled_match_frac
    if abs(df) < margin:
        return BranchVerdict(B_TIE, a, b, abs(df), "both_confirmed_tie")
    return (BranchVerdict(B_A, a, b, df, "both_confirmed_A_wins") if df > 0
            else BranchVerdict(B_B, a, b, -df, "both_confirmed_B_wins"))
