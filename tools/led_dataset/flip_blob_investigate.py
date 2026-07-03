#!/usr/bin/env python3
"""STEP 1 — clutter-vs-real investigation on the residual committed mirror-flips.

For each genuine FLIP frame (adjudication.json), in the COMMITTING camera:
  1. Take the committed flip pose (candidate.bin selected=1 for that device) = P_cam_obj.
  2. Detect blobs on the committing camera's short-exposure raw frame (prep.detect_candidates,
     the SAME detector that produced the hand-GT candidates.json) + density-cluster them.
  3. GNN-match the flip's projected LED model to those blobs (matcher gate: led_radius ellipse,
     mutual exclusion) -> the flip's matched blob set + per-blob reprojection.
  4. Project the OTHER controller's committed pose into the same camera; GNN-match it.
  5. Classify each flip-matched blob:
       2ND_CTRL   coincides with the other device's projected LED (within gate)
       CLUSTER    part of a dense bright cluster (real LED of the tracked controller)
       ISOLATED   isolated bright spot, not in any cluster, not 2nd-ctrl  -> clutter/noise
  6. Build the gravity-correct twin counterpart (re-tilt the flip to camera-frame gravity)
     WHEN camera-frame gravity is available (head pose + cam extrinsics), GNN-match it, and
     report how many of the flip's matched blobs the correct twin ALSO claims (SHARED = a real
     LED both twins want = geometric ambiguity).

Outputs a per-frame table and an aggregate split. Run in the BASE conda (has cv2):
  PYTHONNOUSERSITE=1 ~/miniconda3/bin/python tools/led_dataset/flip_blob_investigate.py \
      --out /tmp/flip_blob_investigation.json
"""
from __future__ import annotations
import argparse, glob, json, math, re, sys
from pathlib import Path
import numpy as np, cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "research"))
sys.path.insert(0, str(HERE / "../telemetry"))
import g2cam                                   # noqa: E402
from manifest import Manifest                  # noqa: E402
import g2_geom as G                            # noqa: E402
from prep import detect_candidates, cluster_and_flag  # noqa: E402

CTRL = {1: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K1111630014L.json",
        2: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K5091930012R.json"}

CAPS = {
    "xv1":      dict(tel="/tmp/wtfix_20260528-080421-xv-session1/telemetry",
                     frames="/home/mrwhite0racle/g2-linux-research/captures/20260528-080421-xv-session1/frames"),
    "clean":    dict(tel="/tmp/wtfix_20260526-175615-clean-session2/telemetry",
                     frames="/home/mrwhite0racle/g2-linux-research/captures/20260526-175615-clean-session2/frames"),
    "headpose": dict(tel="/tmp/wtfix_20260524-200416-headpose/telemetry",
                     frames="/home/mrwhite0racle/g2-linux-research/captures/20260524-200416-headpose/frames"),
}

# Matcher gate (pose_metrics.c): led_radius_px ellipse + a blob-too-large cull (>4x radius). We use a
# fixed gate radius in px; the matcher's led_radius_px is depth-derived but is ~led-size px (small). The
# project-and-match here is for PROVENANCE classification, so a fixed reasonable gate is appropriate.
GATE_PX = 6.0   # match acceptance radius (px); ~ the matcher's led_radius scale at typical depth


def short_index(frames_dir):
    idx = {}
    for p in glob.glob(str(Path(frames_dir) / "cam*_*.pgm")):
        m = re.match(r"cam(\d+)_t0*(\d+)_e0*(\d+)_", Path(p).name)
        if m and int(m.group(3)) <= 50:
            idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
    for c in idx:
        idx[c].sort()
    return idx


def nearest_path(idx, cam, ts):
    if cam not in idx or not idx[cam]:
        return None
    arr = idx[cam]
    ks = [t for t, _ in arr]
    j = int(np.searchsorted(ks, ts))
    best = None
    for c in (j - 1, j, j + 1):
        if 0 <= c < len(arr):
            d = abs(arr[c][0] - ts)
            if best is None or d < best[0]:
                best = (d, arr[c][1])
    return best[1] if best and best[0] < 8_000_000 else None  # 8 ms


def gnn_match(proj_uv, visible, blobs_xy, gate=GATE_PX):
    """Global one-to-one greedy assignment of visible projected LEDs to blobs within the gate.
    Returns matched_blob_idx[led] (-1 if none) and the list of matched blob indices."""
    cands = []
    vidx = np.nonzero(visible)[0]
    for li in vidx:
        u, v = proj_uv[li]
        for bi, (bx, by) in enumerate(blobs_xy):
            d2 = (u - bx) ** 2 + (v - by) ** 2
            if d2 <= gate * gate:
                cands.append((d2, li, bi))
    cands.sort()
    led_used, blob_used = set(), set()
    matched = {}
    for d2, li, bi in cands:
        if li in led_used or bi in blob_used:
            continue
        led_used.add(li); blob_used.add(bi)
        matched[int(li)] = (int(bi), math.sqrt(d2))
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", default=str(HERE / "dataset/matcher_failures/adjudication.json"))
    ap.add_argument("--out", default="/tmp/flip_blob_investigation.json")
    ap.add_argument("--gate", type=float, default=GATE_PX)
    args = ap.parse_args()

    adj = json.load(open(args.adj))
    flips = [a for a in adj if a.get("adj") == "FLIP" and a.get("genuine")]

    cams = g2cam.load_cams()
    models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}

    # per-capture telemetry caches
    cand_cache, frame_idx_cache = {}, {}

    def load_cand(split):
        if split not in cand_cache:
            tel = Path(CAPS[split]["tel"])
            m = Manifest.load(tel)
            cand_cache[split] = G.load_stream(tel, m, "candidate")
        return cand_cache[split]

    def load_fidx(split):
        if split not in frame_idx_cache:
            frame_idx_cache[split] = short_index(CAPS[split]["frames"])
        return frame_idx_cache[split]

    results = []
    for a in flips:
        split, dev, cc, ts = a["split"], int(a["dev"]), int(a["commit_cam"]), int(a["ts"])
        rec = dict(split=split, dev=dev, commit_cam=cc, ts=ts, tilt_deg=a.get("tilt_deg"),
                   yaw_deg=a.get("yaw_deg"), flip_conf=a.get("flip_conf"), reproj=a.get("reproj"))

        cand = load_cand(split)
        hw = cand["t_mono_ns"].astype(np.int64)
        sel = cand[(hw >= ts - 1) & (hw <= ts + 1) & (cand["selected"] == 1) & (cand["device_id"] == dev)]
        if len(sel) == 0:
            rec["err"] = "no_selected_candidate"
            results.append(rec); continue
        r = sel[0]
        rec["cam_id"] = int(r["cam_id"]); rec["c_matched"] = int(r["blobs_matched"])
        rec["c_unmatched"] = int(r["unmatched_blobs"]); rec["c_reproj_px"] = round(float(r["reproj_err_px"]), 2)
        cam_committed = int(r["cam_id"])

        # detect blobs on the committing camera's frame
        fidx = load_fidx(split)
        fp = nearest_path(fidx, cam_committed, ts)
        if fp is None:
            rec["err"] = f"no_frame cam{cam_committed}"
            results.append(rec); continue
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        cands_raw = detect_candidates(img)
        kept, flags = cluster_and_flag(cands_raw)
        blobs_xy = np.array([[c["cx"], c["cy"]] for c in kept], float).reshape(-1, 2)
        in_cluster = np.array([bool(c.get("in_cluster", False)) for c in kept], bool)
        rec["n_blobs"] = len(kept); rec["n_cluster_blobs"] = int(in_cluster.sum())
        rec["frame_degenerate"] = bool(flags["degenerate"])

        # project the committed flip pose into its committing camera
        Rj = g2cam._quat_to_R(np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float))
        tj = np.array([r["px"], r["py"], r["pz"]], float)
        pm = g2cam.project_model(cams[cam_committed], Rj, tj, models[dev])
        flip_match = gnn_match(pm["uv"], pm["visible"], blobs_xy, args.gate)
        flip_blobs = {bi for (bi, _) in flip_match.values()}
        rec["flip_matched_blobs"] = len(flip_blobs)
        rec["flip_match_reproj_px"] = round(float(np.mean([d for (_, d) in flip_match.values()])), 2) if flip_match else None

        # project the OTHER controller's committed pose (2nd-controller test)
        other = 2 if dev == 1 else 1
        sel_o = cand[(hw >= ts - 1) & (hw <= ts + 1) & (cand["selected"] == 1) & (cand["device_id"] == other)]
        other_blobs = set()
        if len(sel_o):
            ro = sel_o[0]
            # the other controller may have committed on a different cam; transform into cam_committed
            from dump_frames import xform
            Ro = g2cam._quat_to_R(np.array([ro["qx"], ro["qy"], ro["qz"], ro["qw"]], float))
            to = np.array([ro["px"], ro["py"], ro["pz"]], float)
            oc = int(ro["cam_id"])
            if oc == cam_committed:
                Roi, toi = Ro, to
            else:
                Roi, toi = xform(Ro, to, cams[oc], cams[cam_committed])
            pmo = g2cam.project_model(cams[cam_committed], Roi, toi, models[other])
            om = gnn_match(pmo["uv"], pmo["visible"], blobs_xy, args.gate)
            other_blobs = {bi for (bi, _) in om.values()}
        rec["other_ctrl_matched_blobs"] = len(other_blobs)

        # classify each flip-matched blob
        n_2nd = n_cluster = n_isolated = 0
        for bi in flip_blobs:
            if bi in other_blobs:
                n_2nd += 1
            elif bi < len(in_cluster) and in_cluster[bi]:
                n_cluster += 1
            else:
                n_isolated += 1
        rec["flip_blob_2nd_ctrl"] = n_2nd
        rec["flip_blob_cluster"] = n_cluster
        rec["flip_blob_isolated"] = n_isolated
        results.append(rec)

    json.dump(results, open(args.out, "w"), indent=1)

    # aggregate
    ok = [r for r in results if "flip_matched_blobs" in r]
    print(f"flip frames: {len(flips)}  with-data: {len(ok)}  errored: {len(results)-len(ok)}")
    errs = {}
    for r in results:
        if "err" in r:
            errs[r["err"].split()[0]] = errs.get(r["err"].split()[0], 0) + 1
    print("errors:", errs)
    tot_m = sum(r["flip_matched_blobs"] for r in ok)
    tot_2 = sum(r["flip_blob_2nd_ctrl"] for r in ok)
    tot_c = sum(r["flip_blob_cluster"] for r in ok)
    tot_i = sum(r["flip_blob_isolated"] for r in ok)
    print(f"\nALL flip-matched blobs (n={tot_m}):")
    if tot_m:
        print(f"  2ND_CTRL  {tot_2:4d}  ({100*tot_2/tot_m:.1f}%)")
        print(f"  CLUSTER   {tot_c:4d}  ({100*tot_c/tot_m:.1f}%)  (real controller LED)")
        print(f"  ISOLATED  {tot_i:4d}  ({100*tot_i/tot_m:.1f}%)  (clutter/noise)")
    # per cell
    print("\nper (split,dev):")
    cells = sorted({(r["split"], r["dev"]) for r in ok})
    print(f"  {'cell':<16} {'frames':>6} {'mblob':>6} {'2nd':>5} {'clus':>5} {'isol':>5} {'reproj':>7}")
    for s, d in cells:
        rs = [r for r in ok if r["split"] == s and r["dev"] == d]
        m = sum(r["flip_matched_blobs"] for r in rs)
        rp = [r["flip_match_reproj_px"] for r in rs if r["flip_match_reproj_px"] is not None]
        print(f"  {s+'/dev'+str(d):<16} {len(rs):>6} {m:>6} "
              f"{sum(r['flip_blob_2nd_ctrl'] for r in rs):>5} "
              f"{sum(r['flip_blob_cluster'] for r in rs):>5} "
              f"{sum(r['flip_blob_isolated'] for r in rs):>5} "
              f"{(np.mean(rp) if rp else float('nan')):>7.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
