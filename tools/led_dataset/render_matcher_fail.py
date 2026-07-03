#!/usr/bin/env python3
"""Render the matcher-failure GT frames (UNATTRIB + WRONG) for sub-agent visual adjudication.

For each target (split, tag) we draw a side-by-side panel:
  LEFT  short-exposure LED frame (contrast-stretched): green X = annotated GT LED, cyan o = detected blob,
        red + = the committed pose projected into THIS camera (transformed from its committing camera via
        the rigid inter-camera extrinsics), magenta dot = committed-pose LED that is geometrically visible.
  RIGHT paired long-exposure frame (the controller body is visible to the eye) with the GT LEDs marked.

The committed pose's tilt/yaw residual (from candidate.bin) is printed so the adjudicator can see what the
matcher CLAIMED. The question for each frame: is the committed (red +) pose the SAME controller the GT
marks (green X)? If red sits on green -> tracked-on-another-view (GOOD-equivalent). If red is a rotated/
mirrored twin of green -> mirror-flip. If red is elsewhere/garbage -> position-wrong / clutter. If there is
no red and green LEDs are clearly real -> genuine no-commit.

  ~/miniconda3/envs/g2vr/bin/python render_matcher_fail.py --json /tmp/matcher_failures_xform.json \
      --buckets UNATTRIB WRONG NO_COMMIT --out dataset/matcher_failures
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent / "research")); import g2cam
sys.path.insert(0, str(Path(__file__).parent / "../telemetry")); from manifest import Manifest; import g2_geom as G
from prep import stretch
from dump_frames import xform, CTRL

# capture -> (short-exp frames dir, long-exp source). long-exp is euroc mav0 (cam<c>/data/*.png) where
# present, else the sparse e300 long-exposure pgms in the same frames dir.
CAPS = {
    "xv1":      dict(tel="/home/mrwhite0racle/g2-linux-research/results/h6-rmodel-20260612/v2/handgt/xv1",
                     frames="/home/mrwhite0racle/g2-linux-research/captures/20260528-080421-xv-session1/frames",
                     slam="/home/mrwhite0racle/g2-linux-research/captures/20260528-080421-xv-session1/euroc_20260528080506/mav0"),
    "clean":    dict(tel="/home/mrwhite0racle/g2-linux-research/results/h6-rmodel-20260612/v2/handgt/clean2",
                     frames="/home/mrwhite0racle/g2-linux-research/captures/20260526-175615-clean-session2/frames",
                     slam=""),
    "headpose": dict(tel="/home/mrwhite0racle/g2-linux-research/results/h6-rmodel-20260612/v2/handgt/headpose",
                     frames="/home/mrwhite0racle/g2-linux-research/captures/20260524-200416-headpose/frames",
                     slam="/home/mrwhite0racle/g2-linux-research/captures/20260524-200416-headpose/euroc_20260524200507/mav0"),
}


def short_index(frames_dir):
    """cam -> sorted [(ts, path)] for the e20 short-exposure LED frames only."""
    idx = {}
    for p in glob.glob(str(Path(frames_dir) / "cam*_e20_*.pgm")) or glob.glob(str(Path(frames_dir) / "cam*_*.pgm")):
        import re
        m = re.match(r"cam(\d+)_t0*(\d+)_e0*(\d+)_", Path(p).name)
        if m and int(m.group(3)) <= 50:
            idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
    for c in idx:
        idx[c].sort()
    return idx


def long_index(cap):
    """cam -> sorted [(ts, path)] of long-exposure frames (euroc png, else e300 pgm)."""
    idx = {}
    if cap["slam"]:
        for c in range(4):
            fs = glob.glob(str(Path(cap["slam"]) / f"cam{c}/data/*.png"))
            if fs:
                idx[c] = sorted((int(Path(f).stem), f) for f in fs)
    if not idx:
        import re
        for p in glob.glob(str(Path(cap["frames"]) / "cam*_e300_*.pgm")):
            m = re.match(r"cam(\d+)_t0*(\d+)_", Path(p).name)
            if m:
                idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
        for c in idx:
            idx[c].sort()
    return idx


def nearest(arr, ts):
    if not arr:
        return None
    return min(arr, key=lambda a: abs(a[0] - ts))


def load_gt_frame(split, tag):
    ds = Path("dataset/pool") / split
    g = json.load(open(ds / f"{tag}.gt.json"))
    cands = {c["id"]: c for c in json.load(open(ds / f"{tag}.candidates.json"))["candidates"]}
    led = [(cands[i]["cx"], cands[i]["cy"]) for i in g.get("led_ids", []) if i in cands]
    led += [(m["x"], m["y"]) for m in g.get("missed", [])]
    blobs = [(c["cx"], c["cy"]) for c in cands.values()]
    return np.array(led, float).reshape(-1, 2), np.array(blobs, float).reshape(-1, 2), g.get("notes", "")


def committed_uv(cams, models, r, gt_cam):
    """project committed candidate row r into gt_cam via extrinsics; return visible projected uv."""
    dev = int(r["device_id"]); mdl = models[dev]
    Rj = g2cam._quat_to_R(np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float))
    tj = np.array([r["px"], r["py"], r["pz"]], float)
    Ri, ti = (Rj, tj) if int(r["cam_id"]) == gt_cam else xform(Rj, tj, cams[int(r["cam_id"])], cams[gt_cam])
    pm = g2cam.project_model(cams[gt_cam], Ri, ti, mdl)
    return pm["uv"][pm["visible"]], pm["uv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="/tmp/matcher_failures_xform.json")
    ap.add_argument("--buckets", nargs="*", default=["UNATTRIB", "WRONG", "NO_COMMIT"])
    ap.add_argument("--out", default="dataset/matcher_failures")
    ap.add_argument("--eps", type=float, default=5.0)
    args = ap.parse_args()
    rows = [r for r in json.load(open(args.json)) if r["verdict"] in args.buckets]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cams = g2cam.load_cams(); models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}
    short = {s: short_index(c["frames"]) for s, c in CAPS.items()}
    longi = {s: long_index(c) for s, c in CAPS.items()}
    tels = {}
    manifest = []
    for r in sorted(rows, key=lambda r: (r["verdict"], r["split"], -(max(r["tilt_deg"] or 0, r["yaw_deg"] or 0)))):
        split, tag, cam, ts = r["split"], r["tag"], r["cam"], r["ts"]
        sarr = short[split].get(cam, [])
        hit = nearest(sarr, ts)
        if hit is None:
            continue
        sts, sp = hit
        img = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        led, blobs, notes = load_gt_frame(split, tag)
        vis = cv2.resize(cv2.cvtColor(stretch(img), cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for (x, y) in blobs:
            cv2.circle(vis, (int(x * 2), int(y * 2)), 7, (0, 200, 200), 1)
        # committed pose projection (red +) if there is a commit attributed to this frame
        red_uv = np.zeros((0, 2)); red_all = np.zeros((0, 2))
        if r["dev"] is not None:
            if split not in tels:
                tel = Path(CAPS[split]["tel"]) / "telemetry"
                m = Manifest.load(tel)
                tels[split] = G.load_stream(tel, m, "candidate")
            cand = tels[split]
            sel = cand[(cand["selected"] == 1) & (cand["device_id"] == r["dev"])]
            sts_ns = sel["t_mono_ns"].astype(np.int64)
            j = int(np.argmin(np.abs(sts_ns - ts))) if len(sel) else -1
            if j >= 0 and abs(int(sts_ns[j]) - ts) <= 8_000_000:
                red_uv, red_all = committed_uv(cams, models, sel[j], cam)
        for (x, y) in red_all:
            cv2.circle(vis, (int(x * 2), int(y * 2)), 2, (200, 0, 200), -1)
        for (x, y) in red_uv:
            cv2.drawMarker(vis, (int(x * 2), int(y * 2)), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
        for (x, y) in led:
            cv2.drawMarker(vis, (int(x * 2), int(y * 2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        cc = f"<-cam{r['commit_cam']}" if (r["commit_cam"] is not None and r["commit_cam"] != cam) else ""
        cv2.putText(vis, f"{split}/{tag} {r['verdict']} dev{r['dev']} cam{cam}{cc} GT={len(led)} "
                    f"expl={r['explained']} tilt={r['tilt_deg']} yaw={r['yaw_deg']}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "greenX=GT LED  cyan_o=blob  red+=committed pose (proj)  magenta=committed-LED visible",
                    (4, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        # long-exposure panel
        lhit = nearest(longi[split].get(cam, []), ts)
        if lhit is not None:
            lts, lp = lhit
            lim = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
            if lim is not None:
                lv = cv2.resize(cv2.cvtColor(lim, cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                for (x, y) in led:
                    cv2.drawMarker(lv, (int(x * 2), int(y * 2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
                cv2.putText(lv, f"LONG-EXP cam{cam} dt={(lts - ts) / 1e6:+.0f}ms (greenX=GT LED)", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
                if lv.shape[0] != vis.shape[0]:
                    lv = cv2.copyMakeBorder(lv, 0, max(0, vis.shape[0] - lv.shape[0]), 0, 0, cv2.BORDER_CONSTANT)
                vis = np.hstack([vis, np.full((vis.shape[0], 6, 3), 60, np.uint8), lv[:vis.shape[0]]])
        name = f"{r['verdict']}_{split}_{tag}.png"
        cv2.imwrite(str(out / name), vis)
        manifest.append(dict(file=name, split=split, tag=tag, cam=cam, ts=ts, verdict=r["verdict"],
                             dev=r["dev"], commit_cam=r["commit_cam"], explained=r["explained"],
                             tilt_deg=r["tilt_deg"], yaw_deg=r["yaw_deg"], n_true=r["n_true"],
                             n_proj_vis=r["n_proj_vis"], reproj=r["reproj"], notes=notes))
    json.dump(manifest, open(out / "render_manifest.json", "w"), indent=1)
    print(f"rendered {len(manifest)} frames -> {out}")


if __name__ == "__main__":
    main()
