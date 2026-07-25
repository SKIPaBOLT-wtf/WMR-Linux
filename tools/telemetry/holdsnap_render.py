#!/usr/bin/env python3
"""holdsnap_render.py -- eyes-on panels for the hold-then-snap episodes holdsnap.py scores.

A hold episode is a claim that the fused report sat somewhere the blobs say it was not, while the
optical pose it refused sat right on them. That claim has to be looked at, not just tabulated, so
this renders every frame of an episode onto the raw short-exposure PGMs:

  cyan o    = detected LED blob (the raw observation, pose-independent)
  green X   = cleaned-GT reference pose projected into this camera
  red +     = the FUSED REPORT (what the user saw)
  magenta . = the FRONT-END OPTICAL pose the fusion was handed that frame

A true hold looks like: magenta and green on the blobs, red floating a metre off. Everything here
is the existing renderer -- gt_fix_render._draw_pose_panel, blob_explain.BlobCache, the
detection_f1 camera model -- driven off holdsnap's episode list instead of the GT-fix verdicts.

Run with the BASE conda (cv2):
  ~/miniconda3/bin/python holdsnap_render.py <capture> --episodes hs.json --replay-out DIR --out DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

import blob_explain as BE
import detection_f1 as DF
import gt_blob_fix as GF
import replay_contract as RC
from gt_fix_render import _draw_pose_panel, _nearest
from headpose_anchor import load_head_pose
from mse_eval import load_candidate_csv
from smooth_ref import build_reference


def _long_index(frames_dir: Path):
    """cam -> sorted [(ts, path)] of long-exposure frames (the controller body is visible there)."""
    idx: dict[int, list[tuple[int, str]]] = {}
    for p in glob.glob(str(frames_dir / "cam*_e300_*.pgm")):
        m = re.match(r"cam(\d+)_t0*(\d+)_", Path(p).name)
        if m:
            idx.setdefault(int(m.group(1)), []).append((int(m.group(2)), p))
    for c in idx:
        idx[c].sort()
    return idx


def _nearest_sample(t_ns, pos, quat, ts, max_ms=25.0):
    if t_ns.shape[0] == 0:
        return None, None
    i = int(np.argmin(np.abs(t_ns.astype(np.int64) - int(ts))))
    if abs(int(t_ns[i]) - int(ts)) > max_ms * 1e6:
        return None, None
    return quat[i], pos[i]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--dev", type=int, choices=(1, 2), required=True)
    ap.add_argument("--episodes", type=Path, required=True, help="holdsnap.py --out JSON")
    ap.add_argument("--replay-out", type=Path, required=True, help="replay out dir with dev*.csv")
    ap.add_argument("--frames", type=Path, default=None, help="override the capture's frames dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="", help="arm name stamped into the banner/filenames")
    ap.add_argument("--held-only", action="store_true", help="skip converging (glide) episodes")
    args = ap.parse_args()
    import cv2

    episodes = []
    for row in json.loads(args.episodes.read_text()):
        if int(row["device_id"]) != args.dev:
            continue
        episodes = [e for e in row["episodes"] if e["held"] or not args.held_only]
    if not episodes:
        print("no episodes to render")
        return 0

    capdir = args.capture
    frames_dir = args.frames or GF._frames_dir(capdir)
    ref = build_reference(capdir / "telemetry", args.dev)
    hp = load_head_pose(capdir / "telemetry")
    g2cam, _ = BE._lazy_imports()
    cams_json = RC.cams_for_capture(capdir)
    cams = DF.load_cameras(str(cams_json))
    g2cams = g2cam.load_cams(cams_json)
    mdl = g2cam.load_led_model(g2cam.CTRL_LEFT if args.dev == 1 else g2cam.CTRL_RIGHT)
    cache = BE.BlobCache(frames_dir, disk_cache=Path("/tmp/holdsnap") / f"blobs_{capdir.name}.npz")
    longi = _long_index(frames_dir)

    csv_path = args.replay_out / f"dev{args.dev}.csv"
    t_pred, pos_pred, quat_pred = load_candidate_csv(csv_path, "pred", valid_only=False)
    t_opt, pos_opt, quat_opt = load_candidate_csv(csv_path, "opt")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ep in episodes:
        kind = "HELD" if ep["held"] else "glide"
        sel = np.where((ref.t_ns >= ep["t_start_ns"]) & (ref.t_ns <= ep["t_end_ns"]) & ref.valid)[0]
        for i in sel:
            ts = int(ref.t_ns[i])
            hpos, hq = GF._head_interp(hp, ts)
            if hpos is None:
                continue
            Rh = DF.quat_to_R(hq)
            q_gt, p_gt = ref.quat[i], ref.pos[i]
            q_pred, p_pred = _nearest_sample(t_pred, pos_pred, quat_pred, ts)
            q_opt, p_opt = _nearest_sample(t_opt, pos_opt, quat_opt, ts)
            if q_pred is None:
                continue
            panels = []
            for cid, cam in enumerate(cams):
                Rcm, tcm = BE._world_to_cam(DF.quat_to_R(q_gt), p_gt, Rh, hpos, cam)
                if int(g2cam.project_model(g2cams[cid], Rcm, tcm, mdl)["visible"].sum()) < BE.MIN_VIS_LEDS:
                    continue
                path, _dt = cache.nearest_frame(cid, ts)
                if path is None:
                    continue
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                blobs, _c = cache.blobs(path)
                pan = _draw_pose_panel(cv2, img, cam, g2cam, g2cams, mdl, blobs,
                                       q_gt, p_gt, q_pred, p_pred, q_opt, p_opt, Rh, hpos)
                cv2.putText(pan, f"cam{cid} blobs={len(blobs)}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                panels.append(pan)
            if not panels:
                continue
            h = max(p.shape[0] for p in panels)
            panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in panels]
            row = np.hstack([np.hstack([p, np.full((h, 5, 3), 60, np.uint8)]) for p in panels])
            lp = _nearest(longi.get(0, []) or sum(longi.values(), []), ts) if longi else None
            if lp is not None:
                limg = cv2.imread(lp[1], cv2.IMREAD_GRAYSCALE)
                if limg is not None:
                    lvis = cv2.resize(cv2.cvtColor(limg, cv2.COLOR_GRAY2BGR), None, fx=2, fy=2,
                                      interpolation=cv2.INTER_NEAREST)
                    lvis = cv2.copyMakeBorder(lvis, 0, max(0, h - lvis.shape[0]), 0, 0, cv2.BORDER_CONSTANT)
                    row = np.hstack([row, lvis[:h]])
            d_pred = float(np.linalg.norm(p_pred - p_gt))
            d_opt = float(np.linalg.norm(p_opt - p_gt)) if p_opt is not None else float("nan")
            hdr = (f"{args.label} {capdir.name} dev{args.dev} {kind} ep@{ep['t_start_ns']/1e9:.4f}s "
                   f"dur={ep['duration_s']:.3f}s | ts={ts} report_err={d_pred*100:.1f}cm "
                   f"optical_err={d_opt*100:.1f}cm")
            banner = np.full((44, row.shape[1], 3), 30, np.uint8)
            cv2.putText(banner, hdr, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(banner, "cyan=blob  greenX=cleaned-GT  red+=FUSED REPORT  magenta=front-end optical",
                        (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
            name = (f"{args.label or 'arm'}_{kind}_{ep['t_start_ns']}_dev{args.dev}_{ts}.png")
            cv2.imwrite(str(args.out / name), np.vstack([banner, row]))
            manifest.append(dict(file=name, kind=kind, episode_start_ns=ep["t_start_ns"], ts=ts,
                                 report_err_cm=d_pred * 100.0, optical_err_cm=d_opt * 100.0))
    cache.save_disk()
    (args.out / f"manifest_dev{args.dev}_{args.label or 'arm'}.json").write_text(json.dumps(manifest, indent=1))
    print(f"rendered {len(manifest)} panels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
