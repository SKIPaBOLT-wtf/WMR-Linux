#!/usr/bin/env python3
"""Generate a per-frame LED mask file for the realistic-OOV replay mode.

For one device, project the reference pose's LED model into every camera at every frame
group and emit blackout circles covering (a) each visible projected LED and (b) every
detected blob within ASSOC_R of any projection (robust to reference error). The replay
harness (G2_REPLAY_LED_MASK_FILE + G2_REPLAY_DROP_DEVICE) blacks these out of the camera
images inside drop windows, so the target controller goes optically dark while frames,
clutter, and the other controller keep flowing -- real out-of-view, not transport blackout.

Output: <capture>/led_mask_dev<N>.csv with rows "t_ns,cam,x,y,r".
Run with the BASE conda (cv2 deps via blob_explain): ~/miniconda3/bin/python make_led_mask.py <capture> --dev 1
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import numpy as np

import detection_f1 as DF
import blob_explain as BE
import gt_blob_fix as GF
from headpose_anchor import load_head_pose
from tracking_metrics import build_reference_grid, load_cameras

LED_R = 14.0   # px blackout radius per projected LED (covers smear/defocus)
ASSOC_R = 50.0 # px: detected blobs this close to any projection are masked too

CTRLS = {1: Path.home() / ".config/monado/wmr/controller_A85K1111630014L.json",
         2: Path.home() / ".config/monado/wmr/controller_A85K5091930012R.json"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture", type=Path)
    ap.add_argument("--dev", type=int, choices=(1, 2), required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    g2cam, _ = BE._lazy_imports()
    cams = DF.load_cameras(g2cam.HMD_CAMERAS.as_posix())
    tm_cams = load_cameras(Path(g2cam.HMD_CAMERAS.as_posix()))
    g2cams = g2cam.load_cams()
    mdl = g2cam.load_led_model(g2cam.CTRL_LEFT if args.dev == 1 else g2cam.CTRL_RIGHT)
    hp = load_head_pose(args.capture / "telemetry")
    cache = BE.BlobCache(GF._frames_dir(args.capture),
                         disk_cache=Path("/tmp/gtfix") / f"blobcache_{args.capture.name}.npz")
    grid = build_reference_grid(args.capture, args.dev, tm_cams, str(CTRLS[args.dev]),
                                max_ref_gap_ms=150.0, head_match_ms=100.0,
                                detect_leds=4, pose_leds=4, high_leds=6, reference_qc="none")

    out_path = args.out or (args.capture / f"led_mask_dev{args.dev}.csv")
    n_frames = n_covered = n_circles = 0
    with out_path.open("w") as out:
        out.write("t_ns,cam,x,y,r\n")
        for i, t_ns in enumerate(grid.frames.t_ns):
            n_frames += 1
            if not grid.ref_valid[i]:
                continue
            hpos, hq = GF._head_interp(hp, int(t_ns))
            if hpos is None:
                continue
            Rh = DF.quat_to_R(hq)
            Rd = DF.quat_to_R(grid.ref_quat[i])
            covered = False
            for cid, cam in enumerate(cams):
                Rcm, tcm = BE._world_to_cam(Rd, grid.ref_pos[i], Rh, hpos, cam)
                pm = g2cam.project_model(g2cams[cid], Rcm, tcm, mdl)
                uv = pm["uv"][pm["visible"]]
                if uv.shape[0] == 0:
                    continue
                covered = True
                for (x, y) in uv:
                    out.write(f"{int(t_ns)},{cid},{x:.1f},{y:.1f},{LED_R:.0f}\n")
                    n_circles += 1
                path, _dt = cache.nearest_frame(cid, int(t_ns))
                if path is None:
                    continue
                blobs, _c = cache.blobs(path)
                for (bx, by) in blobs:
                    if min(math.hypot(bx - x, by - y) for (x, y) in uv) <= ASSOC_R:
                        out.write(f"{int(t_ns)},{cid},{bx:.1f},{by:.1f},{LED_R:.0f}\n")
                        n_circles += 1
            n_covered += covered
    print(f"{out_path}: {n_circles} circles, coverage {n_covered}/{n_frames} frame groups "
          f"({100.0 * n_covered / max(1, n_frames):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
