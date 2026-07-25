#!/usr/bin/env python3
"""Independent POSE-correctness check: does the matcher's pose actually explain the GROUND-TRUTH LEDs?
For each controller-visible GT frame, take the matcher's SELECTED pose (telemetry candidate.bin) for that
(cam, ts), project the device's real 32-LED model, and measure how well the projected VISIBLE LEDs overlap
the annotated GT LED positions. A correct pose projects onto the real LEDs; a flipped/wrong pose does not —
regardless of the matcher's self-reported reprojection error. This answers "is the match accurate" without
trusting the matcher.

  ~/miniconda3/envs/g2vr/bin/python pose_eval.py dataset/xv1 --capture <capture_dir>
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "research"))
import g2cam
sys.path.insert(0, str(Path(__file__).parent / "../telemetry"))
from matcher_failure import CAMS_JSON
from manifest import Manifest
import g2_geom as G

CTRL = {1: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K1111630014L.json",
        2: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K5091930012R.json"}

def load_gt(ds_dir: Path):
    out = {}
    for gf in glob.glob(str(ds_dir / "*.gt.json")):
        g = json.load(open(gf)); tag = g["tag"]
        if not g.get("controller_visible"):
            continue
        cf = ds_dir / f"{tag}.candidates.json"
        if not cf.exists():
            continue
        cands = {c["id"]: c for c in json.load(open(cf))["candidates"]}
        leds = [(cands[i]["cx"], cands[i]["cy"]) for i in g.get("led_ids", []) if i in cands]
        leds += [(m["x"], m["y"]) for m in g.get("missed", [])]
        if not leds:
            continue
        out[tag] = dict(led=np.array(leds, float), cam=int(tag.split("_")[0][3:]),
                        ts=int(tag.split("_")[1]), n_ctrl=g.get("n_controllers", 1))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--capture", required=True)
    ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--good-frac", type=float, default=0.6, help="min frac of GT LEDs explained for a GOOD verdict")
    args = ap.parse_args()
    ds = Path(args.dataset); tel = Path(args.capture) / "telemetry"
    cams = g2cam.load_cams(CAMS_JSON)
    models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}
    gt = load_gt(ds)
    if not gt:
        print("No controller-visible GT yet in", ds); return
    m = Manifest.load(tel); cand = G.load_stream(tel, m, "candidate")
    sel = cand[(cand["selected"] == 1)] if "selected" in cand.dtype.names else cand[cand["outcome"] == 1]
    verdict = {"GOOD": 0, "FLIP/WRONG": 0, "NO_POSE": 0}
    rows = []
    for tag, g in sorted(gt.items()):
        # selected poses for this cam near this frame ts (hw_ts)
        cm = sel[(sel["cam_id"] == g["cam"]) & (np.abs(sel["t_mono_ns"].astype(np.int64) - g["ts"]) < 8_000_000)]
        if len(cm) == 0:
            verdict["NO_POSE"] += 1; rows.append((tag, "NO_POSE", 0.0, g["n_ctrl"])); continue
        # The GT led_ids/missed of a multi-controller frame are the UNION of every visible controller's LEDs
        # (the annotator does not tag which controller each blob is). Each committed device pose explains only
        # its own controller, so explanation must be measured against the UNION of all committed poses'
        # projected LEDs, not the best single device — otherwise a frame with both controllers perfectly
        # tracked caps at ~0.5 (each covers half) and is mislabelled FLIP/WRONG.
        proj = []
        for r in cm:  # every selected pose this frame (one per committed device)
            dev = int(r["device_id"]); model = models.get(dev)
            if model is None:
                continue
            q = np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float)
            R = g2cam._quat_to_R(q); t = np.array([r["px"], r["py"], r["pz"]], float)
            pm = g2cam.project_model(cams[g["cam"]], R, t, model)
            vis = pm["uv"][pm["visible"]]
            if len(vis):
                proj.append(vis)
        uv = np.vstack(proj) if proj else np.zeros((0, 2))
        # fraction of GT LEDs explained by ANY committed pose's projected visible model LED within eps
        if len(uv):
            D = np.hypot(g["led"][:, None, 0] - uv[None, :, 0], g["led"][:, None, 1] - uv[None, :, 1])
            best = (D.min(axis=1) <= args.eps).mean() if D.size else 0.0
        else:
            best = 0.0
        v = "GOOD" if best >= args.good_frac else "FLIP/WRONG"
        verdict[v] += 1; rows.append((tag, v, round(best, 2), g["n_ctrl"]))
    n = len(gt)
    print(f"=== POSE-vs-GT ({n} controller-visible frames, eps={args.eps}px, good>={args.good_frac}) ===")
    for k, v in verdict.items():
        print(f"  {k:12} {v:4}  ({100*v/n:.0f}%)")
    wrong = [r for r in rows if r[1] == "FLIP/WRONG"]
    print(f"  -- worst-explained FLIP/WRONG frames (matcher pose does NOT hit the real LEDs): --")
    for tag, v, frac, nc in sorted(wrong, key=lambda r: r[2])[:12]:
        print(f"     {tag}  explained={frac}  n_ctrl={nc}")

if __name__ == "__main__":
    main()
