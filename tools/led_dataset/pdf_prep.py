#!/usr/bin/env python3
"""Categorize controller-visible GT frames into the 3 user categories, render each (short-exp + overlays +
long-exp SLAM panel), and emit metadata for the PDF. Attribution-robust (cross-cam transform).
  1 MATCHED-WELL    : committed pose explains >=0.6 of GT LEDs
  2 MATCHED-WRONG   : a pose committed for this controller (explained in [0.05,0.6)) but off/flipped
  3 NOT-MATCHED     : no committed pose for this controller (explained <0.05); controller present, few/no LEDs
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent / "research")); import g2cam
sys.path.insert(0, str(Path(__file__).parent / "../telemetry")); from manifest import Manifest; import g2_geom as G
from prep import stretch, frame_index
import matcher_failure as MF

CTRL = {1: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K1111630014L.json",
        2: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K5091930012R.json"}
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--telemetry", type=Path, required=True,
                 help="replay telemetry dir for the xv1 split (the dir holding manifest.json)")
_args = _ap.parse_args()

_HERE = Path(__file__).resolve().parent
DS = _HERE / "dataset/xv1"; OUT = _HERE / "dataset/pdf_samples"
CAP = Path("/home/mrwhite0racle/g2-linux-research/captures/20260528-080421-xv-session1")
FRAMES = CAP / "frames"; SLAM = CAP / "euroc_20260528080506/mav0"; TELDIR = _args.telemetry
cams = g2cam.load_cams(MF.CAMS_JSON); models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}

def load_gt():
    out = {}
    for gf in glob.glob(str(DS / "*.gt.json")):
        g = json.load(open(gf)); tag = g["tag"]
        if not g.get("controller_visible"): continue
        allc = json.load(open(DS / f"{tag}.candidates.json"))["candidates"]; byid = {c["id"]: c for c in allc}
        led = [(byid[i]["cx"], byid[i]["cy"]) for i in g.get("led_ids", []) if i in byid] + [(m["x"], m["y"]) for m in g.get("missed", [])]
        if not led: continue
        out[tag] = dict(led=np.array(led, float), blobs=np.array([(c["cx"], c["cy"]) for c in allc], float).reshape(-1, 2),
                        cam=int(tag.split("_")[0][3:]), ts=int(tag.split("_")[1]), n_ctrl=g.get("n_controllers", 1), notes=g.get("notes", ""))
    return out

def xform(Rj, tj, cj, ci):
    Rij = ci.R_imu_cam.T @ cj.R_imu_cam; tij = ci.R_imu_cam.T @ (cj.t_imu_cam - ci.t_imu_cam); return Rij @ Rj, Rij @ tj + tij
def pose_uv(ci, r):
    mdl = models[int(r["device_id"])]; Rj = g2cam._quat_to_R(np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float)); tj = np.array([r["px"], r["py"], r["pz"]], float)
    Ri, ti = xform(Rj, tj, cams[int(r["cam_id"])], cams[ci]); pm = g2cam.project_model(cams[ci], Ri, ti, mdl); return pm["uv"][pm["visible"]]
def expl(uv, led, eps=5.0):
    return 0.0 if (len(uv) == 0 or len(led) == 0) else float((np.hypot(led[:, None, 0] - uv[None, :, 0], led[:, None, 1] - uv[None, :, 1]).min(1) <= eps).mean())

m = Manifest.load(TELDIR); cand = G.load_stream(TELDIR, m, "candidate"); fr = G.load_stream(TELDIR, m, "frame")
fidx = frame_index(FRAMES); slam = {c: sorted((int(Path(f).stem), f) for f in glob.glob(str(SLAM / f"cam{c}/data/*.png"))) for c in range(4)}
gt = load_gt()

def detected_blobs(cam, ts):  # production (blobwatch_v2) blob count for this cam+ts
    mk = (fr["cam_id"] == cam) & (np.abs(fr["hw_ts_ns"].astype(np.int64) - ts) < 8_000_000)
    return int(fr["n_blobs"][mk].max()) if mk.any() else -1

rows = []
for tag, g in gt.items():
    tsw = np.abs(cand["t_mono_ns"].astype(np.int64) - g["ts"]) < 8_000_000
    asel = cand[tsw & (cand["selected"] == 1)]
    union = np.vstack([pose_uv(g["cam"], r) for r in asel]) if len(asel) else np.zeros((0, 2))
    e = expl(union, g["led"]); det = detected_blobs(g["cam"], g["ts"])
    cat = "MATCHED-WELL" if e >= 0.6 else ("MATCHED-WRONG" if e >= 0.05 else "NOT-MATCHED")
    rows.append(dict(tag=tag, cam=g["cam"], ts=g["ts"], n_gt=len(g["led"]), det=det, expl=round(e, 2),
                     cat=cat, n_ctrl=g["n_ctrl"], union=union, g=g))

def render(r):
    fts, fp = min(fidx[r["cam"]], key=lambda a: abs(a[0] - r["ts"])); img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    vis = cv2.resize(cv2.cvtColor(stretch(img), cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for (x, y) in r["g"]["blobs"]: cv2.circle(vis, (int(x*2), int(y*2)), 7, (0, 200, 200), 1)
    for (x, y) in r["union"]: cv2.drawMarker(vis, (int(x*2), int(y*2)), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
    for (x, y) in r["g"]["led"]: cv2.drawMarker(vis, (int(x*2), int(y*2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
    cv2.putText(vis, f"{r['cat']} | {r['tag']} | GTleds={r['n_gt']} detected={r['det']} explained={r['expl']}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, "greenX=GT LED  cyan_o=detected blob  red+=matcher committed pose", (4, vis.shape[0]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    arr = slam.get(r["cam"], [])
    if arr:
        sts, sp = min(arr, key=lambda a: abs(a[0] - r["ts"])); sim = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        sv = cv2.resize(cv2.cvtColor(sim, cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for (x, y) in r["g"]["led"]: cv2.drawMarker(sv, (int(x*2), int(y*2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(sv, f"LONG-EXP SLAM cam{r['cam']} dt={(sts-r['ts'])/1e6:+.0f}ms (controller GT context)", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        h = max(vis.shape[0], sv.shape[0]); vis = np.hstack([vis, np.full((h, 6, 3), 60, np.uint8), sv[:h]])
    op = OUT / f"{r['cat']}_{r['tag']}.png"; cv2.imwrite(str(op), vis); return op.name

import shutil; shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)
# n_ctrl=1 only -> unambiguous categories (a 2-controller frame with one tracked reads as explained~0.5,
# which is NOT a wrong pose). WELL: clean (det~=GT, no clutter storm), spread over LED count. WRONG: clear
# flips with >=5 LEDs (a wrong pose is only convincing when enough LEDs were available). NOT-MATCHED: fewest
# detected first (the genuine 'too few/no LEDs' cases).
solo = [r for r in rows if r["n_ctrl"] == 1]
well = sorted([r for r in solo if r["cat"] == "MATCHED-WELL" and r["n_gt"] >= 5 and r["det"] <= r["n_gt"] + 2], key=lambda r: r["n_gt"])
wrong = sorted([r for r in solo if r["cat"] == "MATCHED-WRONG" and r["n_gt"] >= 5], key=lambda r: r["expl"])
notm = sorted([r for r in solo if r["cat"] == "NOT-MATCHED"], key=lambda r: (r["det"], r["n_gt"]))
picks = {"MATCHED-WELL": well[::max(1, len(well)//8)][:8], "MATCHED-WRONG": wrong[:8], "NOT-MATCHED": notm[::max(1, len(notm)//9)][:9]}
meta = []
for cat, items in picks.items():
    for r in items:
        nm = render(r); meta.append(dict(cat=cat, png=nm, tag=r["tag"], n_gt=r["n_gt"], det=r["det"], expl=r["expl"], n_ctrl=r["n_ctrl"], notes=r["g"]["notes"][:80]))
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
from collections import Counter
print("category totals:", dict(Counter(r["cat"] for r in rows)))
print("picked:", {k: len(v) for k, v in picks.items()})
for cat in picks:
    ex = [m for m in meta if m["cat"] == cat]
    print(f"  {cat}: " + "; ".join(f"{m['tag'].split('_')[0]}_{str(m['tag'].split('_')[1])[-6:]}(GT{m['n_gt']},det{m['det']},e{m['expl']})" for m in ex[:6]))
