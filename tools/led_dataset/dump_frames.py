#!/usr/bin/env python3
"""Dump controller frames + what the matcher predicted vs the independent GT, into a markdown doc.
Attribution-ROBUST: a controller pose committed on ANY camera is transformed (via inter-camera extrinsics)
into the GT-annotated camera and projected, so 'untracked' means NO committed pose explains these GT LEDs
(not merely 'no commit recorded on this camera'). Tests honestly whether the untracked frames are sparse-LED.

Per frame overlay: green X = GT LED, cyan o = detector candidate blob, red + = matcher's best committed pose
(projected/transformed into this camera), orange + = best generated-but-rejected candidate (when untracked).

  ~/miniconda3/envs/g2vr/bin/python dump_frames.py dataset/xv1 --capture /tmp/all3_tel --frames <frames>
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent / "research")); import g2cam
sys.path.insert(0, str(Path(__file__).parent / "../telemetry")); from manifest import Manifest; import g2_geom as G
from prep import stretch, frame_index

CTRL = {1: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K1111630014L.json",
        2: "/home/mrwhite0racle/.config/monado/wmr/controller_A85K5091930012R.json"}

def load_gt(ds):
    out = {}
    for gf in glob.glob(str(ds / "*.gt.json")):
        g = json.load(open(gf)); tag = g["tag"]
        if not g.get("controller_visible"): continue
        cf = ds / f"{tag}.candidates.json"
        if not cf.exists(): continue
        allc = json.load(open(cf))["candidates"]; byid = {c["id"]: c for c in allc}
        led = [(byid[i]["cx"], byid[i]["cy"]) for i in g.get("led_ids", []) if i in byid]
        led += [(m["x"], m["y"]) for m in g.get("missed", [])]
        if not led: continue
        out[tag] = dict(led=np.array(led, float), allblobs=np.array([(c["cx"], c["cy"]) for c in allc], float).reshape(-1, 2),
                        cam=int(tag.split("_")[0][3:]), ts=int(tag.split("_")[1]), n_ctrl=g.get("n_controllers", 1),
                        notes=g.get("notes", ""))
    return out

def xform(R_cj, t_cj, cj, ci):
    """pose object<-cam_j  ->  object<-cam_i, via imu extrinsics. X_cami = Rij@X_camj + tij."""
    Rij = ci.R_imu_cam.T @ cj.R_imu_cam
    tij = ci.R_imu_cam.T @ (cj.t_imu_cam - ci.t_imu_cam)
    return Rij @ R_cj, Rij @ t_cj + tij

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--capture", required=True); ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default="investigate"); ap.add_argument("--eps", type=float, default=5.0)
    ap.add_argument("--slam", default="", help="euroc mav0 dir with long-exposure SLAM frames (cam<C>/data/<ts>.png)")
    args = ap.parse_args()
    slam_idx = {}
    if args.slam:
        for c in range(4):
            fs = glob.glob(str(Path(args.slam) / f"cam{c}/data/*.png"))
            slam_idx[c] = sorted((int(Path(f).stem), f) for f in fs)
    def slam_panel(cam, ts, led, h):
        arr = slam_idx.get(cam, [])
        if not arr: return None
        sts, sp = min(arr, key=lambda a: abs(a[0] - ts)); im = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if im is None: return None
        v = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR); v = cv2.resize(v, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for (x, y) in led: cv2.drawMarker(v, (int(x*2), int(y*2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        cv2.putText(v, f"LONG-EXP SLAM cam{cam} dt={(sts-ts)/1e6:+.0f}ms (greenX=GT LED here too)", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        if v.shape[0] != h: v = cv2.copyMakeBorder(v, 0, max(0, h - v.shape[0]), 0, 0, cv2.BORDER_CONSTANT)
        return v[:h]
    ds = Path(args.dataset); out = ds.parent / args.out
    import shutil; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    cams = g2cam.load_cams(); models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}
    gt = load_gt(ds); tel = Path(args.capture) / "telemetry"; m = Manifest.load(tel); cand = G.load_stream(tel, m, "candidate")
    fidx = frame_index(Path(args.frames))

    def pose_uv(cam_i, r):
        """project committed candidate r (in cam r.cam_id) into cam_i (transformed)."""
        dev = int(r["device_id"]); mdl = models.get(dev)
        if mdl is None: return np.zeros((0, 2))
        cj = cams[int(r["cam_id"])]; ci = cams[cam_i]
        Rj = g2cam._quat_to_R(np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float))
        tj = np.array([r["px"], r["py"], r["pz"]], float)
        Ri, ti = xform(Rj, tj, cj, ci)
        pm = g2cam.project_model(ci, Ri, ti, mdl); return pm["uv"][pm["visible"]]

    def explained(uv, led):
        if len(uv) == 0 or len(led) == 0: return 0.0
        D = np.hypot(led[:, None, 0] - uv[None, :, 0], led[:, None, 1] - uv[None, :, 1])
        return float((D.min(axis=1) <= args.eps).mean())

    rows = []
    for tag, g in gt.items():
        tswin = np.abs(cand["t_mono_ns"].astype(np.int64) - g["ts"]) < 8_000_000
        allsel = cand[tswin & (cand["selected"] == 1)]            # committed poses this ts, any cam, any controller
        rej = cand[tswin & (cand["cam_id"] == g["cam"]) & (cand["selected"] == 0)]
        # UNION of every committed pose's projected LEDs (handles 2-controller frames: GT LEDs are the union
        # of both controllers', so a single pose only explains ~half — must score against all committed poses).
        uvs = [pose_uv(g["cam"], r) for r in allsel]
        union_uv = np.vstack([u for u in uvs if len(u)]) if any(len(u) for u in uvs) else np.zeros((0, 2))
        best = explained(union_uv, g["led"]); best_uv = union_uv
        best_cam = sorted(set(int(r["cam_id"]) for r in allsel)) if len(allsel) else None
        verdict = "GOOD" if best >= 0.6 else ("TRACKED-WRONG" if best >= 0.2 else "UNTRACKED")
        best_rej = rej[np.argmin(rej["total_cost"])] if len(rej) else None
        rows.append(dict(tag=tag, cam=g["cam"], ts=g["ts"], n_gt=len(g["led"]), n_blobs=len(g["allblobs"]),
                         verdict=verdict, explained=round(best, 2), best_uv=best_uv, best_cam=best_cam,
                         n_commits=len(allsel), best_rej=best_rej, g=g))

    def render(r):
        arr = fidx.get(r["cam"], [])
        if not arr: return None
        fts, fp = min(arr, key=lambda a: abs(a[0] - r["ts"])); img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        vis = cv2.resize(cv2.cvtColor(stretch(img), cv2.COLOR_GRAY2BGR), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for (x, y) in r["g"]["allblobs"]: cv2.circle(vis, (int(x*2), int(y*2)), 7, (0, 200, 200), 1)
        if len(r["best_uv"]):
            for (x, y) in r["best_uv"]: cv2.drawMarker(vis, (int(x*2), int(y*2)), (0, 0, 255), cv2.MARKER_CROSS, 13, 2)
        elif r["best_rej"] is not None:
            for (x, y) in pose_uv(r["cam"], r["best_rej"]): cv2.drawMarker(vis, (int(x*2), int(y*2)), (0, 165, 255), cv2.MARKER_CROSS, 12, 1)
        for (x, y) in r["g"]["led"]: cv2.drawMarker(vis, (int(x*2), int(y*2)), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 14, 2)
        pc = f" pred-from-cam{r['best_cam']}" if (len(r['best_uv']) and r['best_cam'] != r['cam']) else ""
        cv2.putText(vis, f"{r['tag']} | GTleds={r['n_gt']} blobs={r['n_blobs']} | {r['verdict']} explained={r['explained']}{pc}",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, "greenX=GT LED  cyan_o=detected blob  red+=matcher committed(projected)  orange+=rejected cand",
                    (4, vis.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        sp = slam_panel(r["cam"], r["ts"], r["g"]["led"], vis.shape[0])
        if sp is not None:
            vis = np.hstack([vis, np.full((vis.shape[0], 6, 3), 60, np.uint8), sp])
        op = out / f"{r['verdict']}_{r['tag']}.png"; cv2.imwrite(str(op), vis); return op.name

    unt = sorted([r for r in rows if r["verdict"] == "UNTRACKED"], key=lambda r: r["n_gt"])
    good = sorted([r for r in rows if r["verdict"] == "GOOD"], key=lambda r: r["n_gt"])
    wrong = [r for r in rows if r["verdict"] == "TRACKED-WRONG"]
    pick = {"UNTRACKED — sparsest (fewest GT LEDs)": unt[:5],
            "UNTRACKED — richest (MOST GT LEDs; refutes 'sparse' if many)": unt[-6:],
            "GOOD — correct/accepted (sample by LED count)": good[::max(1, len(good)//6)][:6],
            "TRACKED-WRONG (flip/imperfect)": wrong[:5]}
    nG, nW, nU = sum(r['verdict']=='GOOD' for r in rows), sum(r['verdict']=='TRACKED-WRONG' for r in rows), sum(r['verdict']=='UNTRACKED' for r in rows)
    md = ["# G2 matcher predictions vs independent GT — frames for manual review\n",
          f"Telemetry: g2-all3 (winner tracking), attribution-robust (pose committed on any cam transformed into the GT cam).\n",
          f"On {len(rows)} controller-visible GT frames: **GOOD {nG} ({100*nG//len(rows)}%)**, TRACKED-WRONG {nW}, **UNTRACKED {nU} ({100*nU//len(rows)}%)**.\n",
          "Legend: **green X**=GT LED, **cyan o**=detector candidate blob, **red +**=matcher's committed pose projected "
          "(transformed from whichever camera committed), **orange +**=best generated-but-rejected candidate. "
          "`explained`=fraction of GT LEDs the matcher's pose lands on (>=0.6 GOOD, 0.2-0.6 wrong, <0.2 untracked).\n",
          "TEST OF THE CLAIM: the UNTRACKED frames should be sparse (few GT LEDs / few blobs). See the 'richest UNTRACKED' "
          "section — if those have many LEDs the matcher missed, the 'sparse-LED floor' claim is wrong.\n"]
    for section, items in pick.items():
        md.append(f"\n## {section} ({len(items)})\n\n| tag | GT LEDs | blobs | commits@ts | verdict | explained | notes |\n|---|---|---|---|---|---|---|\n")
        names = []
        for r in items:
            names.append(render(r)); note = (r["g"]["notes"][:50].replace("|", " ")) if r["g"]["notes"] else ""
            extra = (f" rej_cost={float(r['best_rej']['total_cost']):.1f}" if (r["verdict"]=="UNTRACKED" and r["best_rej"] is not None) else (" no-hypothesis" if r["verdict"]=="UNTRACKED" else ""))
            md.append(f"| {r['tag']} | {r['n_gt']} | {r['n_blobs']} | {r['n_commits']} | {r['verdict']} | {r['explained']} | {note}{extra} |\n")
        for nm in names:
            if nm: md.append(f"\n![{nm}]({nm})\n")
    (out / "MATCHER_FRAMES.md").write_text("".join(md))
    print(f"wrote {out}/MATCHER_FRAMES.md + {len(list(out.glob('*.png')))} imgs")
    if unt:
        ng = np.array([r["n_gt"] for r in unt])
        print(f"UNTRACKED ({len(unt)}): GT-LED count median={np.median(ng):.0f} max={int(ng.max())} | frames with >=6 GT LEDs: {(ng>=6).sum()}/{len(ng)}")
    print(f"GOOD={nG} TRACKED-WRONG={nW} UNTRACKED={nU}")

if __name__ == "__main__":
    main()
