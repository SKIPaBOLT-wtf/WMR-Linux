#!/usr/bin/env python3
"""Cross-reference the agent-annotated GT against the current-best tracker telemetry to separate
RECOVERABLE MATCHER failures from true DETECTION failures and physical sub-4 frames.

For every controller-visible GT frame (cam, hw_ts) we know, from the ANNOTATION (never the matcher):
  * whether the controller is present and HOW MANY of its LEDs a careful viewer can see
    (n_true = |led_ids| + |missed|), and their image positions.
ID is POSSIBLE iff n_true >= MIN_PNP (a 4-point PnP is solvable from this camera alone).

From the tracker telemetry (candidate.bin SELECTED rows = the committed poses, joint multi-cam PnP
commits exactly ONE pose per device per frame-time from ONE camera) we determine the matcher's
outcome at this frame-time and CREDIT IT CROSS-CAMERA: a pose committed via cam Y is transformed
object<-cam_Y -> object<-GT-cam_X purely through the rigid inter-camera IMU extrinsics (head-pose-free;
the HMD pose is common to both cameras and cancels) and its visible model LEDs are matched to the GT LED
positions. explained = fraction of GT LEDs a committed pose lands on (<=EPS px).

Per (GT frame, device that owns this controller's LEDs) verdict:
  GOOD          committed AND explained >= GOOD_FRAC
  WRONG         committed AND explained <  GOOD_FRAC  (flip / clutter-corrupted pose)
  NO_COMMIT     no SELECTED pose for this device within MATCH_NS of the frame-time
  ABSTAIN       outside the telemetry window, or the device was never bracketed/present
                (warm-up / pre-track tail / device genuinely absent) -> no matcher opportunity

Failure bucket (the deliverable split):
  MATCHER_FAIL  ID possible (n_true>=MIN_PNP) AND verdict in {NO_COMMIT, WRONG}   <- recoverable
  DETECT_LIMITED  n_true < MIN_PNP  AND verdict in {NO_COMMIT, WRONG}             <- physical sub-4
  SUCCESS       verdict GOOD
  ABSTAIN       no opportunity

Wrong-pose MODE for WRONG: tilt-flip vs yaw-flip via candidate.bin tilt_err_rad / yaw_err_rad
(gravity-anchored => tilt reliable), plus a CLUTTER signal (unmatched_blobs high / had_twin).
NO_COMMIT MODE: CLUTTER_FLOOD (frame blob count >> n_true => over-detection drowned the matcher) vs
NO_HYPOTHESIS (matcher generated nothing for this device near this time).

Usage:
  matcher_failure.py --telemetry-root results/redo-20260723/current-handgt-a
  matcher_failure.py --telemetry-root <root> --json out.json
"""
from __future__ import annotations
import argparse, glob, json, sys, re
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "research"))
import g2cam
sys.path.insert(0, str(Path(__file__).parent / "../telemetry"))
from manifest import Manifest
from replay_contract import PINNED_CAMS
import g2_geom as G
from dump_frames import xform   # cross-cam pose transform via the rigid inter-camera extrinsics
# xform is head-pose-free: a pose object<-cam_j is mapped to object<-cam_i purely through the rigid
# HMD inter-camera IMU extrinsics. The HMD head pose is common to both cameras (one rigid body) and so
# cancels in cam_j -> world -> cam_i; the old head-pose path differed by ~1e-13 and is removed.

MIN_PNP = 4               # >= this many true LEDs in a camera => identification is geometrically possible
GOOD_FRAC = 0.6           # fraction of GT LEDs a committed pose must explain to be GOOD
EPS_PX = 5.0              # GT-LED <-> projected-model-LED match radius
MATCH_NS = 8_000_000      # a commit "at" this frame-time
BRACKET_NS = 250_000_000  # device "present" if bracketed by commits within +-this on both sides
FLIP_DEG = 15.0           # tilt/yaw residual above which a wrong-pose axis is dominant

CTRL = {1: g2cam.CTRL_LEFT, 2: g2cam.CTRL_RIGHT}
# Every split is a pre-provenance May-2026 capture, so the pinned config IS the capture-era
# one. Scoring must never read the ~/.config copy the live driver rewrites at session start
# (tools/telemetry/replay_contract.py).
CAMS_JSON = PINNED_CAMS
# Replay-output dirs holding each split's telemetry/candidate.bin (regenerate by running
# offline_vio_replay with G2_REPLAY_TELEMETRY=<dir>/telemetry on the capture).
SPLITS = {
    "xv1": "xv1",
    "clean": "clean2",
    "headpose": "headpose",
}


def split_paths(root) -> dict:
    """Resolve SPLITS against an explicit replay root -> {split: Path}. The single place that knows
    a split's directory name, so no tool carries a baked-in path to a dated result dir."""
    base = Path(root)
    return {split: base / subdir for split, subdir in SPLITS.items()}


def load_gt(ds_dir: Path):
    """controller-visible GT frames -> dict(tag -> {cam, ts, n_ctrl, led_xy (N,2), n_true})."""
    out = {}
    for gf in glob.glob(str(ds_dir / "*.gt.json")):
        g = json.load(open(gf))
        if not g.get("controller_visible"):
            continue
        tag = g["tag"]
        cf = ds_dir / f"{tag}.candidates.json"
        if not cf.exists():
            continue
        cands = {c["id"]: c for c in json.load(open(cf))["candidates"]}
        leds = [(cands[i]["cx"], cands[i]["cy"]) for i in g.get("led_ids", []) if i in cands]
        leds += [(m["x"], m["y"]) for m in g.get("missed", [])]
        if not leds:
            continue
        out[tag] = dict(cam=int(tag.split("_")[0][3:]), ts=int(tag.split("_")[1]),
                        n_ctrl=int(g.get("n_controllers", 1)),
                        led_xy=np.asarray(leds, float), n_true=len(leds),
                        degenerate=bool(g.get("degenerate", False)))
    return out


def nearest_idx(t_sorted, t, max_dt):
    j = int(np.searchsorted(t_sorted, t)); best = -1
    for c in (j - 1, j):
        if 0 <= c < len(t_sorted) and abs(int(t_sorted[c]) - int(t)) <= max_dt:
            if best < 0 or abs(int(t_sorted[c]) - int(t)) < abs(int(t_sorted[best]) - int(t)):
                best = c
    return best


def classify_split(split, cap, cams, models, frame_nblobs):
    tel = Path(cap) / "telemetry"
    m = Manifest.load(tel)
    cand = G.load_stream(tel, m, "candidate")
    sel = cand[cand["selected"] == 1]
    # per-device commit timeline (for the present/bracket test) and per (dev) selected rows
    dev_sel = {d: sel[sel["device_id"] == d] for d in (1, 2)}
    dev_cts = {}
    for d in (1, 2):
        s = dev_sel[d]
        dev_cts[d] = np.sort(s["t_mono_ns"].astype(np.int64)) if len(s) else np.array([], dtype=np.int64)
    win_lo = int(sel["t_mono_ns"].astype(np.int64).min()) if len(sel) else 0
    win_hi = int(sel["t_mono_ns"].astype(np.int64).max()) if len(sel) else 0

    gt = load_gt(Path("dataset/pool") / split)
    rows = []
    for tag, g in sorted(gt.items()):
        cam_id, ts, n_true = g["cam"], g["ts"], g["n_true"]
        id_possible = n_true >= MIN_PNP
        # opportunity: ts within telemetry window?
        in_window = win_lo - BRACKET_NS <= ts <= win_hi + BRACKET_NS
        best = dict(explained=-1.0, dev=None, commit_cam=None, n_proj_vis=0,
                    tilt_deg=None, yaw_deg=None, unmatched=None, had_twin=None,
                    reproj=None, blobs_matched=None)
        any_present = False
        for d in (1, 2):
            cts = dev_cts[d]
            if len(cts) == 0:
                continue
            lo = cts[cts <= ts]; hi = cts[cts >= ts]
            present = len(lo) and len(hi) and (ts - lo[-1] < BRACKET_NS) and (hi[0] - ts < BRACKET_NS)
            any_present = any_present or present
            s = dev_sel[d]
            sts = s["t_mono_ns"].astype(np.int64)
            order = np.argsort(sts); s_sorted = s[order]; sts_sorted = sts[order]
            k = nearest_idx(sts_sorted, ts, MATCH_NS)
            if k < 0:
                continue
            r = s_sorted[k]
            commit_cam = int(r["cam_id"])
            R = g2cam._quat_to_R(np.array([r["qx"], r["qy"], r["qz"], r["qw"]], float))
            t = np.array([r["px"], r["py"], r["pz"]], float)
            if commit_cam == cam_id:
                Rp, tp = R, t                                   # same cam: project directly
            else:
                # cross-cam: object<-cam_Y -> object<-GT-cam_X via the rigid inter-camera extrinsics
                # (head-pose-free; the HMD pose is common to both cameras and cancels).
                Rp, tp = xform(R, t, cams[commit_cam], cams[cam_id])
            pm = g2cam.project_model(cams[cam_id], Rp, tp, models[d])
            uv = pm["uv"][pm["visible"]]
            n_vis = int(len(uv))
            if n_vis:
                D = np.hypot(g["led_xy"][:, None, 0] - uv[None, :, 0],
                             g["led_xy"][:, None, 1] - uv[None, :, 1])
                hit = D.min(axis=1) <= EPS_PX
                # The annotation marks the LED union of ALL visible controllers; one device's pose can
                # only ever explain its own controller's share (a correct pose capped at ~50% on
                # n_ctrl=2 frames was the May "25 near-miss WRONGs" artifact). Score against the
                # spatial cluster of GT LEDs nearest this pose: 2-means split when two controllers.
                if g["n_ctrl"] >= 2 and len(g["led_xy"]) >= 4:
                    own = np.argmin(np.hypot(g["led_xy"][:, 0] - np.median(uv[:, 0]),
                                             g["led_xy"][:, 1] - np.median(uv[:, 1])))
                    seed_a = g["led_xy"][own]
                    seed_b = g["led_xy"][np.argmax(np.hypot(g["led_xy"][:, 0] - seed_a[0],
                                                            g["led_xy"][:, 1] - seed_a[1]))]
                    for _ in range(8):
                        da = np.hypot(g["led_xy"][:, 0] - seed_a[0], g["led_xy"][:, 1] - seed_a[1])
                        db = np.hypot(g["led_xy"][:, 0] - seed_b[0], g["led_xy"][:, 1] - seed_b[1])
                        mine = da <= db
                        if mine.sum() == 0 or (~mine).sum() == 0:
                            break
                        seed_a = g["led_xy"][mine].mean(axis=0)
                        seed_b = g["led_xy"][~mine].mean(axis=0)
                    expl = float(hit[mine].mean()) if mine.sum() else float(hit.mean())
                else:
                    expl = float(hit.mean()) if D.size else 0.0
            else:
                expl = 0.0
            if expl > best["explained"]:
                tilt = float(np.rad2deg(abs(r["tilt_err_rad"])))
                yaw = float(np.rad2deg(abs(r["yaw_err_rad"])))
                best = dict(explained=round(expl, 3), dev=d, commit_cam=commit_cam, n_proj_vis=n_vis,
                            tilt_deg=round(tilt, 1), yaw_deg=round(yaw, 1),
                            unmatched=int(r["unmatched_blobs"]), had_twin=int(r["had_twin"]),
                            reproj=round(float(r["reproj_err_px"]), 2),
                            blobs_matched=int(r["blobs_matched"]))
        # finalize verdict / opportunity. Verdicts:
        #   GOOD        a committed pose (any cam) explains >= GOOD_FRAC of the GT LEDs  -> tracked correctly
        #   WRONG       SAME-cam commit lands >= MIN_PNP visible model LEDs in the GT cam but misses the GT LEDs
        #               -> a real flip/clutter wrong-pose IN this view (gravity-anchored axis decides tilt/yaw).
        #               Requires SAME-cam evidence: a cross-cam pose grazing this view's FOV is too easily a
        #               wide-baseline projection near-miss to call wrong with confidence (same-cam WRONG ~9%
        #               vs cross-cam ~48%, while GOOD explained is identical 1.00 both ways).
        #   UNATTRIB    device committed but (cross-cam) does not explain this view, or its pose reaches
        #               < MIN_PNP visible model LEDs here -> the controller the annotator saw here is not what
        #               the tracker locked from this geometry; ambiguous (tracked-on-another-view vs missed) ->
        #               sub-agent adjudicates.
        #   NO_COMMIT   no committed pose for either device near this frame-time.
        #   ABSTAIN     outside telemetry window or device never present (no matcher opportunity).
        committed = best["dev"] is not None
        same_cam = committed and best["commit_cam"] == cam_id
        if not in_window or (not committed and not any_present):
            verdict = "ABSTAIN"
        elif not committed:
            verdict = "NO_COMMIT"
        elif best["explained"] >= GOOD_FRAC:
            verdict = "GOOD"
        elif same_cam and best["n_proj_vis"] >= MIN_PNP:
            verdict = "WRONG"
        else:
            verdict = "UNATTRIB"
        # bucket: conservative matcher-fail = WRONG (real in-view wrong pose) or true NO_COMMIT, with ID possible.
        if verdict == "ABSTAIN":
            bucket = "ABSTAIN"
        elif verdict == "GOOD":
            bucket = "SUCCESS"
        elif verdict == "UNATTRIB":
            bucket = "UNATTRIB"
        else:  # WRONG or NO_COMMIT
            bucket = "MATCHER_FAIL" if id_possible else "DETECT_LIMITED"
        # failure mode
        mode = None
        nblob = frame_nblobs.get((cam_id, ts))
        if verdict == "WRONG":
            if best["tilt_deg"] is not None and max(best["tilt_deg"], best["yaw_deg"]) > FLIP_DEG:
                mode = "WRONG_TILT" if best["tilt_deg"] >= best["yaw_deg"] else "WRONG_YAW"
            elif best["unmatched"] and best["unmatched"] >= max(3, n_true):
                mode = "WRONG_CLUTTER"
            else:
                mode = "WRONG_OTHER"
        elif verdict == "NO_COMMIT":
            if nblob is not None and nblob >= 2 * max(n_true, 4) and nblob >= 12:
                mode = "CLUTTER_FLOOD"     # detector found far more blobs than real LEDs -> matcher drowned
            else:
                mode = "NO_HYPOTHESIS"
        rows.append(dict(split=split, tag=tag, cam=cam_id, ts=ts, n_true=n_true, n_ctrl=g["n_ctrl"],
                         id_possible=bool(id_possible), degenerate=g["degenerate"],
                         verdict=verdict, bucket=bucket, mode=mode,
                         explained=(best["explained"] if best["dev"] is not None else 0.0),
                         dev=best["dev"], commit_cam=best["commit_cam"], n_proj_vis=best["n_proj_vis"],
                         same_cam=(best["commit_cam"] == cam_id if best["dev"] is not None else None),
                         tilt_deg=best["tilt_deg"], yaw_deg=best["yaw_deg"],
                         unmatched=best["unmatched"], had_twin=best["had_twin"],
                         reproj=best["reproj"], blobs_matched=best["blobs_matched"],
                         frame_nblobs=nblob))
    return rows


def frame_blob_map(tag_dirs):
    """(cam, ts) -> detector blob count, read from the candidates.json n_raw (over-detect count) AND the
    pgm filename _n<N> (production blobwatch count). We use n_raw (what an over-detector sees) for the
    clutter-flood signal, falling back to production n."""
    out = {}
    for d in tag_dirs:
        for cf in glob.glob(str(d / "*.candidates.json")):
            c = json.load(open(cf))
            out[(int(c["cam"]), int(c["ts"]))] = int(c["flags"]["n_raw"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry-root", type=Path, required=True,
                    help="replay root containing xv1, clean2, and headpose outputs")
    ap.add_argument("--json", default=None)
    ap.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = ap.parse_args()
    cams = g2cam.load_cams(CAMS_JSON)
    models = {d: g2cam.load_led_model(Path(p)) for d, p in CTRL.items()}
    allrows = []
    for split in args.splits:
        cap = args.telemetry_root / SPLITS[split]
        fb = frame_blob_map([Path("dataset/pool") / split])
        rows = classify_split(split, cap, cams, models, fb)
        allrows += rows
        report(split, rows)
    print("\n" + "=" * 78)
    print("GRAND TOTAL (all splits)")
    report("ALL", allrows, grand=True)
    if args.json:
        json.dump(allrows, open(args.json, "w"), indent=1)
        print(f"\nwrote {len(allrows)} classified GT frames -> {args.json}")


def report(name, rows, grand=False):
    n = len(rows)
    from collections import Counter
    bk = Counter(r["bucket"] for r in rows)
    vd = Counter(r["verdict"] for r in rows)
    scored = [r for r in rows if r["verdict"] != "ABSTAIN"]
    ns = len(scored)
    print(f"\n--- {name}: {n} controller-visible GT frames "
          f"({ns} with matcher opportunity, {bk.get('ABSTAIN',0)} ABSTAIN) ---")
    print(f"  verdict:  GOOD {vd.get('GOOD',0)}  WRONG {vd.get('WRONG',0)}  "
          f"UNATTRIB {vd.get('UNATTRIB',0)}  NO_COMMIT {vd.get('NO_COMMIT',0)}  ABSTAIN {vd.get('ABSTAIN',0)}")
    if ns:
        succ = bk.get("SUCCESS", 0); mf = bk.get("MATCHER_FAIL", 0)
        dl = bk.get("DETECT_LIMITED", 0); un = bk.get("UNATTRIB", 0)
        print(f"  of {ns} opportunities:  SUCCESS {succ} ({100*succ/ns:.0f}%)  "
              f"MATCHER_FAIL {mf} ({100*mf/ns:.0f}%)  UNATTRIB {un} ({100*un/ns:.0f}%)  "
              f"DETECT_LIMITED {dl} ({100*dl/ns:.0f}%)")
        # recall gap = all non-SUCCESS opportunities. Conservative matcher-recoverable = MATCHER_FAIL;
        # UNATTRIB is ambiguous (tracked-on-another-view vs missed-here) -> sub-agents adjudicate.
        gap = ns - succ
        if gap:
            print(f"  recall gap = {gap} non-success ({100*gap/ns:.0f}% of opportunities): "
                  f"MATCHER-recoverable(conf) {mf} ({100*mf/gap:.0f}%), "
                  f"UNATTRIB(ambig) {un} ({100*un/gap:.0f}%), "
                  f"detection-limited {dl} ({100*dl/gap:.0f}%)")
        mfrows = [r for r in rows if r["bucket"] == "MATCHER_FAIL"]
        md = Counter(r["mode"] for r in mfrows)
        if mfrows:
            same = sum(1 for r in mfrows if r["same_cam"])
            print(f"  MATCHER_FAIL modes: {dict(md)}  (same-cam evidence: {same}/{len(mfrows)})")
    if grand:
        mf_by = Counter((r["split"], r["dev"]) for r in rows if r["bucket"] == "MATCHER_FAIL")
        print("  MATCHER_FAIL by (split,dev):",
              {f"{s}/d{d}": c for (s, d), c in sorted(mf_by.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))})


if __name__ == "__main__":
    main()
