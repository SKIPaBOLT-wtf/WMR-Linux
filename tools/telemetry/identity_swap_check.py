#!/usr/bin/env python3
"""Cross-device identity-swap gate for replayed captures.

Classifies every ACCEPTED optical pose_attempt of a replay against the capture's
cleaned-GT of BOTH controllers (world frame, via the recorded SLAM head pose):

  OWN     -- within --own-cm of the device's own cleaned-GT
  SWAP    -- within --other-cm of the OTHER device's cleaned-GT while being far
             from (or without) its own: the mutual identity-swap signature found
             on 20260723-112531 (results/recall-cycle-20260723, episodes at
             41.9-45.0s and 119.3-120.4s: a stale coasted prior re-acquires the
             partner's LED ring, the fold poisons the prior, both devices settle
             into a mutually-exclusive swapped equilibrium)
  WRONG   -- near neither reference (scored by detection_f1 as plain FP)
  NO_GT   -- neither reference bracket valid (unscorable)

Gate: ZERO SWAP rows across the whole capture => exit 0, else exit 1. With
--gate-windows, the exit gate applies only inside the given windows (the frozen
knife-edge fixture windows); whole-capture counts are always reported.

Usage:
  identity_swap_check.py REPLAY_DIR --capture CAPTURE_DIR [--cams CAMS_JSON] \
      [--ctrl-left J] [--ctrl-right J] [--own-cm 5] [--other-cm 5] [--far-own-cm 8] [--out JSON]
  identity_swap_check.py OUT_DIR --regression --bin offline_vio_replay

REPLAY_DIR must contain telemetry/ (pose_attempt.bin etc. from G2_REPLAY_TELEMETRY);
with --regression it is instead the output dir every pinned fixture below is driven
into, one subdirectory per fixture, and each is gated on its own frozen windows. Every
--regression skip path (missing harness, capture, frames or controller jsons) exits 77
so the ctest registration reports a LOUD SKIP.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blob_explain as BE  # noqa: E402
import detection_f1 as DF  # noqa: E402
import g2_geom as G  # noqa: E402
import gt_blob_fix as GF  # noqa: E402
import replay_contract as RC  # noqa: E402
from frame_view import frame_source  # noqa: E402
from manifest import DEVICE_NAMES, Manifest  # noqa: E402
from smooth_ref import build_reference  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: The standing identity-swap regression fixtures. Each pins a capture, the replay regime that
#: reproduces cross-device transposition on it, and the frozen episode windows the gate counts in.
#: Zero SWAP accepts inside those windows is the gate; whole-capture counts are always reported.
#:
#: A window is pinned only where the capture makes identity physically contestable AND the shipped
#: tracker is clean, so the gate reads "this stays clean" rather than "this is still broken". Each
#: capture also carries a KNOWN one-sided steal that is deliberately NOT gated (xv1 dev1 onto dev2's
#: ring at 76.6-77.3 s after a 1.8 s starve, 16 accepts; July-23 dev1 at 98.2 s, 2 accepts, 0.4 cm
#: from the partner's reference at 35 cm ring separation) — those are open front-end debt, and a gate
#: that fails on the tracker it guards guards nothing.
REGRESSION_FIXTURES = (
    {
        # Under a 300 ms/1 s optical blackout both controllers trade LED rings at 60.92-62.05 s:
        # each device accepts the partner's ring, the mutual two-way transposition.
        "name": "xv1-blackout",
        "capture": ROOT / "captures/20260528-080421-xv-session1",
        "drop_env": {
            "G2_REPLAY_DROP_OPTICAL_PERIOD_MS": "1000",
            "G2_REPLAY_DROP_OPTICAL_DURATION_MS": "300",
        },
        "windows": ((60.9, 62.1),),
    },
    {
        # The hands-close class of results/recall-cycle-20260723: a stale coasted prior re-acquires
        # the partner's LED ring, the fold poisons the prior, and both devices settle into a
        # mutually-exclusive swapped equilibrium. Needs no forced dropout — the capture's own
        # hands-close instants are the stimulus, and the capture's cleaned-GT confirms both windows
        # are exactly where the two rings close to ~20 cm (41.90-45.02 s and 119.26-119.54 s are
        # sub-30 cm runs; 41.9-45.0 is 80.6 % scorable, 119.3-120.4 is 100 % scorable for dev2).
        # This is the class that took an otherwise-clean anchor fix from 2 to 93 swap accepts
        # (results/cluster-jump-20260725) and a pre-filter removal from 0 to 5
        # (results/lockout-20260725), with no standing bar in either case.
        "name": "jul23-hands-close",
        "capture": ROOT / "captures/20260723-112531-comprehensive-stack",
        "drop_env": {},
        "windows": ((41.9, 45.0), (119.3, 120.4)),
    },
)


def run_fixture_replay(binary: Path, out_dir: Path, fixture: dict, ctrl_left: str,
                       ctrl_right: str) -> int:
    """Drive the harness over one fixture into out_dir. 0 = produced, 77 = SKIP."""
    for label, path in (("harness binary", binary), ("capture", fixture["capture"]),
                        ("left controller json", Path(ctrl_left)),
                        ("right controller json", Path(ctrl_right))):
        if not Path(path).exists():
            print(f"SKIP: {label} not found ({path}) -- identity-swap gate not run")
            return 77
    try:
        frames = frame_source(fixture["capture"])
    except (OSError, RuntimeError) as exc:
        print(f"SKIP: no replayable frame set for {fixture['name']} ({exc}) "
              f"-- identity-swap gate not run")
        return 77
    telemetry = out_dir / "telemetry"
    replay_out = out_dir / "out"
    for path in (telemetry, replay_out):
        path.mkdir(parents=True, exist_ok=True)
    env = RC.replay_env(fixture["drop_env"], {"G2_REPLAY_TELEMETRY": str(telemetry)},
                        imu_cal_dir=RC.imu_cal_dir_for_capture(fixture["capture"]))
    proc = subprocess.run(
        [str(binary), str(frames), str(RC.cams_for_capture(fixture["capture"])),
         str(fixture["capture"] / "telemetry"), ctrl_left, ctrl_right, str(replay_out)],
        env=env, capture_output=True, text=True)
    for line in (proc.stderr or "").splitlines():
        if any(key in line for key in ("WARN", "ERROR", "FATAL")):
            print(f"replay {line}", file=sys.stderr)
    if proc.returncode != 0:
        print(f"FATAL: {fixture['name']} replay failed ({proc.returncode})", file=sys.stderr)
        return 2
    return 0


def classify_device(replay_telem: Path, cap_telem: Path, dev: int, cams, refs, hp,
                    own_cm: float, other_cm: float, far_own_cm: float, t0: int,
                    collect_poses: bool = False):
    hp_t, hp_pos, hp_q = hp
    mm = Manifest.load(replay_telem)
    pa = G.load_stream(replay_telem, mm, "pose_attempt")
    rows = pa[(pa["device_id"] == dev) & ((pa["outcome"] == 1) | (pa["outcome"] == 2))]

    recs = []
    poses = []
    counts = {"OWN": 0, "SWAP": 0, "WRONG": 0, "NO_GT": 0}
    for row in rows:
        t_hw = int(row["hw_ts_ns"])
        cam_id = int(row["cam_id"])
        if cam_id >= len(cams):
            continue
        hq, hpp = DF._interp_quat_pos(hp_t, hp_q, hp_pos, np.array([t_hw]))
        if not np.isfinite(hpp[0, 0]):
            continue
        R_wo, t_wo = DF.pose_attempt_to_xrworld_device(row, cams[cam_id], hq[0], hpp[0])

        d = {}
        ref_pose = {}
        for d_id in (dev, 3 - dev):
            q, p, valid = DF._interp_reference_to_times(
                refs[d_id], np.array([t_hw], dtype=np.int64), 150.0)
            d[d_id] = float(np.linalg.norm(t_wo - p[0]) * 100.0) if valid[0] else float("inf")
            ref_pose[d_id] = (q[0], p[0], bool(valid[0]))
        d_own, d_other = d[dev], d[3 - dev]

        if d_own < own_cm:
            tag = "OWN"
        elif d_other < other_cm and not (d_own < far_own_cm):
            tag = "SWAP"
        elif not np.isfinite(d_own) and not np.isfinite(d_other):
            tag = "NO_GT"
        else:
            tag = "WRONG" if not (d_own < far_own_cm) else "OWN"
        counts[tag] += 1
        if tag == "SWAP":
            recs.append({
                "t_hw_ns": t_hw,
                "t_rel_s": round((t_hw - t0) / 1e9, 3),
                "cam_id": cam_id,
                "d_own_cm": None if not np.isfinite(d_own) else round(d_own, 2),
                "d_other_cm": round(d_other, 2),
                "blobs_matched": int(row["blobs_matched"]),
                "reproj_err_px": round(float(row["reproj_err_px"]), 2),
            })
            if collect_poses:
                poses.append({"t_hw_ns": t_hw, "cam_id": cam_id,
                              "R_acc": R_wo, "t_acc": t_wo,
                              "own": ref_pose[dev], "other": ref_pose[3 - dev],
                              "head": (hq[0], hpp[0])})
    return counts, recs, poses


def render_swaps(out_dir: Path, capdir: Path, dev: int, recs, poses, cams, cams_json, scale: int = 2):
    """Panel per SWAP accept: the accepted pose projected with THIS device's LED model against the
    blobs it actually explained, plus both devices' cleaned-GT projections. A steal shows as the red
    accepted projection sitting on the magenta partner reference instead of the green own reference."""
    import cv2
    g2cam, _prep = BE._lazy_imports()
    g2cams = g2cam.load_cams(cams_json)
    mdl = {1: g2cam.load_led_model(g2cam.CTRL_LEFT), 2: g2cam.load_led_model(g2cam.CTRL_RIGHT)}
    cache = BE.BlobCache(GF._frames_dir(capdir))
    out_dir.mkdir(parents=True, exist_ok=True)

    def project(R_dev, t_dev, R_hmd, t_hmd, cam, model):
        Rcm, tcm = BE._world_to_cam(R_dev, t_dev, R_hmd, t_hmd, cam)
        pm = g2cam.project_model(g2cams[cam.id], Rcm, tcm, model)
        return pm["uv"][pm["visible"]]

    written = 0
    for rec, ps in zip(recs, poses):
        cam = cams[ps["cam_id"]]
        path, _dt = cache.nearest_frame(ps["cam_id"], ps["t_hw_ns"])
        if path is None:
            continue
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        hi = max(24.0, float(np.percentile(img, 99.9)))
        vis = cv2.cvtColor(np.clip(img.astype(np.float32) / hi * 255.0, 0, 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        blobs, _c = cache.blobs(path)
        for b in blobs:
            cv2.circle(vis, (int(b[0] * scale), int(b[1] * scale)), 7, (0, 200, 200), 1)
        R_h, t_h = DF.quat_to_R(ps["head"][0]), ps["head"][1]
        for (q, p, valid), model, colour in (
                (ps["own"], mdl[dev], (0, 255, 0)),
                (ps["other"], mdl[3 - dev], (255, 0, 255))):
            if not valid:
                continue
            for (x, y) in project(DF.quat_to_R(q), p, R_h, t_h, cam, model):
                cv2.circle(vis, (int(x * scale), int(y * scale)), 3, colour, -1)
        for (x, y) in project(ps["R_acc"], ps["t_acc"], R_h, t_h, cam, mdl[dev]):
            x, y = int(x * scale), int(y * scale)
            cv2.line(vis, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), 1)
            cv2.line(vis, (x - 5, y + 5), (x + 5, y - 5), (0, 0, 255), 1)
        pad = np.zeros((44, vis.shape[1], 3), np.uint8)
        own = "—" if rec["d_own_cm"] is None else f"{rec['d_own_cm']:.1f}cm"
        cv2.putText(pad, f"dev{dev} t={rec['t_rel_s']:.3f}s cam{rec['cam_id']} "
                         f"d_own={own} d_other={rec['d_other_cm']:.1f}cm "
                         f"blobs={rec['blobs_matched']} reproj={rec['reproj_err_px']:.2f}px",
                    (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(pad, "cyan=blobs  red=ACCEPTED(own model)  green=own GT  magenta=partner GT",
                    (6, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        name = f"dev{dev}_{rec['t_rel_s']:.3f}s_cam{rec['cam_id']}.png"
        cv2.imwrite(str(out_dir / name), np.vstack([pad, vis]))
        written += 1
    return written


def score_replay(replay: Path, capture: Path, cams_json: str, windows, own_cm: float,
                 other_cm: float, far_own_cm: float, render: Path | None):
    """Classify every accepted optical pose of `replay` against `capture`'s two cleaned-GT tracks.

    Prints the per-device tally, the swap episodes, and the two-way transposition signature inside
    each gated window. Returns the verdict dict (`gated_swaps` is the exit gate's count), or None
    when the capture cannot serve as a reference.
    """
    cap_telem = capture / "telemetry"
    cams = DF.load_cameras(cams_json)
    refs = {d: build_reference(cap_telem, d) for d in (1, 2)}
    if refs[1] is None or refs[2] is None:
        print("FATAL: missing cleaned-GT reference for a device", file=sys.stderr)
        return None
    hp = DF._load_head_pose(cap_telem)
    if hp is None:
        print("FATAL: no head_pose.bin in the reference capture", file=sys.stderr)
        return None
    frames = G.load_stream(cap_telem, Manifest.load(cap_telem), "frame")
    t0 = int(frames["hw_ts_ns"].astype(np.int64).min())

    verdict = {"replay": str(replay), "capture": str(capture), "own_cm": own_cm,
               "other_cm": other_cm, "far_own_cm": far_own_cm, "gate_windows": windows,
               "devices": {}}
    all_recs = {}
    total_swaps = 0
    gated_swaps = 0
    for dev in (1, 2):
        counts, recs, poses = classify_device(replay / "telemetry", cap_telem, dev, cams, refs, hp,
                                              own_cm, other_cm, far_own_cm, t0,
                                              collect_poses=render is not None)
        if render is not None:
            n = render_swaps(render, capture, dev, recs, poses, cams, cams_json)
            print(f"  rendered {n} panel(s) -> {render}")
        all_recs[dev] = recs
        eps = episodes(recs)
        verdict["devices"][str(dev)] = {"counts": counts, "episodes": eps, "swap_rows": recs}
        total_swaps += counts["SWAP"]
        if windows is not None:
            gated_swaps += sum(1 for r in recs
                               if any(lo <= r["t_rel_s"] <= hi for lo, hi in windows))
        print(f"dev{dev} ({DEVICE_NAMES.get(dev, str(dev))}): "
              + "  ".join(f"{k}={v}" for k, v in counts.items()))
        for e in eps:
            print(f"  swap episode {e['start_s']:.2f}s - {e['end_s']:.2f}s  ({e['n']} accepts)")

    verdict["total_swaps"] = total_swaps
    verdict["gated_swaps"] = gated_swaps if windows is not None else total_swaps
    verdict["mutual_swap_frames"] = {}
    for lo, hi in windows or ():
        mutual = mutual_swap_frames(all_recs, (lo, hi))
        verdict["mutual_swap_frames"][f"{lo}-{hi}"] = mutual
        print(f"  window {lo}-{hi}s: two-way transposition frames: {len(mutual)}"
              + (f" -> {mutual[:8]}" if mutual else ""))
    return verdict


def mutual_swap_frames(recs_by_dev, window, tol_ns: int = 20_000_000):
    """Frames where BOTH devices accepted a pose on the other's reference at the same instant — the
    two-way transposition signature. A one-sided steal is a single device losing its track; a mutual
    one is the pair settling into a self-consistent swapped equilibrium that no per-device metric can
    see, because each controller is then tracked accurately, just as the wrong controller."""
    lo, hi = window
    out = []
    for a in recs_by_dev[1]:
        if not (lo <= a["t_rel_s"] <= hi):
            continue
        for b in recs_by_dev[2]:
            if lo <= b["t_rel_s"] <= hi and abs(a["t_hw_ns"] - b["t_hw_ns"]) <= tol_ns:
                out.append(round(a["t_rel_s"], 3))
                break
    return sorted(set(out))


def episodes(recs, gap_s: float = 1.0):
    out = []
    for r in recs:
        t = r["t_rel_s"]
        if out and t - out[-1][1] <= gap_s:
            out[-1][1] = t
            out[-1][2] += 1
        else:
            out.append([t, t, 1])
    return [{"start_s": a, "end_s": b, "n": n} for a, b, n in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay", help="replay output dir (contains telemetry/); with --regression, "
                                   "the dir each fixture replay is driven into")
    ap.add_argument("--regression", action="store_true",
                    help="run every pinned REGRESSION_FIXTURES replay with --bin and gate each on "
                         "its own frozen windows")
    ap.add_argument("--bin", dest="binary", help="offline_vio_replay harness (required by --regression)")
    ap.add_argument("--capture", help="reference capture dir (cleaned-GT + head pose)")
    ap.add_argument("--cams", help="hmd-cameras.json (default: the capture's own provenance snapshot)")
    ap.add_argument("--ctrl-left", default=DF.DEFAULT_CTRL_LEFT)
    ap.add_argument("--ctrl-right", default=DF.DEFAULT_CTRL_RIGHT)
    ap.add_argument("--own-cm", type=float, default=5.0)
    ap.add_argument("--other-cm", type=float, default=5.0)
    ap.add_argument("--far-own-cm", type=float, default=8.0)
    ap.add_argument("--gate-windows", default=None,
                    help="comma-separated start-end windows in capture-relative seconds "
                         "(e.g. '7.2-8.7,41.9-45.0,119.1-123.2'); the exit gate then counts "
                         "only swaps inside these windows")
    ap.add_argument("--out", default=None, help="write full JSON verdict here")
    ap.add_argument("--render", default=None, metavar="DIR",
                    help="render a per-SWAP eyes-on panel (accepted pose vs both devices' GT) here")
    args = ap.parse_args()
    if args.regression:
        if args.binary is None:
            ap.error("--regression needs --bin (the offline_vio_replay harness under test)")
        if args.capture or args.cams or args.gate_windows:
            ap.error("--regression pins its own capture, camera config and windows per fixture; "
                     "--capture/--cams/--gate-windows would silently apply one fixture's inputs "
                     "to all of them")
        return run_regression(args)
    if args.capture is None:
        ap.error("--capture is required without --regression")
    args.cams = args.cams or str(RC.cams_for_capture(args.capture))
    windows = None
    if args.gate_windows:
        windows = [tuple(float(x) for x in w.split("-")) for w in args.gate_windows.split(",")]

    verdict = score_replay(Path(args.replay), Path(args.capture), args.cams, windows, args.own_cm,
                           args.other_cm, args.far_own_cm,
                           Path(args.render) if args.render else None)
    if verdict is None:
        return 2
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=1))
        print(f"wrote {args.out}")
    scope = "in gate windows" if windows is not None else "whole capture"
    if verdict["gated_swaps"] > 0:
        print(f"FAIL: {verdict['gated_swaps']} identity-swap accepts {scope} "
              f"({verdict['total_swaps']} whole-capture)")
        return 1
    print(f"PASS: zero identity-swap accepts {scope} ({verdict['total_swaps']} whole-capture)")
    return 0


def run_regression(args) -> int:
    """Replay and gate every pinned fixture. 0 = all clean, 1 = a gated window swapped, 77 = SKIP."""
    verdicts = []
    failed = []
    for fixture in REGRESSION_FIXTURES:
        out_dir = Path(args.replay) / fixture["name"]
        print(f"=== fixture {fixture['name']}: {fixture['capture'].name} "
              f"windows={fixture['windows']}")
        code = run_fixture_replay(Path(args.binary), out_dir, fixture, args.ctrl_left,
                                  args.ctrl_right)
        if code != 0:
            return code
        verdict = score_replay(out_dir, fixture["capture"],
                               str(RC.cams_for_capture(fixture["capture"])),
                               fixture["windows"], args.own_cm, args.other_cm, args.far_own_cm,
                               Path(args.render) / fixture["name"] if args.render else None)
        if verdict is None:
            return 2
        verdict["fixture"] = fixture["name"]
        verdicts.append(verdict)
        if verdict["gated_swaps"]:
            failed.append(fixture["name"])
            print(f"FAIL: {fixture['name']}: {verdict['gated_swaps']} identity-swap accepts in "
                  f"{fixture['windows']} ({verdict['total_swaps']} whole-capture)")
        else:
            print(f"PASS: {fixture['name']}: zero identity-swap accepts in {fixture['windows']} "
                  f"({verdict['total_swaps']} whole-capture)")
    if args.out:
        Path(args.out).write_text(json.dumps(verdicts, indent=1))
        print(f"wrote {args.out}")
    if failed:
        print(f"FAIL: identity-swap gate failed on {', '.join(failed)}")
        return 1
    print(f"PASS: {len(verdicts)} identity-swap fixture(s) clean in every gated window")
    return 0


if __name__ == "__main__":
    sys.exit(main())
