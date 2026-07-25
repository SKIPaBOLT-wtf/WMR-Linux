#!/usr/bin/env python3
"""holdsnap.py -- measure the felt "hold at a wrong pose, then snap" failure mode.

The symptom: the fused report sits at a materially wrong position for a few hundred ms while the
optical front-end is, all along, delivering a BETTER pose that the filter refuses to adopt; then the
filter lets go and the report jumps. The user feels a stuck controller followed by a teleport.

Both streams come from the SAME offline_vio_replay dev CSV (`opt_*` is the front-end pose as
submitted to the fusion, before the fusion's own adoption gates; `pred_*` is the fused report), and
both are scored against the cleaned-GT reference from smooth_ref (deflip -> speed-gate -> RTS).

Episode definition (ONE threshold, and it is derived, not tuned):

    gain(t) = pos_err(pred, t) - pos_err(opt, t)      [metres of error given up by not adopting opt]

    A hold episode is a maximal run of consecutive scored frames with gain(t) >= HOLD_GAIN_M, where
    HOLD_GAIN_M = OPTICAL_JUMP_SLACK_M = 0.50 m, the fusion's own jump-gate slack
    (t_tracker_kalman_fusion.cpp). That constant is the right and only threshold here:

      * gain >= slack implies |pred - opt| >= slack, which is exactly the condition under which the
        adoption gate (`optical_adoption_motion_plausible`: |opt - last_good| <= 12*dt + slack) can
        reject the optical candidate at all when the optical anchor is fresh (dt ~ 0). Below the
        slack the gate is physically incapable of rejecting, so a sub-slack disagreement can never be
        an instance of this failure mode.
      * it is signed against the alternative: the filter is not merely wrong, it is wrong BY MORE
        than the error it would have had if it had simply believed the optical it was handed.

    The trailing SNAP is measured, not required, so an episode that never resolves is still counted
    (a hold that never ends is strictly worse than one that snaps).

Reported per device: episode count, total wrong-pose dwell (s and % of scored time), the duration
distribution, the felt-band subset (FELT_MIN_S..FELT_MAX_S -- the ~0.5-1 s the user reports), the
mean/max gain, and the snap magnitude at episode exit.

Usage:
  holdsnap.py <capture_telemetry_dir> <replay_out_dir> [--device N] [--out JSON]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from mse_eval import load_candidate_csv, match_to_reference
from smooth_ref import build_reference
from manifest import DEVICE_NAMES

#: The fusion's own optical jump-gate slack (OPTICAL_JUMP_SLACK_M in t_tracker_kalman_fusion.cpp).
#: Below this the adoption gate cannot reject a candidate, so a smaller disagreement is out of scope.
HOLD_GAIN_M = 0.50
#: Candidate/reference pairing window; matches mse_eval's default so both scores see the same frames.
MATCH_MS = 25.0
#: The band the user reports feeling ("holds ~0.5-1 s then snaps"). Reported as a subset, never as a
#: filter on the episode set -- shorter holds are still real, they are just not the felt complaint.
FELT_MIN_S = 0.3
FELT_MAX_S = 2.0
#: Frames after an episode ends over which the exit jump is measured (a snap resolves within a
#: couple of optical frames at 30-45 Hz).
SNAP_LOOKAHEAD_FRAMES = 3


def _percentile(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def hold_snap_episodes(t_ns, err_pred, err_opt, pos_pred):
    """Maximal runs where the fused report is worse than the optical it was handed by >= HOLD_GAIN_M.

    t_ns/err_pred/err_opt/pos_pred are aligned per-scored-frame arrays (same reference frames)."""
    gain = err_pred - err_opt
    hot = gain >= HOLD_GAIN_M
    episodes = []
    n = hot.shape[0]
    i = 0
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hot[j + 1]:
            j += 1
        dur_s = float(int(t_ns[j]) - int(t_ns[i])) / 1e9
        snap_m = 0.0
        for k in range(j + 1, min(j + 1 + SNAP_LOOKAHEAD_FRAMES, n)):
            snap_m = max(snap_m, float(np.linalg.norm(pos_pred[k] - pos_pred[k - 1])))
        episodes.append(dict(
            t_start_ns=int(t_ns[i]),
            t_end_ns=int(t_ns[j]),
            n_frames=int(j - i + 1),
            duration_s=dur_s,
            gain_mean_m=float(np.mean(gain[i:j + 1])),
            gain_max_m=float(np.max(gain[i:j + 1])),
            pred_err_max_m=float(np.max(err_pred[i:j + 1])),
            opt_err_mean_m=float(np.mean(err_opt[i:j + 1])),
            exit_snap_m=snap_m,
            resolved=bool(j + 1 < n),
        ))
        i = j + 1
    return episodes


def score_device(telemetry_dir: Path, csv_path: Path, dev: int) -> dict | None:
    """Score one device's replay CSV for hold-then-snap episodes. None if unscoreable."""
    ref = build_reference(telemetry_dir, dev)
    if ref is None:
        return None

    # load_candidate_csv already drops non-finite and non-valid samples (pred_tracked / opt_valid).
    t_p, pos_p, _ = load_candidate_csv(csv_path, "pred")
    t_o, pos_o, _ = load_candidate_csv(csv_path, "opt")

    # Score both streams on the SAME reference frames: an episode is only meaningful where the
    # optical actually delivered a pose (a coast is a different failure mode, measured elsewhere).
    ci_p, ri_p = match_to_reference(ref, t_p, MATCH_MS, valid_only=True)
    ci_o, ri_o = match_to_reference(ref, t_o, MATCH_MS, valid_only=True)
    if ci_p.shape[0] == 0 or ci_o.shape[0] == 0:
        return None

    def _nearest_by_ref(cand_idx, ref_idx, t_cand, pos_cand):
        """Per reference frame keep the temporally nearest candidate: (err_m, |dt|, position)."""
        best: dict[int, tuple[float, int, np.ndarray]] = {}
        for c, r in zip(cand_idx, ref_idx):
            dt = abs(int(t_cand[c]) - int(ref.t_ns[r]))
            prev = best.get(int(r))
            if prev is None or dt < prev[1]:
                best[int(r)] = (float(np.linalg.norm(pos_cand[c] - ref.pos[r])), dt, pos_cand[c])
        return best

    err_pred_by_ref = _nearest_by_ref(ci_p, ri_p, t_p, pos_p)
    err_opt_by_ref = _nearest_by_ref(ci_o, ri_o, t_o, pos_o)

    shared = sorted(set(err_pred_by_ref) & set(err_opt_by_ref))
    if len(shared) < 2:
        return None
    t_s = np.array([int(ref.t_ns[r]) for r in shared], dtype=np.int64)
    e_pred = np.array([err_pred_by_ref[r][0] for r in shared])
    e_opt = np.array([err_opt_by_ref[r][0] for r in shared])
    p_pred = np.array([err_pred_by_ref[r][2] for r in shared])

    episodes = hold_snap_episodes(t_s, e_pred, e_opt, p_pred)
    scored_span_s = float(int(t_s[-1]) - int(t_s[0])) / 1e9
    dwell_s = float(sum(e["duration_s"] for e in episodes))
    durations = np.array([e["duration_s"] for e in episodes]) if episodes else np.array([])
    felt = [e for e in episodes if FELT_MIN_S <= e["duration_s"] <= FELT_MAX_S]
    return dict(
        device_id=dev,
        device=DEVICE_NAMES.get(dev, str(dev)),
        scored_frames=len(shared),
        scored_span_s=scored_span_s,
        hold_gain_m=HOLD_GAIN_M,
        n_episodes=len(episodes),
        n_felt_band=len(felt),
        dwell_s=dwell_s,
        dwell_pct=100.0 * dwell_s / scored_span_s if scored_span_s > 0 else None,
        felt_dwell_s=float(sum(e["duration_s"] for e in felt)),
        duration_med_s=_percentile(durations, 50),
        duration_p95_s=_percentile(durations, 95),
        duration_max_s=float(durations.max()) if durations.size else None,
        gain_max_m=max((e["gain_max_m"] for e in episodes), default=None),
        exit_snap_max_m=max((e["exit_snap_m"] for e in episodes), default=None),
        pred_err_mean_m=float(np.mean(e_pred)),
        opt_err_mean_m=float(np.mean(e_opt)),
        episodes=episodes,
    )


def score_run(telemetry_dir: Path, out_dir: Path, devices=None) -> list[dict]:
    results = []
    for csv_path in sorted(Path(out_dir).glob("dev*.csv")):
        if csv_path.name.endswith("_render.csv"):
            continue
        dev = int(csv_path.stem.replace("dev", ""))
        if devices and dev not in devices:
            continue
        res = score_device(Path(telemetry_dir), csv_path, dev)
        if res is not None:
            res["csv"] = str(csv_path)
            results.append(res)
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("telemetry_dir", type=Path, help="capture telemetry dir (cleaned-GT source)")
    ap.add_argument("out_dir", type=Path, help="offline_vio_replay out dir with dev*.csv")
    ap.add_argument("--device", type=int, action="append")
    ap.add_argument("--out", type=Path, help="write the full result (incl. per-episode rows) as JSON")
    args = ap.parse_args()

    results = score_run(args.telemetry_dir, args.out_dir, args.device)
    if not results:
        print("no scoreable device", file=sys.stderr)
        return 1
    for r in results:
        print(f"dev{r['device_id']} ({r['device']}): {r['n_episodes']} hold episodes "
              f"({r['n_felt_band']} in the {FELT_MIN_S}-{FELT_MAX_S}s felt band), "
              f"dwell {r['dwell_s']:.3f}s = {r['dwell_pct']:.3f}% of {r['scored_span_s']:.1f}s scored; "
              f"max dur {r['duration_max_s'] if r['duration_max_s'] is not None else float('nan'):.3f}s, "
              f"max gain {r['gain_max_m'] if r['gain_max_m'] is not None else float('nan'):.3f}m, "
              f"max exit snap {r['exit_snap_max_m'] if r['exit_snap_max_m'] is not None else float('nan'):.3f}m")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
