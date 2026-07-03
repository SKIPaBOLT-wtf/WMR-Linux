"""Post-session spike-attribution report over a var/perf/<ts>/ session
(render-program §4 — turns Phase-0 raw channels into an answer).

Reads frametiming.csv (per-frame), dmon.csv (1 Hz GPU power/clock/violation),
pmon.csv (1 Hz per-process SM%), joins them on the wall clock via clock.json
(written by core.frametiming), and reports: frame-time distributions, every
spike burst with its attribution (power-cap / thermal / memclk excursion /
core-clock dip / desktop GPU contention / CPU present-wait / unattributed),
and a class summary. The async-less present path turns every unhandled spike
into a full 11.1 ms judder — this is the tool that says which lever to pull.

    python3 -m core.perfreport [session-dir]     # default: var/perf/latest
"""
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FRAME_BUDGET_MS = 11.111        # 90 Hz panel
BURST_GAP_S = 1.0               # spikes closer than this merge into one burst
JOIN_WINDOW_S = 2.0             # dmon/pmon rows considered around a burst
VR_PROCS = ("vrcompositor", "vrserver", "vrmonitor", "vrdashboard",
            "vrwebhelper", "vrstartup")


def _percentiles(values, points=(50, 90, 95, 99, 99.9)):
    if not values:
        return {}
    s = sorted(values)
    out = {p: s[min(len(s) - 1, int(len(s) * p / 100))] for p in points}
    out["max"] = s[-1]
    return out


def _fmt_dist(name, dist):
    if not dist:
        return f"  {name:<26} (no data)"
    body = "  ".join(f"p{p}={v:6.2f}" for p, v in dist.items() if p != "max")
    return f"  {name:<26} {body}  max={dist['max']:6.2f}"


def load_frames(path):
    with open(path) as f:
        return [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(f)]


def load_smi(path):
    """Parse `nvidia-smi {dmon,pmon} -o DT -f` output: '#'-prefixed header
    rows name the columns; data rows are whitespace-separated, '-' = absent.
    Returns (rows, columns) with rows carrying a wall-clock 't' (epoch s)."""
    import datetime
    cols, rows = None, []
    if not Path(path).exists():
        return rows, cols
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0].startswith("#"):
            if parts[0] == "#Date":     # header (not the units row)
                cols = [p.lstrip("#").lower() for p in parts]
            continue
        if cols is None or len(parts) < 3:
            continue
        row = dict(zip(cols, parts))
        try:
            row["t"] = datetime.datetime.strptime(
                f"{row['date']} {row['time']}", "%Y%m%d %H:%M:%S").timestamp()
        except (KeyError, ValueError):
            continue
        rows.append(row)
    return rows, cols


def _num(row, key):
    try:
        return float(row.get(key, "-"))
    except ValueError:
        return None


def find_spikes(frames):
    """Per-frame spike flags -> list of spike frames with reasons.
    Drop/mispresent/present counts are per-frame in Compositor_FrameTiming,
    not cumulative."""
    spikes = []
    for fr in frames:
        reasons = []
        gpu = fr["m_flTotalRenderGpuMs"] + fr["m_flCompositorRenderGpuMs"]
        if fr["m_nNumDroppedFrames"] > 0:
            reasons.append("dropped")
        if fr["m_nNumMisPresented"] > 0:
            reasons.append("mispresented")
        if fr["m_nNumFramePresents"] > 1:
            reasons.append("represented")
        if gpu > FRAME_BUDGET_MS:
            reasons.append(f"gpu={gpu:.1f}ms")
        if fr["m_flWaitForPresentCpuMs"] > FRAME_BUDGET_MS:
            reasons.append(f"present-wait={fr['m_flWaitForPresentCpuMs']:.1f}ms")
        if reasons:
            spikes.append({"frame": fr, "reasons": reasons})
    return spikes


def group_bursts(spikes):
    bursts = []
    for s in spikes:
        t = s["frame"]["m_flSystemTimeInSeconds"]
        if bursts and t - bursts[-1]["end"] <= BURST_GAP_S:
            b = bursts[-1]
            b["end"] = t
            b["spikes"].append(s)
        else:
            bursts.append({"start": t, "end": t, "spikes": [s]})
    return bursts


def attribute(burst, offset, dmon, pmon, mclk_max, pclk_base):
    """Attribution flags for one burst from the ±JOIN_WINDOW_S dmon/pmon rows."""
    if offset is None:
        return ["no-clock-bridge"]
    w0 = burst["start"] + offset - JOIN_WINDOW_S
    w1 = burst["end"] + offset + JOIN_WINDOW_S
    flags = []
    near_d = [r for r in dmon if w0 <= r["t"] <= w1]
    if any((_num(r, "pviol") or 0) > 0 for r in near_d):
        flags.append("power-cap")
    if any((_num(r, "tviol") or 0) > 0 for r in near_d):
        flags.append("thermal")
    if mclk_max and any((_num(r, "mclk") or mclk_max) < mclk_max for r in near_d):
        flags.append("memclk-excursion")
    if pclk_base and any((_num(r, "pclk") or pclk_base) < 0.95 * pclk_base
                         for r in near_d):
        flags.append("coreclk-dip")
    hogs = {}
    for r in pmon:
        cmd = r.get("command") or r.get("name") or "-"
        if w0 <= r["t"] <= w1 and not cmd.startswith(VR_PROCS):
            sm = _num(r, "sm")
            if sm:
                hogs[cmd] = max(hogs.get(cmd, 0), sm)
    for cmd, sm in sorted(hogs.items(), key=lambda kv: -kv[1])[:2]:
        flags.append(f"contention:{cmd}={sm:.0f}%sm")
    if any("present-wait" in x for s in burst["spikes"] for x in s["reasons"]):
        flags.append("cpu-present-wait")
    return flags or ["unattributed"]


def report(sess):
    frames = load_frames(sess / "frametiming.csv")
    if not frames:
        return f"{sess}: frametiming.csv empty — no report."
    dmon, _ = load_smi(sess / "dmon.csv")
    pmon, _ = load_smi(sess / "pmon.csv")
    offset = None
    clock = sess / "clock.json"
    if clock.exists():
        offset = json.loads(clock.read_text()).get("wall_minus_vrsys_s")

    dur = frames[-1]["m_flSystemTimeInSeconds"] - frames[0]["m_flSystemTimeInSeconds"]
    n = len(frames)
    dropped = sum(f["m_nNumDroppedFrames"] for f in frames)
    mis = sum(f["m_nNumMisPresented"] for f in frames)
    reproj = sum(1 for f in frames if f["m_nReprojectionFlags"] != 0)
    app_gpu = [f["m_flTotalRenderGpuMs"] for f in frames]
    comp_gpu = [f["m_flCompositorRenderGpuMs"] for f in frames]
    total_gpu = [a + c for a, c in zip(app_gpu, comp_gpu)]
    interval = [f["m_flClientFrameIntervalMs"] for f in frames]
    over = sum(1 for g in total_gpu if g > FRAME_BUDGET_MS)

    mclk_vals = [v for r in dmon if (v := _num(r, "mclk")) is not None]
    pclk_vals = [v for r in dmon if (v := _num(r, "pclk")) is not None]
    mclk_max = max(mclk_vals) if mclk_vals else None
    pclk_base = sorted(pclk_vals)[len(pclk_vals) // 2] if pclk_vals else None

    bursts = group_bursts(find_spikes(frames))
    for b in bursts:
        b["flags"] = attribute(b, offset, dmon, pmon, mclk_max, pclk_base)

    lines = [
        f"perf report — {sess}",
        f"  frames={n}  duration={dur:.1f}s  rate={n / dur:.2f}/s"
        f"  dropped={dropped:.0f}  mispresented={mis:.0f}  reprojected-frames={reproj}",
        f"  frames over {FRAME_BUDGET_MS:.3f}ms GPU budget: {over}"
        f" ({100 * over / n:.2f}%)",
        "",
        _fmt_dist("App GPU ms", _percentiles(app_gpu)),
        _fmt_dist("Compositor GPU ms", _percentiles(comp_gpu)),
        _fmt_dist("Serialized GPU ms", _percentiles(total_gpu)),
        _fmt_dist("Client frame interval ms", _percentiles(interval)),
        _fmt_dist("WaitForPresent CPU ms",
                  _percentiles([f["m_flWaitForPresentCpuMs"] for f in frames])),
        "",
        f"  spike bursts: {len(bursts)}"
        + ("" if offset is not None else "  (no clock.json — attribution off)"),
    ]
    for b in bursts[:40]:
        t0 = b["start"] - frames[0]["m_flSystemTimeInSeconds"]
        worst = max((s["frame"]["m_flTotalRenderGpuMs"]
                     + s["frame"]["m_flCompositorRenderGpuMs"])
                    for s in b["spikes"])
        kinds = sorted({r.split("=")[0] for s in b["spikes"] for r in s["reasons"]})
        lines.append(f"    +{t0:7.1f}s  {len(b['spikes'])}fr  worst-gpu={worst:5.1f}ms"
                     f"  [{','.join(kinds)}]  -> {', '.join(b['flags'])}")
    if len(bursts) > 40:
        lines.append(f"    ... {len(bursts) - 40} more bursts")
    counts = {}
    for b in bursts:
        for fl in b["flags"]:
            counts[fl.split(":")[0]] = counts.get(fl.split(":")[0], 0) + 1
    if bursts:
        lines += ["", "  attribution summary (bursts may carry several flags):"]
        lines += [f"    {k:<20} {v}" for k, v in
                  sorted(counts.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines)


def main(argv):
    arg = argv[1] if len(argv) > 1 else str(REPO / "var" / "perf" / "latest")
    sess = Path(arg).resolve()
    if not sess.is_dir():
        sys.exit(f"perfreport: no session dir at {sess}")
    print(report(sess))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
