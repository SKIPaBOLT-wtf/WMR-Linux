#!/usr/bin/env python3
"""Assemble the annotation pool from the per-capture select.py outputs: exclude already-annotated frames,
prioritize the failure-rich HARD(<=3 cand) + MED(4-6) bins (the hardest samples), keep a sampled EASY/RICH
baseline for precision, and emit one frame list per capture for prep.py."""
import glob, json, re
from pathlib import Path
import numpy as np

DS = Path("dataset")
SEL = {"20260528-080421-xv-session1": "xv1", "20260526-175615-clean": "clean2", "20260524-200416-headpose": "headpose"}
EASY_FRAC = 0.30   # keep this fraction of easy/rich (GOOD-frame baseline); take all hard+med
rng = np.random.default_rng(0)

def annotated_ts(split):
    out = set()
    for gf in glob.glob(str(DS / split / "*.gt.json")):
        t = Path(gf).stem.replace(".gt", "")           # cam{N}_{ts}
        m = re.match(r"cam(\d+)_(\d+)", t)
        if m: out.add((int(m.group(1)), int(m.group(2))))
    return out

def dbin(n): return "hard" if n <= 3 else "med" if n <= 6 else "easy" if n <= 10 else "rich"

tot = {}
for cap, split in SEL.items():
    done = annotated_ts(split)
    rows = []
    for line in open(f"/tmp/sel_{cap}.txt"):
        m = re.match(r"(\d+)\s+(\d+)\s+#\s*n=(\d+)", line)
        if not m: continue
        cam, ts, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if (cam, ts) in done: continue
        rows.append((cam, ts, n, dbin(n)))
    hard_med = [r for r in rows if r[3] in ("hard", "med")]
    easy_rich = [r for r in rows if r[3] in ("easy", "rich")]
    keep_er = rng.permutation(len(easy_rich))[: int(len(easy_rich) * EASY_FRAC)]
    sel = hard_med + [easy_rich[i] for i in keep_er]
    sel.sort(key=lambda r: (r[0], r[1]))
    with open(f"/tmp/anno_{cap}.txt", "w") as f:
        for cam, ts, n, b in sel:
            f.write(f"{cam} {ts}  # n={n} bin={b}\n")
    from collections import Counter
    c = Counter(r[3] for r in sel)
    tot[cap] = len(sel)
    print(f"{cap:34} {len(sel):4} frames  (excluded {len(done)} done)  bins {dict(c)}")
print(f"\nTOTAL annotation pool: {sum(tot.values())} frames")
