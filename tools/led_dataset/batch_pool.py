#!/usr/bin/env python3
"""Batch the prepped annotation pool for the fleet. Priority: HARD (few-LED, controller-present) first,
then MED, then degenerate (clutter-storm), then easy/rich. Balanced round-robin across captures so each
batch is mixed. Emits /tmp/batches/batch_NN.json = {batch_id, frames:[{split, tag, png, candidates}]}."""
import glob, json
from pathlib import Path

POOL = Path("dataset/pool"); OUT = Path("/tmp/batches"); BATCH = 25
import shutil; shutil.rmtree(OUT, ignore_errors=True); OUT.mkdir(parents=True)

def bin_of(c):
    nclu = c["flags"]["n_cluster"]; degen = c["flags"]["degenerate"]
    n = c["flags"]["n_rendered"]
    if degen: return 2
    if nclu <= 3: return 0       # hard: tiny/sparse cluster
    if nclu <= 6: return 1       # med
    return 3                      # easy/rich

frames = []
for split in ("xv1", "clean", "headpose"):
    for cf in glob.glob(str(POOL / split / "*.candidates.json")):
        c = json.load(open(cf)); tag = Path(cf).stem.replace(".candidates", "")
        frames.append(dict(split=split, tag=tag, png=f"dataset/pool/{split}/{tag}.png",
                           candidates=f"dataset/pool/{split}/{tag}.candidates.json",
                           prio=bin_of(c), nclu=c["flags"]["n_cluster"]))
# priority order, round-robin across captures within each priority for balance
order = []
for prio in (0, 1, 2, 3):
    bysplit = {s: [f for f in frames if f["prio"] == prio and f["split"] == s] for s in ("xv1", "clean", "headpose")}
    for v in bysplit.values(): v.sort(key=lambda f: f["tag"])
    while any(bysplit.values()):
        for s in ("xv1", "clean", "headpose"):
            if bysplit[s]: order.append(bysplit[s].pop(0))
batches = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
for i, b in enumerate(batches):
    json.dump({"batch_id": i, "frames": b}, open(OUT / f"batch_{i:02d}.json", "w"), indent=1)
from collections import Counter
pc = Counter(f["prio"] for f in order)
print(f"{len(order)} frames -> {len(batches)} batches of {BATCH}")
print(f"  priority counts: hard={pc[0]} med={pc[1]} degenerate={pc[2]} easy/rich={pc[3]}")
print(f"  batches 00-{len(batches)-1:02d} written to {OUT}")
