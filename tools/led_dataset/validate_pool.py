#!/usr/bin/env python3
"""Validate the fleet's annotations (schema + id-partition integrity) and report dataset science:
composition, controller-visible rate, per-capture/camera, LED-count distribution, detector-miss rate,
confidence mix. Catches any agent error before the annotations become GT."""
import glob, json
from pathlib import Path
from collections import Counter
import numpy as np

POOL = Path(__file__).resolve().parent / "dataset/pool"
REQ = {"tag", "controller_visible", "n_controllers", "degenerate", "led_ids", "clutter_ids",
       "ambiguous_ids", "confidence", "notes"}
errors = []; rows = []
for split in ("xv1", "clean", "headpose"):
    for gf in glob.glob(str(POOL / split / "*.gt.json")):
        tag = Path(gf).stem.replace(".gt", "")
        try:
            g = json.load(open(gf))
        except Exception as e:
            errors.append(f"{split}/{tag}: unreadable ({e})"); continue
        miss = REQ - set(g)
        if miss: errors.append(f"{split}/{tag}: missing {miss}")
        cf = POOL / split / f"{tag}.candidates.json"
        cand_ids = {c["id"] for c in json.load(open(cf))["candidates"]} if cf.exists() else set()
        led = set(g.get("led_ids", [])); clut = set(g.get("clutter_ids", [])); amb = set(g.get("ambiguous_ids", []))
        # partition integrity: every candidate id in exactly one bucket, no extras
        union = led | clut | amb
        if cand_ids and union != cand_ids:
            errors.append(f"{split}/{tag}: id partition != candidates (extra {union-cand_ids}, missing {cand_ids-union})")
        if led & clut or led & amb or clut & amb:
            errors.append(f"{split}/{tag}: overlapping buckets")
        if not g.get("controller_visible") and led:
            errors.append(f"{split}/{tag}: not visible but led_ids non-empty")
        rows.append(dict(split=split, tag=tag, cam=int(tag.split("_")[0][3:]), vis=bool(g.get("controller_visible")),
                         ndev=g.get("n_controllers", 0), degen=bool(g.get("degenerate")),
                         nled=len(led), nmiss=len(g.get("missed", [])), conf=g.get("confidence", "?")))

n = len(rows); vis = [r for r in rows if r["vis"]]
print(f"=== POOL ANNOTATIONS: {n} frames validated, {len(errors)} integrity errors ===")
for e in errors[:20]: print("  ERR", e)
print(f"\ncontroller_visible: {len(vis)} ({100*len(vis)/max(n,1):.0f}%)   degenerate: {sum(r['degen'] for r in rows)}")
print("per split:", dict(Counter(r['split'] for r in rows)))
print("  visible per split:", dict(Counter(r['split'] for r in vis)))
print("per camera (visible):", dict(sorted(Counter(r['cam'] for r in vis).items())))
print("confidence (visible):", dict(Counter(r['conf'] for r in vis)))
if vis:
    nl = np.array([r["nled"] for r in vis])
    print(f"\nLED-count per visible frame: min={nl.min()} median={np.median(nl):.0f} mean={nl.mean():.1f} max={nl.max()}")
    bins = {"1-2 (hardest)": int(((nl>=1)&(nl<=2)).sum()), "3-4 (flip-prone)": int(((nl>=3)&(nl<=4)).sum()),
            "5-6": int(((nl>=5)&(nl<=6)).sum()), "7+ (rich)": int((nl>=7).sum())}
    print("  distribution:", bins)
    print(f"  frames with detector-MISSED LEDs flagged: {sum(1 for r in vis if r['nmiss']>0)} "
          f"(total {sum(r['nmiss'] for r in vis)} missed LEDs)")
print(f"\nn_controllers=2 frames: {sum(1 for r in rows if r['ndev']==2)}")
