#!/usr/bin/env python3
"""Compile the categorized samples (dataset/pdf_samples/*.png + meta.json) into a single PDF.
One sample per landscape page: composite image (short-exp + overlays | long-exp SLAM) + caption.
Section header pages introduce each category. Title page explains the legend + provenance."""
import json, sys, os
from pathlib import Path
# local select.py / manifest.py shadow stdlib modules matplotlib needs; drop this dir from the import path
sys.path = [p for p in sys.path if p and os.path.basename(os.path.abspath(p)) != "led_dataset"]
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

OUT = Path("/home/mrwhite0racle/g2-linux-research/tools/led_dataset/dataset/pdf_samples"); meta = json.load(open(OUT / "meta.json"))
PDF = Path("/tmp/g2_controller_matching_samples.pdf")
by = {}
for mr in meta: by.setdefault(mr["cat"], []).append(mr)

CAT = {
 "MATCHED-WELL": ("1.  MATCHED WELL  —  pose lands on the LEDs",
   "The matcher committed a pose (red +) that projects onto the ground-truth LEDs (green X). Correct track.\n"
   "These are the frames the tracker gets right: enough LEDs are detected, the right pose-twin is chosen."),
 "MATCHED-WRONG": ("2.  MATCHED BUT WRONG POSE  —  a pose was committed, but flipped / offset",
   "The matcher DID commit a pose (red +), and enough LEDs were present (5-10), but it locked the wrong\n"
   "mirror-twin / a misaligned pose: the red does NOT sit on the green LEDs. 'explained' is the fraction of\n"
   "GT LEDs the committed pose actually hits (<0.6). This is the classic constellation flip."),
 "NOT-MATCHED": ("3.  DID NOT MATCH  —  controller present, too few LEDs detected",
   "No correct pose for this controller (explained ~ 0). The short-exposure frame caught only 1-4 LEDs\n"
   "(cyan o = detected), too few to solve the 6-DOF pose. The LONG-EXPOSURE SLAM panel (right) confirms the\n"
   "controller IS physically present (at the FOV edge / far / turned away). Any stray red = a drifted/prior\n"
   "pose carried by the filter, NOT a match to these LEDs. This is the detection-limited 'optical floor'."),
}
INTERP = {
 "MATCHED-WELL": lambda m: f"Correct: the committed pose hits {int(m['expl']*100)}% of the {m['n_gt']} ground-truth LEDs.",
 "MATCHED-WRONG": lambda m: f"Wrong/flipped pose: {m['n_gt']} LEDs present but the committed pose hits only {int(m['expl']*100)}% of them.",
 "NOT-MATCHED": lambda m: f"No match: {m['det']} of {m['n_gt']} LEDs detected -- too few to solve. Controller present (see SLAM).",
}
ORDER = ["MATCHED-WELL", "MATCHED-WRONG", "NOT-MATCHED"]

with PdfPages(PDF) as pdf:
    # ---- title page ----
    fig = plt.figure(figsize=(11.7, 8.3)); fig.text(0.5, 0.92, "HP Reverb G2 — Controller LED Matching", ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.875, "Per-frame samples vs an independent, hand-annotated ground-truth dataset", ha="center", fontsize=12, style="italic")
    body = (
     "WHAT THIS IS\n"
     "  Each sample is one real controller camera frame from an in-headset capture. The LEDs were annotated\n"
     "  by hand (NOT by the tracker), by inspecting contrast-stretched frames -- an independent judge.\n\n"
     "  LEFT panel  = short-exposure controller frame (LEDs only, room black), shown 2x + contrast-stretched.\n"
     "  RIGHT panel = nearest long-exposure SLAM frame (room visible) -- context proving the controller is real.\n\n"
     "OVERLAY LEGEND\n"
     "    green  X   = ground-truth LED position (hand-annotated)\n"
     "    cyan   o   = blob the detector found\n"
     "    red    +   = the pose the matcher COMMITTED, projected back (the 32-LED model under that pose)\n"
     "  Correct pose -> red on green.  Flip -> red near-but-off green.  Detection failure -> few LEDs, no useful red.\n\n"
     "THE THREE CATEGORIES (this capture, 135 single-controller GT frames = basis for the samples below)\n"
     "    1. MATCHED WELL           93 frames (69%)   pose lands on the LEDs\n"
     "    2. MATCHED BUT WRONG      17 frames (13%)   a pose committed, but flipped/offset (>=5 LEDs in 7)\n"
     "    3. DID NOT MATCH          25 frames (18%)   controller present but too few LEDs detected (median 3)\n\n"
     "  Single-controller frames only so each category is unambiguous; across ALL controller-visible frames\n"
     "  (incl. two-controller) the matched-well rate is ~70%. 'explained' = fraction of the ground-truth LEDs\n"
     "  the committed pose actually projects onto (correct >= 0.6)."
    )
    fig.text(0.06, 0.79, body, ha="left", va="top", fontsize=9.8, family="monospace")
    fig.text(0.5, 0.05, "Generated from tools/led_dataset — edited & maintained by Claude, presented as-is.", ha="center", fontsize=8, color="gray")
    pdf.savefig(fig); plt.close(fig)

    for cat in ORDER:
        items = by.get(cat, [])
        # section header
        fig = plt.figure(figsize=(11.7, 8.3)); title, desc = CAT[cat]
        fig.text(0.5, 0.62, title, ha="center", fontsize=18, weight="bold")
        fig.text(0.5, 0.5, desc, ha="center", va="center", fontsize=12, family="monospace")
        fig.text(0.5, 0.3, f"{len(items)} samples follow", ha="center", fontsize=11, style="italic", color="gray")
        pdf.savefig(fig); plt.close(fig)
        # one sample per page
        for i, mr in enumerate(items, 1):
            img = mpimg.imread(OUT / mr["png"])
            fig = plt.figure(figsize=(11.7, 8.3))
            ax = fig.add_axes([0.02, 0.16, 0.96, 0.74]); ax.imshow(img); ax.axis("off")
            ax.set_title(f"{cat}  —  sample {i}/{len(items)}   |   {mr['tag']}", fontsize=11, weight="bold")
            cap = (f"{INTERP[cat](mr)}\n"
                   f"GT LEDs={mr['n_gt']}    detected={mr['det']}    explained={mr['expl']}\n"
                   f"annotator note: {mr['notes'][:58]}")
            fig.text(0.5, 0.075, cap, ha="center", va="center", fontsize=9.2, family="monospace")
            pdf.savefig(fig); plt.close(fig)

print("wrote", PDF, PDF.stat().st_size, "bytes,", 1 + sum(1 + len(by.get(c, [])) for c in ORDER), "pages")
