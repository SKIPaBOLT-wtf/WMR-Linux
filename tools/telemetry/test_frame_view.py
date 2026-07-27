#!/usr/bin/env python3
"""test_frame_view.py -- behavior tests for the recorded-frame selection rule.

The rule's whole job is to be unambiguous where a directory listing is not, so these tests assert
what it must never do: split a contaminated directory arbitrarily, or quietly hand back a frame set
that is not the one the telemetry recorded. A short frame set is the dangerous failure -- it looks
like a clean run and moves every number downstream -- so each way of being short is asserted to
raise. Run with the conda env:
    ~/miniconda3/envs/g2vr/bin/python test_frame_view.py
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

from frame_view import CAM_COUNT, VIEW_DIRNAME, disk_index, frame_source, recorded_keys

PASS = 0
FAIL = 0

ROOT = Path(__file__).resolve().parents[2]
JULY23 = ROOT / "captures/20260723-112531-comprehensive-stack"

#: The 8,744 four-camera groups frame.bin recorded for the July-23 session, and nothing else.
JULY23_RECORDED_FRAMES = 34976

#: frame.bin's row layout, from the producer's self-describing manifest (u_g2_telemetry.c).
FRAME_FIELDS = [("t_mono_ns", "u64", 0), ("hw_ts_ns", "u64", 8), ("cam_id", "u8", 16),
                ("frame_seq", "u32", 17), ("n_blobs", "u16", 21), ("exposure", "u16", 23),
                ("gain", "u16", 25), ("led_intensity", "u16", 27), ("dropped_capacity", "u16", 29)]
FRAME_ROW_SIZE = 31


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {extra}")


def raises(name, fn, expect: str):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the point is that SOMETHING loud comes out
        check(name, expect in str(exc), f"raised {type(exc).__name__}: {exc}")
        return
    check(name, False, "no exception raised")


def pgm_name(cam, seq, ts, n_blobs, exposure) -> str:
    return f"cam{cam}_t{ts:020d}_e{exposure}_s{seq:010d}_n{n_blobs}.pgm"


def make_capture(root: Path, groups, *, rows_written=None, dump=None, stale=()) -> Path:
    """A capture whose frame.bin records `groups` = [(seq, ts, [(cam, n_blobs, exposure), ...])].

    `dump` (default: exactly the recorded frames) is what actually lands in frames/, plus `stale`.
    """
    telem = root / "telemetry"
    frames = root / "frames"
    telem.mkdir(parents=True)
    frames.mkdir(parents=True)
    rows = bytearray()
    recorded = []
    for seq, ts, cams in groups:
        for cam, n_blobs, exposure in cams:
            row = bytearray(FRAME_ROW_SIZE)
            struct.pack_into("<Q", row, 0, ts)
            struct.pack_into("<Q", row, 8, ts)
            row[16] = cam
            struct.pack_into("<I", row, 17, seq)
            struct.pack_into("<H", row, 21, n_blobs)
            struct.pack_into("<H", row, 23, exposure)
            rows += row
            recorded.append(pgm_name(cam, seq, ts, n_blobs, exposure))
    (telem / "frame.bin").write_bytes(bytes(rows))
    (telem / "manifest.json").write_text(json.dumps({
        "version": 1, "clock": "monotonic", "start": {}, "types": "le",
        "streams": {"frame": {
            "file": "frame.bin", "row_size": FRAME_ROW_SIZE,
            "fields": [{"name": n, "type": t, "offset": o} for n, t, o in FRAME_FIELDS],
            "rows_written": len(recorded) if rows_written is None else rows_written,
            "overflow_total": 0}}}))
    for name in (recorded if dump is None else dump), stale:
        for entry in name:
            (frames / entry).write_bytes(b"P5\n1 1\n255\n\x00")
    return root


def four_cam_group(seq, ts):
    return (seq, ts, [(cam, 10 + cam, 20) for cam in range(CAM_COUNT)])


def main() -> int:
    print("[1] a directory that already holds exactly the recorded set is used as-is")
    with tempfile.TemporaryDirectory() as tmp:
        capture = make_capture(Path(tmp) / "clean", [four_cam_group(1, 1000), four_cam_group(2, 2000)])
        check("clean dump -> frames/ itself", frame_source(capture) == capture / "frames",
              str(frame_source(capture)))
        check("no view materialised for a clean dump", not (capture / VIEW_DIRNAME).exists())

    print("[2] a dead launch's leftovers are excluded, including when they reuse (cam, seq)")
    with tempfile.TemporaryDirectory() as tmp:
        groups = [four_cam_group(1, 5000), four_cam_group(2, 6000)]
        # Same seq/cam as the survivor's first group, different timestamp: what a relaunch leaves.
        stale = [pgm_name(cam, 1, 99, 3, 20) for cam in range(CAM_COUNT)] + \
                [pgm_name(cam, 7, 42, 3, 20) for cam in range(CAM_COUNT)]
        capture = make_capture(Path(tmp) / "dirty", groups, stale=stale)
        source = frame_source(capture)
        selected = sorted(p.name for p in source.iterdir())
        expected = sorted(pgm_name(cam, seq, ts, n, e)
                          for seq, ts, cams in groups for cam, n, e in cams)
        check("contaminated dump -> a view beside frames/", source == capture / VIEW_DIRNAME,
              str(source))
        check("the view is exactly the recorded set", selected == expected,
              f"{len(selected)} selected, {len(expected)} recorded")
        check("every stale (cam, seq) twin is excluded",
              not any(n in selected for n in stale))
        check("the view reads the survivor's image, not the twin's",
              all((source / n).resolve() == (capture / "frames" / n).resolve() for n in selected))
        check("view links are relative to the capture, not to this machine",
              all(not Path(p.readlink()).is_absolute() for p in source.iterdir()))

        print("[3] selection is deterministic, and re-selecting is a no-op")
        before = {p.name: p.readlink() for p in source.iterdir()}
        again = frame_source(capture)
        after = {p.name: p.readlink() for p in again.iterdir()}
        check("the second call selects the same frames", again == source and after == before,
              f"{len(before)} -> {len(after)}")

        print("[4] a view left holding the wrong frames is repaired, not trusted")
        (source / selected[0]).unlink()
        (source / "cam0_t00000000000000000042_e20_s0000000009_n3.pgm").symlink_to("../frames/nope")
        repaired = sorted(p.name for p in frame_source(capture).iterdir())
        check("a stale/broken view is brought back to the recorded set", repaired == expected,
              f"{len(repaired)} vs {len(expected)}")

    print("[5] a frame set short of its own recording is refused, never replayed")
    with tempfile.TemporaryDirectory() as tmp:
        groups = [four_cam_group(1, 1000), four_cam_group(2, 2000)]
        full = [pgm_name(cam, seq, ts, n, e) for seq, ts, cams in groups for cam, n, e in cams]
        short = make_capture(Path(tmp) / "short", groups, dump=full[:-1])
        raises("a recorded frame with no PGM refuses the capture",
               lambda: frame_source(short), "short of its own recording")

        trunc = make_capture(Path(tmp) / "trunc", groups, rows_written=len(full) + 4)
        raises("frame.bin shorter than the manifest declares refuses the capture",
               lambda: frame_source(trunc), "the recording is truncated")

        partial = make_capture(Path(tmp) / "partial",
                               [four_cam_group(1, 1000), (2, 2000, [(0, 10, 20), (1, 11, 20)])])
        raises("a recorded group missing a camera refuses the capture",
               lambda: frame_source(partial), f"not {CAM_COUNT}-camera")

        empty = Path(tmp) / "empty"
        (empty / "telemetry").mkdir(parents=True)
        raises("a capture with no frame source at all is refused",
               lambda: frame_source(empty), "no frames/ or euroc*/mav0")

    print("[6] two files claiming one frame identity are refused, not picked between")
    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp) / "frames"
        frames.mkdir()
        (frames / pgm_name(0, 1, 1000, 10, 20)).write_bytes(b"P5\n1 1\n255\n\x00")
        (frames / "cam0_t1000_e20_s1_n10.pgm").write_bytes(b"P5\n1 1\n255\n\x00")
        raises("an ambiguous identity refuses the directory",
               lambda: disk_index(frames), "same frame identity")

    print("[7] the real July-23 capture: the rule reproduces the frozen selection")
    if not (JULY23 / "telemetry" / "frame.bin").is_file():
        check("July-23 capture present", False, f"{JULY23} missing")
    else:
        keys = recorded_keys(JULY23)
        index = disk_index(JULY23 / "frames")
        check(f"frame.bin records {JULY23_RECORDED_FRAMES} frames",
              len(keys) == JULY23_RECORDED_FRAMES, str(len(keys)))
        check("every recorded frame is one full four-camera group",
              len(keys) == CAM_COUNT * len({seq for _cam, seq, *_ in keys}),
              str(len({seq for _cam, seq, *_ in keys})))
        check("the directory holds more files than the session recorded",
              len(index) > len(keys), f"{len(index)} on disk")
        check("(cam, seq) alone cannot select them",
              len({(cam, seq) for cam, seq, *_ in index}) < len(index),
              str(len({(cam, seq) for cam, seq, *_ in index})))
        source = frame_source(JULY23)
        n = sum(1 for p in source.iterdir() if p.is_file())
        check(f"the selected view holds {JULY23_RECORDED_FRAMES} readable frames",
              n == JULY23_RECORDED_FRAMES, str(n))
        check("the selected view is exactly the recorded frames",
              {p.name for p in source.iterdir()} == {index[key] for key in keys})

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
