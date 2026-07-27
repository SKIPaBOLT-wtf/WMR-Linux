#!/usr/bin/env python3
"""Which frames a replay of a capture must read -- the third pinned harness input.

`replay_contract` pins the camera config and the G2_REPLAY_* environment. This module pins the
remaining one, and for the same reason: the frame source has to be a property of the capture, not
of whatever happens to be lying in its directory.

A driver launch that dies before it finishes still leaves its PGMs behind, and the next launch
dumps into the SAME `frames/` directory. Eight of this tree's captures carry such a stale tail; on
`20260723-112531-comprehensive-stack` it is 3,484 files whose `frame_seq` restarts at 140 under the
survivor's 1222, so 1,700 `(cam, seq)` keys resolve to two different images. Listing the directory
therefore does not describe any single recorded session, and picking per key would pick arbitrarily.

The recording itself says which files are its own. `telemetry/frame.bin` has one row per emitted
camera frame carrying `(cam_id, frame_seq, hw_ts_ns, n_blobs, exposure)`, and the dumper names each
PGM from that same row (blobwatch.c:1690, `cam%u_t%020d_e%u_s%010u_n%u.pgm`). So:

    THE RECORDED FRAME SET IS EXACTLY THE PGMs WHOSE FULL NAME TUPLE IS A ROW OF frame.bin.

Full tuple, not `(cam, seq)`: the stale tail reuses sequence numbers but never reproduces a
survivor's timestamp, so the five-field identity is unique across the contaminated directory while
the two-field key is not. On the July-23 capture the rule selects 34,976 files -- every one of the
8,744 four-camera groups `frame.bin` recorded, no group split across sessions, nothing arbitrary.

Selection is only trusted when it is complete. A recorded row with no PGM on disk, a name tuple
present twice, or a recorded group that is not a full camera set means the directory cannot serve
the recording, and this module raises instead of replaying a silently smaller world -- an unnoticed
short frame set changes every downstream number while looking like a clean run.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import g2_geom as G
from manifest import Manifest

#: The dumper's filename format. The five captured fields are the whole of a PGM's identity, and
#: this is the only place that knows how they are spelled.
PGM_RE = re.compile(r"^cam(?P<cam>\d+)_t(?P<ts>\d+)_e(?P<exp>\d+)_s(?P<seq>\d+)_n(?P<n>\d+)\.pgm$")

#: Cameras per recorded frame group. The harness reconstructs one callback frontier per group and
#: refuses a partial one, so a capture whose recording is missing a camera is not replayable.
CAM_COUNT = 4

#: Where a de-contaminated view is materialised: symlinks into `frames/`, next to it, named for
#: what it is. Derived and disposable -- it is re-verified against `frame.bin` on every use.
VIEW_DIRNAME = "frames.recorded"

FrameKey = tuple[int, int, int, int, int]  # (cam_id, frame_seq, hw_ts_ns, n_blobs, exposure)


def recorded_keys(capture: Path | str) -> set[FrameKey]:
    """The frame identities `capture`'s telemetry recorded. Raises if the recording is unusable."""
    telem = Path(capture) / "telemetry"
    manifest = Manifest.load(telem)
    rows = G.load_stream(telem, manifest, "frame")
    # rows_written is the producer's own count, and the only witness to a frame.bin truncated
    # mid-stream. Captures older than the counter record 0 for every stream; that is "not counted",
    # not "counted zero" -- a genuinely empty stream also has zero whole rows and still agrees here.
    declared = manifest.streams["frame"].rows_written
    if declared and int(declared) != rows.shape[0]:
        raise RuntimeError(
            f"{capture}: frame.bin holds {rows.shape[0]} whole rows but the manifest declares "
            f"{declared} written -- the recording is truncated, not a frame authority")

    keys = set(zip(rows["cam_id"].tolist(), rows["frame_seq"].tolist(), rows["hw_ts_ns"].tolist(),
                   rows["n_blobs"].tolist(), rows["exposure"].tolist()))
    if len(keys) != rows.shape[0]:
        raise RuntimeError(
            f"{capture}: {rows.shape[0] - len(keys)} frame.bin row(s) share an identity -- the "
            f"recording cannot name its own frames")

    groups: dict[int, int] = {}
    for _cam, seq, *_rest in keys:
        groups[seq] = groups.get(seq, 0) + 1
    partial = sorted(seq for seq, count in groups.items() if count != CAM_COUNT)
    if partial:
        raise RuntimeError(
            f"{capture}: {len(partial)} recorded frame group(s) are not {CAM_COUNT}-camera (first: "
            f"seq={partial[0]} cams={groups[partial[0]]}) -- the harness cannot build a callback "
            f"frontier for a partial group")
    return keys


def disk_index(frames: Path | str) -> dict[FrameKey, str]:
    """Filename per frame identity in `frames`. Raises when two files claim one identity."""
    index: dict[FrameKey, str] = {}
    for entry in Path(frames).iterdir():
        match = PGM_RE.match(entry.name)
        if match is None:
            continue  # not a frame the dumper named; it cannot be one frame.bin recorded
        key: FrameKey = (int(match["cam"]), int(match["seq"]), int(match["ts"]),
                         int(match["n"]), int(match["exp"]))
        if key in index:
            raise RuntimeError(
                f"{frames}: {entry.name} and {index[key]} claim the same frame identity -- "
                f"the recorded frame cannot be selected unambiguously")
        index[key] = entry.name
    return index


def _sync_view(view: Path, frames: Path, names: set[str]) -> None:
    """Make `view` hold exactly one symlink per name, pointing at its sibling in `frames`.

    Targets are relative (`../frames/<name>`) so the view keeps working wherever the capture is
    mounted or moved to; it is a description of the recording, not of one machine's paths.
    """
    view.mkdir(parents=True, exist_ok=True)
    for stale in {p.name for p in view.iterdir()} - names:
        (view / stale).unlink()
    for name in names:
        link, target = view / name, Path("..") / frames.name / name
        if link.is_symlink() and os.readlink(link) == str(target):
            continue
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    readable = {p.name for p in view.iterdir() if p.is_file()}
    if readable != names:
        raise RuntimeError(
            f"{view}: view resolves to {len(readable)} readable frames, the recording has "
            f"{len(names)} -- refusing to replay a frame set that does not match frame.bin")


def frame_source(capture: Path | str) -> Path:
    """The directory a replay of `capture` must read frames from.

    The capture's own `frames/` when it holds exactly the recorded set, a de-contaminated symlink
    view of it when a dead launch left extra files behind, or the EuRoC `mav0` dir for captures
    recorded without a controller PGM dump. Raises when no frame source can serve the recording.
    """
    capture = Path(capture)
    frames = capture / "frames"
    if not frames.is_dir():
        for euroc in sorted(capture.glob("euroc*")):
            if (euroc / "mav0" / "cam0" / "data").is_dir():
                return euroc / "mav0"
        raise FileNotFoundError(f"no frames/ or euroc*/mav0 frame source in {capture}")

    recorded = recorded_keys(capture)
    index = disk_index(frames)
    absent = recorded - set(index)
    if absent:
        cam, seq, ts, n, exp = sorted(absent)[0]
        raise RuntimeError(
            f"{capture}: {len(absent)} recorded frame(s) have no PGM in {frames} (first: cam={cam} "
            f"seq={seq} t={ts}) -- the dump is short of its own recording")
    if len(index) == len(recorded):
        return frames
    _sync_view(capture / VIEW_DIRNAME, frames, {index[key] for key in recorded})
    return capture / VIEW_DIRNAME
