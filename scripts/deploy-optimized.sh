#!/usr/bin/env bash
# deploy-optimized.sh — build, gate, and install the PGO(+BOLT)-optimized
# driver_monado.so from worktrees/g2-sota-stack.
#
# Flow (results/oob-feasibility-20260704/probe1-bolt-pgo.md methodology):
#   1. control   — verify the worktree is clean; replay BOTH yardstick workloads
#                  (xv1 + clutter band-2) with the existing build-cmake binary and
#                  record the control dev-CSV md5s.
#   2. profile   — GCC -fprofile-generate=<dir> -fprofile-update=atomic build of
#                  offline_vio_replay; train on both workloads (profiles merge).
#   3. rebuild   — -fprofile-use build of the replay, driver_monado.so and the
#                  three ctest suites. Linked with --emit-relocs so BOLT can
#                  post-process.
#   4. gate      — HARD byte-identity: PGO dev CSVs must equal the control md5s
#                  on BOTH full workloads, and the suites must be green. This is
#                  the whole proof: GCC PGO at -O2 was shown (probe 1, 13 runs)
#                  to change layout without flipping a single FP result; any
#                  deviation here means the profile drifted — the script aborts.
#   5. BOLT      — optional, only if llvm-bolt-22 is installed: perf-LBR profile
#                  the PGO replay on both workloads, perf2bolt, llvm-bolt the
#                  replay AND the driver (profile applied cross-binary by symbol
#                  name — the tracker objects are the same static libs in both).
#                  The bolted replay must pass the SAME byte-identity gate and a
#                  dlopen smoke test must pass on the bolted driver; otherwise
#                  BOLT is dropped and the PGO-only driver ships.
#   6. install   — cp the gated driver to ~/.local/share/steamvr-monado and
#                  print the md5. deploy.json is NOT touched by this script;
#                  update it by hand with the printed md5 + base commit.
#
# The band-2 frame subset is reconstructed if missing: frames of
# captures/20260612-153651-felt-validation-live with t >= 21707744382134
# (16116 files == telemetry frame.bin rows; results/b3-impl-20260704/notes.txt).
#
# Profiles are collected fresh every run — nothing is committed, so there is no
# stale-profile drift; the cost is ~15 min of build+replay time per deploy.
set -euo pipefail

DIR=$(dirname "$(readlink -f "$0")")
RESEARCH=$(cd "$DIR/.." && pwd)
WT=$RESEARCH/worktrees/g2-sota-stack
WORK=${DEPLOY_OPT_WORK:-$(mktemp -d /tmp/deploy-optimized.XXXXXX)}
PROF=$WORK/pgo-profile
CAMS=$RESEARCH/tools/telemetry/data/hmd-cameras-replay.json
CTL_L=$HOME/.config/monado/wmr/controller_A85K1111630014L.json
CTL_R=$HOME/.config/monado/wmr/controller_A85K5091930012R.json
XV1_FRAMES=$RESEARCH/captures/20260528-080421-xv-session1/frames
XV1_TELEM=$RESEARCH/captures/20260528-080421-xv-session1/telemetry
B2_CAPTURE=$RESEARCH/captures/20260612-153651-felt-validation-live
B2_FRAMES=/tmp/clutterframes-band2
B2_T0=21707744382134
DEST=$HOME/.local/share/steamvr-monado/bin/linux64/driver_monado.so
SUITES="tests_kalman_fusion|tests_constellation_pnp|tests_g2_telemetry"

hdr() { printf '\n== %s ==\n' "$1"; }

replay() { # replay <binary> <workload:xv1|band2> <outdir>
    local bin=$1 wl=$2 out=$3
    mkdir -p "$out"
    if [ "$wl" = xv1 ]; then
        "$bin" "$XV1_FRAMES" "$CAMS" "$XV1_TELEM" "$CTL_L" "$CTL_R" "$out" \
            > "$out.log" 2>&1
    else
        "$bin" "$B2_FRAMES" "$CAMS" "$B2_CAPTURE/telemetry" "$CTL_L" "$CTL_R" "$out" \
            > "$out.log" 2>&1
    fi
}

csv_md5() { md5sum "$1"/dev1.csv "$1"/dev2.csv | cut -d' ' -f1 | paste -sd:; }

gate() { # gate <binary> <tag> — byte-identity vs control on both workloads
    local bin=$1 tag=$2 wl
    for wl in xv1 band2; do
        replay "$bin" $wl "$WORK/out/$tag-$wl"
        local got want
        got=$(csv_md5 "$WORK/out/$tag-$wl")
        want=$(cat "$WORK/out/ctl-$wl.md5")
        if [ "$got" != "$want" ]; then
            echo "GATE FAIL [$tag/$wl]: $got != control $want"
            return 1
        fi
        echo "  gate OK   $tag/$wl  $got"
    done
}

hdr "worktree"
cd "$WT"
[ -z "$(git status --porcelain)" ] || { echo "worktree dirty — refusing"; exit 1; }
HEAD=$(git rev-parse --short=9 HEAD)
echo "  $WT @ $HEAD (clean)"

hdr "band-2 workload"
if [ "$(ls "$B2_FRAMES" 2>/dev/null | wc -l)" != 16116 ]; then
    echo "  reconstructing $B2_FRAMES (t >= $B2_T0)"
    rm -rf "$B2_FRAMES" && mkdir -p "$B2_FRAMES"
    (cd "$B2_CAPTURE/frames" && ls | awk -F'_t' -v t0=$B2_T0 \
        '{ts=$2; sub(/_.*/,"",ts); if (ts+0 >= t0) print}' \
        | while read -r f; do ln -s "$PWD/$f" "$B2_FRAMES/$f"; done)
    [ "$(ls "$B2_FRAMES" | wc -l)" = 16116 ] || { echo "band-2 rebuild wrong"; exit 1; }
else
    echo "  $B2_FRAMES present (16116 frames)"
fi

hdr "control (build-cmake @ HEAD)"
ninja -C build-cmake offline_vio_replay driver_monado.so >/dev/null
mkdir -p "$WORK/out"
for wl in xv1 band2; do
    replay build-cmake/tests/offline_vio_replay $wl "$WORK/out/ctl-$wl"
    csv_md5 "$WORK/out/ctl-$wl" > "$WORK/out/ctl-$wl.md5"
    echo "  control $wl  $(cat "$WORK/out/ctl-$wl.md5")"
done

hdr "PGO instrument + train"
mkdir -p "$PROF"
cmake -S "$WT" -B "$WORK/build-gen" -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXRT_HAVE_GST=OFF \
    -DCMAKE_C_FLAGS="-fprofile-generate=$PROF -fprofile-update=atomic" \
    -DCMAKE_CXX_FLAGS="-fprofile-generate=$PROF -fprofile-update=atomic" \
    > "$WORK/cmake-gen.log" 2>&1
ninja -C "$WORK/build-gen" offline_vio_replay > "$WORK/ninja-gen.log" 2>&1
replay "$WORK/build-gen/tests/offline_vio_replay" xv1 "$WORK/out/train-xv1"
replay "$WORK/build-gen/tests/offline_vio_replay" band2 "$WORK/out/train-band2"
echo "  trained: $(find "$PROF" -name '*.gcda' | wc -l) .gcda TUs"

hdr "PGO-use build"
cmake -S "$WT" -B "$WORK/build-use" -GNinja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DXRT_HAVE_GST=OFF \
    -DCMAKE_C_FLAGS="-fprofile-use=$PROF -fprofile-correction -Wno-missing-profile" \
    -DCMAKE_CXX_FLAGS="-fprofile-use=$PROF -fprofile-correction -Wno-missing-profile" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--emit-relocs" \
    -DCMAKE_MODULE_LINKER_FLAGS="-Wl,--emit-relocs" \
    > "$WORK/cmake-use.log" 2>&1
ninja -C "$WORK/build-use" offline_vio_replay driver_monado.so \
    tests_kalman_fusion tests_constellation_pnp tests_g2_telemetry \
    > "$WORK/ninja-use.log" 2>&1

hdr "gate: PGO byte-identity + suites"
gate "$WORK/build-use/tests/offline_vio_replay" pgo
(cd "$WORK/build-use" && ctest -R "$SUITES" --output-on-failure) \
    > "$WORK/ctest.log" 2>&1 || { echo "SUITES FAIL"; tail "$WORK/ctest.log"; exit 1; }
echo "  suites green"

SHIP_SO=$WORK/build-use/steamvr-monado/bin/linux64/driver_monado.so
SHIP_KIND="PGO"

hdr "BOLT (optional)"
if command -v llvm-bolt-22 >/dev/null; then
    for wl in xv1 band2; do
        perf record -e cycles:u -j any,u -o "$WORK/perf-$wl.data" -- \
            "$WORK/build-use/tests/offline_vio_replay" \
            $([ $wl = xv1 ] && echo "$XV1_FRAMES" || echo "$B2_FRAMES") "$CAMS" \
            $([ $wl = xv1 ] && echo "$XV1_TELEM" || echo "$B2_CAPTURE/telemetry") \
            "$CTL_L" "$CTL_R" "$WORK/out/lbr-$wl" > "$WORK/out/lbr-$wl.log" 2>&1
        perf2bolt-22 "$WORK/build-use/tests/offline_vio_replay" \
            -p "$WORK/perf-$wl.data" -o "$WORK/$wl.fdata" > "$WORK/p2b-$wl.log" 2>&1
    done
    merge-fdata-22 "$WORK/xv1.fdata" "$WORK/band2.fdata" > "$WORK/merged.fdata"
    BOLT_OPTS="-reorder-blocks=ext-tsp -reorder-functions=hfsort -split-functions -split-all-cold"
    llvm-bolt-22 "$WORK/build-use/tests/offline_vio_replay" \
        -o "$WORK/replay.bolt" -data "$WORK/merged.fdata" $BOLT_OPTS \
        > "$WORK/bolt-replay.log" 2>&1
    llvm-bolt-22 "$SHIP_SO" \
        -o "$WORK/driver.bolt.so" -data "$WORK/merged.fdata" $BOLT_OPTS \
        > "$WORK/bolt-driver.log" 2>&1
    if gate "$WORK/replay.bolt" bolt && \
       python3 -c "import ctypes,sys; ctypes.CDLL(sys.argv[1]).HmdDriverFactory" \
           "$WORK/driver.bolt.so" 2>/dev/null; then
        SHIP_SO=$WORK/driver.bolt.so
        SHIP_KIND="PGO+BOLT"
    else
        echo "  BOLT failed the gate or the dlopen smoke test — shipping PGO-only"
    fi
else
    echo "  llvm-bolt-22 not installed — shipping PGO-only"
fi

hdr "install ($SHIP_KIND, base $HEAD)"
cp "$SHIP_SO" "$DEST"
md5sum "$DEST"

hdr "runpath stamp + in-container load gate"
# Why $ORIGIN: vrserver runs inside the Steam Linux Runtime (sniper)
# pressure-vessel container, where LD_LIBRARY_PATH holds only SteamVR's own
# dirs + the pv override aliases. The vendored lib bundle next to the driver
# historically resolved only because a local patch in SteamVR's vrstartup.sh
# exported LD_LIBRARY_PATH=<bundle dir> — on 2026-07-06 Steam's file
# validation restored the pristine vrstartup.sh ("3 files missing") and every
# driver load broke (dlopen: libuvc.so.0 cannot open). DT_RUNPATH=$ORIGIN on
# the driver and every bundled lib makes resolution self-contained and immune
# to Steam-managed files ($HOME is bind-mounted at the identical path inside
# the container, so $ORIGIN is stable). Stamp anything new/unstamped here so
# no future deploy can regress this.
BUNDLE=$(dirname "$DEST")
patchelf --set-rpath '$ORIGIN' "$DEST"
find "$BUNDLE" -maxdepth 1 -name '*.so*' -type f \
    ! -name '*before-*' ! -name '*.pre-*' ! -name '*.bak' | while read -r so; do
    head -c4 "$so" | grep -q $'\x7fELF' || continue
    readelf -d "$so" 2>/dev/null | grep -qE '(RUNPATH|RPATH).*\$ORIGIN' \
        || { patchelf --set-rpath '$ORIGIN' "$so"; echo "  stamped $(basename "$so")"; }
done
SLR=$HOME/.local/share/Steam/steamapps/common/SteamLinuxRuntime_sniper/run
if [ -x "$SLR" ]; then
    env -i HOME="$HOME" "$SLR" -- python3 -c \
        "import ctypes; ctypes.CDLL('$DEST', mode=ctypes.RTLD_LOCAL|2).HmdDriverFactory" \
        || { echo "IN-CONTAINER dlopen FAIL — driver would not load in vrserver"; exit 1; }
    echo "  in-container dlopen (sniper, clean env, RTLD_NOW) OK"
else
    echo "  SLR sniper not found — skipped the in-container gate"
fi

echo "Done. Update results/benchmark-definitive-20260609/deploy.json with the md5 above."
