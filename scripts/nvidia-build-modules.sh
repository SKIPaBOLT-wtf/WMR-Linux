#!/usr/bin/env bash
# Build patched NVIDIA kernel modules from source, against the running kernel.
# Version-agnostic: works on any nvidia-N-open branch.
set -euo pipefail
RESEARCH=$HOME/g2-linux-research

# tee everything (stdout+stderr) to a persistent log
LOG_DIR=${LOG_DIR:-$RESEARCH/logs}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/nvidia-build-modules-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "#### nvidia-build-modules.sh   $(date)   log=$LOG"
trap 'echo "#### nvidia-build-modules.sh EXIT rc=$? at $(date)"' EXIT

SRC=${SRC:-$RESEARCH/src/open-gpu-kernel-modules}
KVER=${KVER:-$(uname -r)}
SYSSRC=/lib/modules/$KVER/build

[ -d "$SRC" ] || { echo "ERR: $SRC not found" >&2; exit 1; }
[ -d "$SYSSRC" ] || { echo "ERR: kernel headers missing: $SYSSRC" >&2; exit 1; }
[ -f "$SRC/version.mk" ] || { echo "ERR: $SRC/version.mk missing" >&2; exit 1; }

NV_VERSION=$(awk -F= '/^NVIDIA_VERSION/ {gsub(/ /,"",$2); print $2}' "$SRC/version.mk")
echo "Building NVIDIA $NV_VERSION modules for kernel $KVER..."

cd "$SRC"
# Detect the target kernel's toolchain. A clang/LLVM-built kernel (CONFIG_CC_IS_CLANG=y)
# passes clang-only flags (e.g. -mretpoline-external-thunk) to module builds; gcc chokes
# on them. In that case the out-of-tree module must also be built with LLVM=1.
# The .config can live in either /boot/config-<KVER> (Ubuntu linux-headers convention)
# or $SYSSRC/.config (e.g. self-built trees) -- check both.
KCFG=""
for c in "/boot/config-$KVER" "$SYSSRC/.config"; do
    [ -f "$c" ] && KCFG="$c" && break
done
EXTRA_MAKE=()
if [ -n "$KCFG" ] && grep -q '^CONFIG_CC_IS_CLANG=y' "$KCFG"; then
    # NVIDIA's nested Makefiles don't reliably propagate LLVM=1 to the
    # actual conftest/Kbuild compile -- set every tool explicitly. Also
    # export them so any helper sub-shells inherit them.
    EXTRA_MAKE+=(LLVM=1 LLVM_IAS=1
                 CC=clang HOSTCC=clang CXX=clang++ HOSTCXX=clang++
                 LD=ld.lld HOSTLD=ld.lld
                 AR=llvm-ar NM=llvm-nm STRIP=llvm-strip
                 OBJCOPY=llvm-objcopy OBJDUMP=llvm-objdump READELF=llvm-readelf)
    export CC=clang HOSTCC=clang LD=ld.lld HOSTLD=ld.lld \
           AR=llvm-ar NM=llvm-nm STRIP=llvm-strip \
           OBJCOPY=llvm-objcopy OBJDUMP=llvm-objdump READELF=llvm-readelf
    echo "kernel $KVER built with clang (per $KCFG) -> full LLVM toolchain pinned for module build"
else
    echo "kernel $KVER built with gcc (per ${KCFG:-no config found}) -> default toolchain"
fi
# Clean any partial objects from a previous failed build so make starts fresh.
make -C "$SRC/kernel-open" clean >/dev/null 2>&1 || true
# Full output (no `| tail`) so the LOG captures everything.
make -j"$(nproc)" modules SYSSRC="$SYSSRC" "${EXTRA_MAKE[@]}"

echo
echo "Produced:"
for m in nvidia nvidia-modeset nvidia-drm nvidia-uvm; do
    ko="$SRC/kernel-open/$m.ko"
    if [ -f "$ko" ]; then
        printf "  %-20s %8d bytes  srcversion=%s\n" "$m.ko" "$(stat -c%s "$ko")" \
            "$(modinfo -F srcversion "$ko" 2>/dev/null || echo \?)"
    else
        echo "  MISSING: $ko"
    fi
done

echo
echo "Next: scripts/nvidia-install-modules.sh"
