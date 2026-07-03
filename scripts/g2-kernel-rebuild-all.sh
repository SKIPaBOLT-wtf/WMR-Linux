#!/usr/bin/env bash
# Top-level orchestrator: build+sign the custom BORE kernel, then rebuild the
# g2-patched NVIDIA open modules against it. Mirrors g2-nvidia-rebuild-all.sh.
#
#   scripts/g2-kernel-rebuild-all.sh              # build current tkg config + nvidia
#   scripts/g2-kernel-rebuild-all.sh --bump 7.1   # bump tkg _version, then build
#
# The stock kernel and its NVIDIA install are never touched -> always-available fallback.
set -euo pipefail
DIR=$(dirname "$(readlink -f "$0")")
RESEARCH=$HOME/g2-linux-research
TKG=$RESEARCH/src/linux-tkg
CFG=~/.config/frogminer/linux-tkg.cfg

# tee everything (stdout+stderr) to a persistent log for post-mortem
LOG_DIR=${LOG_DIR:-$RESEARCH/logs}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/g2-kernel-rebuild-all-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
# child scripts pick up the same LOG_DIR so their per-script logs land alongside
export LOG_DIR
echo "#### g2-kernel-rebuild-all.sh   $(date)   log=$LOG"
trap 'echo "#### g2-kernel-rebuild-all.sh EXIT rc=$? at $(date)"' EXIT

PROMOTE=1
ARGS=("$@")
for a in "${ARGS[@]}"; do
    [ "$a" = "--no-promote" ] && PROMOTE=0
done
if [ "${1:-}" = "--bump" ]; then
    NEWV="${2:-}"; [ -n "$NEWV" ] || { echo "Usage: $0 --bump <x.y[-latest]> [--no-promote]" >&2; exit 2; }
    echo "== bump tkg _version -> $NEWV =="
    sed -i -E "s/^_version=.*/_version=\"$NEWV\"/" "$CFG"
fi

echo "== Step 1/3: build + sign kernel =="
"$DIR/kernel-build.sh"
KVER=$(cat "$TKG/DEBS/.last-kver")

echo "== Step 2/3: build NVIDIA modules for $KVER =="
KVER="$KVER" "$DIR/nvidia-build-modules.sh"

echo "== Step 3/3: sign + install NVIDIA modules into $KVER =="
KVER="$KVER" "$DIR/nvidia-install-modules.sh"

if [ "$PROMOTE" = 1 ]; then
    echo "== promoting $KVER to persistent GRUB default (pass --no-promote to skip) =="
    "$DIR/kernel-set-default.sh" "$KVER"
fi

cat <<EOF

ALL DONE — kernel $KVER + g2-NVIDIA built & signed$([ "$PROMOTE" = 1 ] && echo " + set as default").
  Reboot. The stock kernel stays in GRUB > "Advanced options for Ubuntu" as the fallback.
  Verify after first boot: uname -r ; nvidia-smi ; <test G2 VR> ; cat /sys/kernel/sched_ext/state
  Trouble? Pick the stock kernel in GRUB; NVIDIA rollback: scripts/nvidia-rollback.sh <backup>
EOF
