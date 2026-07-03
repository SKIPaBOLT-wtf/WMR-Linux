#!/usr/bin/env bash
# Build + install the custom BORE kernel via linux-tkg, then sbsign the image with
# the kernel-MOK so it boots under Secure Boot. Non-interactive (driven by
# ~/.config/frogminer/linux-tkg.cfg). Run as your normal user (uses sudo internally).
#
#   scripts/kernel-build.sh
#
# GRUB default is left on your STOCK kernel (fallback). After verifying the new
# kernel, promote it with scripts/kernel-set-default.sh <KVER>.
set -euo pipefail
RESEARCH=$HOME/g2-linux-research
TKG=${TKG:-$RESEARCH/src/linux-tkg}
MOK_KEY=$RESEARCH/infra/mok-kernel/MOK-kernel.key
MOK_CERT=$RESEARCH/infra/mok-kernel/MOK-kernel.crt

# tee everything (stdout+stderr) to a persistent log for post-mortem
LOG_DIR=${LOG_DIR:-$RESEARCH/logs}
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/kernel-build-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "#### kernel-build.sh   $(date)   log=$LOG"
trap 'echo "#### kernel-build.sh EXIT rc=$? at $(date)"' EXIT

[ -d "$TKG" ] || { echo "ERR: linux-tkg not at $TKG (git clone Frogging-Family/linux-tkg)" >&2; exit 1; }
[ -f ~/.config/frogminer/linux-tkg.cfg ] || { echo "ERR: ~/.config/frogminer/linux-tkg.cfg missing" >&2; exit 1; }
command -v sbsign >/dev/null || { echo "ERR: sbsign missing (sudo apt install sbsigntool)" >&2; exit 1; }
[ -f "$MOK_KEY" ] && [ -f "$MOK_CERT" ] || { echo "ERR: kernel-MOK missing in $RESEARCH/infra/mok-kernel/" >&2; exit 1; }

# Detect any already-built deb so we can SKIP the rebuild / RESUME from where we left off.
# (exclude the -dbg debug-symbols variant from KVER detection)
detect_kver() {
    ls -1 "$TKG"/DEBS/linux-image-*_*_amd64.deb 2>/dev/null \
        | grep -v -- '-dbg_' | sort -V | tail -1 \
        | sed -E 's#.*/linux-image-([^_]+)_.*#\1#'
}
KVER=$(detect_kver || true)

if [ -n "$KVER" ] && [ -f "/boot/vmlinuz-$KVER" ]; then
    echo "== kernel $KVER already built + installed -> skipping tkg build =="
elif [ -n "$KVER" ]; then
    echo "== kernel $KVER deb already built (not installed) -> dpkg -i (no rebuild) =="
    sudo dpkg -i "$TKG"/DEBS/linux-image-${KVER}_*_amd64.deb \
                 "$TKG"/DEBS/linux-headers-${KVER}_*_amd64.deb \
                 "$TKG"/DEBS/linux-libc-dev_*_amd64.deb
else
    echo "== Building + installing BORE kernel via tkg (long; ~20-60 min) =="
    # `yes` auto-answers tkg's "Do you want to install ? Y/[n]:" prompt (default is n)
    ( cd "$TKG" && yes | ./install.sh install )
    KVER=$(detect_kver)
fi

[ -n "$KVER" ] || { echo "ERR: couldn't detect built kernel version in $TKG/DEBS" >&2; exit 1; }
IMG=/boot/vmlinuz-$KVER
[ -f "$IMG" ] || { echo "ERR: $IMG not installed (dpkg -i failure?)" >&2; exit 1; }
echo "$KVER" > "$TKG/DEBS/.last-kver"
echo "Built kernel: $KVER"

echo "== Secure Boot: signing $IMG with kernel-MOK =="
if sbverify --cert "$MOK_CERT" "$IMG" >/dev/null 2>&1; then
    echo "  already signed with this MOK."
else
    sudo sbsign --key "$MOK_KEY" --cert "$MOK_CERT" --output "$IMG.signed" "$IMG"
    sudo mv "$IMG.signed" "$IMG"
    echo "  signed OK."
fi
sudo update-grub

# mokutil --test-key's exit code is unreliable on some systems (it can return
# non-zero from a secondary "kernel trusted keyring" check even when the key
# IS enrolled in MokList). Parse the output text instead.
MOK_DER="$RESEARCH/infra/mok-kernel/MOK-kernel.der"
if mokutil --test-key "$MOK_DER" 2>&1 | grep -q "is already enrolled"; then
    echo "  kernel-MOK is enrolled (verified via mokutil)"
else
    echo
    echo "!! kernel-MOK NOT enrolled — signed kernel won't boot under Secure Boot until you run:"
    echo "     sudo mokutil --import $MOK_DER   (set a one-time password, reboot, choose Enroll MOK)"
fi

cat <<EOF

DONE: kernel $KVER built + signed. GRUB default unchanged (stock = fallback).
Next: scripts/g2-kernel-rebuild-all.sh handles NVIDIA; or manually:
  KVER=$KVER scripts/nvidia-build-modules.sh && KVER=$KVER scripts/nvidia-install-modules.sh
EOF
