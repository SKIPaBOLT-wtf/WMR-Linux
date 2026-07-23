#!/usr/bin/env bash
# g2-repatch-all.sh — single idempotent verify/repair entry point for every
# patched/custom surface on this machine. Default mode VERIFIES (read-only)
# and prints OK / DRIFT / INFO per surface; --apply repairs drifted surfaces
# by re-running the owning script (nothing else). Never reboots, never bumps
# versions: staged flips (e.g. NVIDIA 595.84) stay behind their own runbooks.
#
#   scripts/g2-repatch-all.sh            # verify all surfaces, exit 1 on drift
#   scripts/g2-repatch-all.sh --apply    # repair drifted surfaces (uses sudo)
#
# Surfaces (owners in parentheses):
#   1. tkg kernel        image present + MOK-signed + GRUB saved default + cmdline
#                        (scripts/kernel-build.sh, kernel-set-default.sh)
#   2. NVIDIA modules    installed .ko MOK-signed + version == running userspace
#                        (scripts/nvidia-install-modules.sh)
#   3. apt holds         patched/self-built package families all on hold
#   4. mutter            installed == newest +g2~ deb (scripts/mutter-install-debs.sh)
#   5. root state        sudoers/limits/linux-tools (g2-studio scripts/setup-system.sh)
#   6. VR user surfaces  environment.d, openvrpaths, udev, driver_monado.so
#
# Runbook: docs/update-repatch-runbook.md
set -euo pipefail

DIR=$(dirname "$(readlink -f "$0")")
RESEARCH=$(cd "$DIR/.." && pwd)
G2_STUDIO=${G2_STUDIO:-$HOME/g2-studio}
NVSRC=$RESEARCH/src/open-gpu-kernel-modules
MUTTER_DEBS=$RESEARCH/src/mutter-patch
KERNEL_MOK_CRT=$RESEARCH/infra/mok-kernel/MOK-kernel.crt
MODULE_MOK_CN="White0racle Secure Boot Module Signature key"
KVER=$(uname -r)

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

DRIFT=0
ok()    { printf '  OK    %s\n' "$1"; }
info()  { printf '  INFO  %s\n' "$1"; }
drift() { printf '  DRIFT %s\n' "$1"; DRIFT=1; }
hdr()   { printf '\n== %s ==\n' "$1"; }
fix()   { # fix <description> <cmd...> — run repair only under --apply
    if [ "$APPLY" = 1 ]; then printf '  FIX   %s\n' "$1"; shift; "$@";
    else printf '  FIX?  %s (run with --apply)\n' "$1"; fi
}

# --- 1. tkg kernel --------------------------------------------------------
hdr "kernel ($KVER)"
case "$KVER" in
    *tkg*) ok "running the tkg kernel" ;;
    *)     info "NOT running a tkg kernel — fallback boot? (expected only during recovery)" ;;
esac
IMG=/boot/vmlinuz-$KVER
if [ -f "$IMG" ]; then
    if sbverify --cert "$KERNEL_MOK_CRT" "$IMG" >/dev/null 2>&1; then
        ok "$IMG signed with kernel-MOK"
    else
        drift "$IMG NOT signed with kernel-MOK (Secure Boot fallback will fail)"
        fix "re-sign kernel image (kernel-build.sh is idempotent, skips the build)" "$DIR/kernel-build.sh"
    fi
else
    drift "$IMG missing"
fi
if grep -q '^GRUB_DEFAULT=saved' /etc/default/grub; then
    SAVED=$(grub-editenv /boot/grub/grubenv list 2>/dev/null | sed -n 's/^saved_entry=//p')
    if [ -n "$SAVED" ]; then
        ok "GRUB saved default pinned: $SAVED"
    else
        drift "GRUB_DEFAULT=saved but saved_entry UNSET — tkg boots only by version-sort luck; a higher-versioned stock kernel would steal the default"
        fix "pin GRUB default to $KVER" "$DIR/kernel-set-default.sh" "$KVER"
    fi
else
    drift "GRUB_DEFAULT is not 'saved' in /etc/default/grub"
    fix "pin GRUB default to $KVER" "$DIR/kernel-set-default.sh" "$KVER"
fi
for opt in mitigations=off nvidia-drm.modeset=1; do
    grep -qw "$opt" /proc/cmdline \
        && ok "cmdline has $opt" \
        || drift "cmdline missing $opt (check GRUB_CMDLINE_LINUX* in /etc/default/grub, then update-grub + reboot)"
done
dpkg -s "linux-headers-$KVER" >/dev/null 2>&1 \
    && ok "linux-headers-$KVER installed (module sign-file available)" \
    || drift "linux-headers-$KVER missing — NVIDIA module signing needs it (dpkg -i $RESEARCH/src/linux-tkg/DEBS/linux-headers-*.deb)"

# --- 2. NVIDIA kernel modules --------------------------------------------
hdr "NVIDIA modules"
NV_RUNNING=$(sed -n 's/^NVRM version.*Kernel Module for x86_64  \([0-9.]*\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
NV_STAGED=$(awk -F= '/^NVIDIA_VERSION/ {gsub(/ /,"",$2); print $2; exit}' "$NVSRC/version.mk" 2>/dev/null || true)
[ -n "$NV_RUNNING" ] && info "running userspace/RM: $NV_RUNNING; repo staged: ${NV_STAGED:-?} ($(git -C "$NVSRC" branch --show-current 2>/dev/null))"
[ -n "$NV_STAGED" ] && [ "$NV_STAGED" != "$NV_RUNNING" ] \
    && info "staged $NV_STAGED != installed $NV_RUNNING — version flips are user-gated (docs/render-l8-59584-runbook.md); this script only repairs the INSTALLED version"
NV_DIR=/lib/modules/$KVER/kernel/nvidia-595-open
NV_KO_DRIFT=0
if [ -d "$NV_DIR" ]; then
    for m in nvidia nvidia-modeset nvidia-drm nvidia-uvm; do
        ko=$NV_DIR/$m.ko
        if [ ! -f "$ko" ]; then drift "$ko missing"; NV_KO_DRIFT=1; continue; fi
        signer=$(modinfo -F signer "$ko" 2>/dev/null || true)
        ver=$(modinfo -F version "$ko" 2>/dev/null || true)
        if [ "$signer" != "$MODULE_MOK_CN" ]; then
            drift "$m.ko signer='$signer' (not our MOK) — clobbered by a package or never installed"
            NV_KO_DRIFT=1
        elif [ -n "$NV_RUNNING" ] && [ "$ver" != "$NV_RUNNING" ]; then
            info "$m.ko on disk is $ver, running is $NV_RUNNING (pending reboot?)"
        else
            ok "$m.ko $ver, MOK-signed"
        fi
    done
else
    drift "$NV_DIR missing entirely"
    NV_KO_DRIFT=1
fi
if [ "$NV_KO_DRIFT" = 1 ]; then
    if [ "$NV_STAGED" = "$NV_RUNNING" ]; then
        fix "rebuild+sign+install patched modules" "$DIR/g2-nvidia-rebuild-all.sh"
    else
        info "cannot auto-repair: repo is staged at $NV_STAGED but installed userspace is $NV_RUNNING —"
        info "  git -C $NVSRC checkout g2-patches-on-$NV_RUNNING, then $DIR/g2-nvidia-rebuild-all.sh"
    fi
fi

# --- 3. apt holds ----------------------------------------------------------
hdr "apt holds"
HELD=$(apt-mark showhold)
# Every installed package in these self-built/patched families must be held.
hold_family() { # hold_family <regex> <label>
    local missing
    missing=$(dpkg-query -W -f '${db:Status-Abbrev} ${Package}\n' 2>/dev/null \
        | awk '$1=="ii"||$1=="hi"{print $2}' | grep -E "$1" | grep -vxF -f <(echo "$HELD") || true)
    if [ -n "$missing" ]; then
        drift "$2 not held: $(echo "$missing" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        fix "apt-mark hold $2" sudo apt-mark hold $missing
    else
        ok "$2 family held"
    fi
}
hold_family '^(lib)?nvidia-.*-595|^nvidia-(driver|utils|compute-utils|kernel-common|kernel-source|firmware)-595|^xserver-xorg-video-nvidia-595|^linux-modules-nvidia-595' \
    "NVIDIA 595 (patched .ko must not be clobbered)"
hold_family '^(mutter|mutter-common|mutter-common-bin|gir1\.2-mutter-18|libmutter-18-0)$' \
    "mutter +g2~ (distro point release must not clobber)"
hold_family '^linux-(image|headers)-generic|^linux-generic' \
    "kernel meta (a new stock ABI would pull unpatched linux-modules-nvidia)"

# --- 4. mutter -------------------------------------------------------------
hdr "mutter"
MUTTER_INST=$(dpkg-query -W -f '${Version}' mutter 2>/dev/null || true)
MUTTER_NEWEST=$(ls -1 "$MUTTER_DEBS"/mutter_*+g2~*_amd64.deb 2>/dev/null | sort -V | tail -1 \
    | sed -E 's#.*/mutter_([^_]+)_.*#\1#' || true)
case "$MUTTER_INST" in
    *+g2~*) ok "installed mutter is patched: $MUTTER_INST" ;;
    *)      drift "installed mutter '$MUTTER_INST' is NOT a +g2~ build" ;;
esac
if [ -n "$MUTTER_NEWEST" ] && [ "$MUTTER_INST" != "$MUTTER_NEWEST" ]; then
    drift "newest built +g2~ deb is $MUTTER_NEWEST but installed is $MUTTER_INST"
    fix "install newest +g2~ mutter debs (takes effect at next GNOME session)" sudo "$DIR/mutter-install-debs.sh"
fi

# --- 5. root-owned system state (g2-studio) --------------------------------
hdr "root state (g2-studio setup-system.sh)"
ROOT_DRIFT=0
[ -f /etc/sudoers.d/g2-studio ] && ok "/etc/sudoers.d/g2-studio present" \
    || { drift "/etc/sudoers.d/g2-studio missing"; ROOT_DRIFT=1; }
[ -f /etc/security/limits.d/99-g2-vr-rtprio.conf ] && ok "rtprio limits present" \
    || { drift "/etc/security/limits.d/99-g2-vr-rtprio.conf missing"; ROOT_DRIFT=1; }
[ -e "/usr/lib/linux-tools/$KVER" ] && ok "linux-tools resolves for $KVER (cpupower works)" \
    || { drift "/usr/lib/linux-tools/$KVER missing — cpupower silently no-ops"; ROOT_DRIFT=1; }
[ "$ROOT_DRIFT" = 1 ] && fix "re-run g2-studio system setup" sudo "$G2_STUDIO/scripts/setup-system.sh"

# conceal_vrr_caps=1 (2026-07-15): NVIDIA cross-head VRR bug (their #5801122) — a VRR-capable
# desktop monitor (the 360Hz DP-3) disturbs the leased HMD's pacing even with desktop VRR off;
# concealing VRR caps keeps fixed 360Hz/4K modes intact. Boot-time option. STATUS 2026-07-23:
# one unexplained G2 vsync-stall on its first session (WaitForPresent stall -> compositor
# watchdog abort at ~60s), immediate relaunch healthy — verdict OPEN, keep + watch; if the
# stall recurs under it, revert (sudo rm + reboot) and re-adjudicate.
VRRF=/etc/modprobe.d/g2-conceal-vrr.conf
grep -qs 'conceal_vrr_caps=1' "$VRRF" && ok "nvidia-modeset conceal_vrr_caps=1 pinned ($VRRF; verdict open, watch vsync stalls)" \
    || { drift "$VRRF missing conceal_vrr_caps=1 (tearing A/B lever; verdict open)"; \
         info "  echo 'options nvidia-modeset conceal_vrr_caps=1' | sudo tee $VRRF"; }

# --- 6. VR user surfaces ----------------------------------------------------
hdr "VR user surfaces"
ENVF=$HOME/.config/environment.d/g2-vr.conf
if [ -f "$ENVF" ]; then
    for k in WMR_SLAM VIT_SYSTEM_LIBRARY_PATH SLAM_SUBMIT_FROM_START WMR_AUTOEXPOSURE; do
        grep -q "^$k=" "$ENVF" && ok "environment.d: $k set" || drift "environment.d: $k MISSING from $ENVF"
    done
else
    drift "$ENVF missing (head tracking dead-reckons without SLAM_SUBMIT_FROM_START)"
fi
VRPATH=$HOME/.config/openvr/openvrpaths.vrpath
if python3 - "$VRPATH" <<'EOF' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if any('steamvr-monado' in p for p in d.get('external_drivers', [])) else 1)
EOF
then ok "openvrpaths external_drivers has steamvr-monado"
else drift "openvrpaths missing steamvr-monado external driver (SteamVR/vrpathreg may have rewritten it)"; fi
[ -f /etc/udev/rules.d/70-xrhardware.rules ] && ok "xr-hardware udev rules present" \
    || drift "/etc/udev/rules.d/70-xrhardware.rules missing (HMD device permissions)"
DRIVER=$HOME/.local/share/steamvr-monado/bin/linux64/driver_monado.so
[ -f "$DRIVER" ] && ok "driver_monado.so deployed ($(md5sum "$DRIVER" | cut -c1-12)…)" \
    || drift "$DRIVER missing (ninja -C src/monado-thaytan/build-cmake driver_monado.so, then cp)"
VRSERVER=$HOME/.local/share/Steam/steamapps/common/SteamVR/bin/linux64/vrserver
if [ -x "$VRSERVER" ]; then
    getcap "$VRSERVER" | grep -q cap_sys_nice \
        && ok "vrserver has cap_sys_nice" \
        || info "vrserver setcap absent (Steam update wiped it; launcher reapplies at session start, rtprio limit is the backstop)"
fi

# ---------------------------------------------------------------------------
echo
if [ "$DRIFT" = 0 ]; then
    echo "ALL SURFACES OK"
else
    if [ "$APPLY" = 1 ]; then
        echo "REPAIRS ATTEMPTED — re-run without --apply to confirm clean"
    else
        echo "DRIFT DETECTED — re-run with --apply to repair"
    fi
    exit 1
fi
