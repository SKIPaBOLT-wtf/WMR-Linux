# Preserve the working G2 display path

This directory preserves the additional source corrections used with the reference **HP Reverb G2, NVIDIA RTX 4080 SUPER, NVIDIA 595.91.07, Ubuntu 26.04 / GNOME Wayland / SteamVR** setup. It is a reconstruction record, not a general kernel-driver installation recommendation. A different GPU, NVIDIA release, compositor or headset needs its own capability and regression checks. Tracking work must not silently undo working display/presentation behavior.

## Exact source ancestry

The NVIDIA upstream base is `NVIDIA/open-gpu-kernel-modules` commit `9f087a6d4e86d85acc0ce1d354d6276fe5047b29` (`595.91.07`). The locally preserved research history applies three commits:

1. `0cab284d80985ed1646676ff731a884c08e38b15`: DisplayID/DSC/VSDB parsing and conformance corrections.
2. `4993d40035f0d783a8a1265d41d3dfe632a0bb83`: Wayland DRM leasing of VR HMDs.
3. `aefe80a8f1b65bff45119360c8f5bf5a69835a8c`: G2 maximum DisplayPort link configuration.

The companion project's original `patches/consolidated/nvidia/0001–0003` were exported against an older NVIDIA branch. Their original commit IDs and older `patches/INDEX.md` version descriptions must not be mistaken for this tested baseline.

**Reconstruction was verified on 2026-09-23:** applying those three patch files in order to the exact `595.91.07` base succeeds without changes and produces tree `9726da23d0fcdf70fee9919f9a3f87c1d289c9e2`, exactly the tree of `aefe80a8f1b65bff45119360c8f5bf5a69835a8c`. This closes the source-tree ancestry gap; commit IDs can differ because metadata differs. Verification used a temporary index/object directory and did not edit or build the live driver.

`nvidia-595.91.07-g2-vendor-bpc.patch` applies cleanly after that reconstructed tree. Its bytes exactly match the two-file diff in the working research source. SHA-256:

```text
ade6338474fdd75f91a3f2c03a5c6d1d33cb082a6082220f1b1b68dcb263fdca
```

The machine-readable check is `display-patch-verification.json`. The original upstream component licenses still apply; these patches retain their source context and do not relicense NVIDIA code.

## What the continuation patch changes

- Correct the G2 EDID manufacturer comparison to the value returned by this parser (`0x0E22`), and require G2 product `0x36C1`. The previous `0x220E` comparison used the wrong byte order and applied the intended device rule incorrectly. The replacement deliberately matches the specific model, not every HP monitor.
- Treat an undefined DisplayPort color-depth field (`0`) separately from an explicitly declared depth below 8. Only the latter follows the 6-bpc branch. This is the bounded version-specific backport associated with [NVIDIA open-gpu-kernel-modules PR 1275](https://github.com/NVIDIA/open-gpu-kernel-modules/pull/1275). The patch's presence here does not imply that an upstream release has accepted it or that another release needs it.

Earlier synthetic checks preserved all seven manufacturer/product matching cases and all nonzero bit-depth inputs 1–32, while the undefined case changed from 6 to 8. Those checks establish the branch behavior. They do not establish output on a different physical device.

## Presentation settings preserved outside the patch

The working reference setup also has these separate settings:

```text
nvidia-modeset module option: conceal_vrr_caps=1
SteamVR setting: steamvr.disableLinuxWaitForPresent=true
G2 mode: native 4320 x 2160 at 90 Hz
```

Both configuration values and the loaded module's `conceal_vrr_caps=Y` were read back on 2026-09-23. The module option is persisted in a dedicated modprobe file and was included in the reference initramfs. The user reported that horizontal tearing disappeared after the prior reboot. This document does not claim a new physical test during repository preparation, nor that either setting alone caused the improvement.

Keep the reference profile's existing frame timing and display configuration while investigating tracking. These settings are explicitly NVIDIA/SteamVR-stack workarounds, not generic recommendations for AMD, Intel, another headset or future NVIDIA versions. The continuation installer must not set them system-wide merely because it detects a WMR USB device.

## Future build and rollback requirements

Before reproducing a kernel module, verify the exact upstream source, intended patch sequence, NVIDIA userspace/kernel version agreement, target kernel headers and `vermagic`, compiler requirements, symbol-version compatibility and module-signing/Secure Boot procedure. A successful patch application does not qualify the resulting binary. Detect equivalent upstream fixes rather than applying a patch twice.

Stage builds outside active module directories. Record source tree, toolchain, artifact hashes and build IDs; back up the currently effective signed module, module-option file and relevant SteamVR setting. Keep private signer identity, key material, hostnames and absolute personal paths out of public records. Do not distribute a kernel module built for the reference machine as a universal package.

Installation must be an explicit supported profile step with a rollback that restores the previous signed module and exact configuration, regenerates dependency/initramfs state using the host distribution's established tooling, and verifies the loaded module after reboot. A machine that requires Secure Boot signing must retain its own signing process. Never infer a successful reboot from a successful build or an on-disk file replacement.

This publication contains only source patches and a sanitized verification record. It contains **no raw EDID, headset serial, signed/unsigned binary, signing key, initramfs, private log or local configuration dump**. No display setting, module, boot image or active VR process was changed to prepare it.
