# WMR Linux — HP Reverb G2, Monado and SteamVR

An experimental continuation of [AshishKumar4/Project-VR](https://github.com/AshishKumar4/Project-VR), focused on reliable Windows Mixed Reality hardware on Linux, reproducible tracking fixes, and reversible installation. Runtime changes live in the companion [monado-wmr fork](https://github.com/SKIPaBOLT-wtf/monado-wmr/tree/g2-stability).

**Current status: display and DiRT Rally 2.0 VR have worked on the reference system; head and controller tracking are not accepted for a stable release.** This is a development snapshot, not a promise of comfortable or plug-and-play VR. No consumer binary release is qualified yet.

HP Reverb G2 uses cameras and inertial sensors; it does not require Lighthouse base stations. Headset support is native Linux. A Windows game such as DiRT Rally 2.0 still uses Proton; that is separate from the headset driver.

## Start here

- **Users:** [status and limitations](docs/current/STATUS.md), [tracking questions](docs/current/TRACKING-FAQ.md), [hardware profiles](profiles/README.md), [installation and rollback](docs/INSTALLATION.md).
- **Developers:** [architecture and open questions](docs/current/TRACKING-ARCHITECTURE.md), [roadmap](docs/ROADMAP.md), [contributing](CONTRIBUTING.md), [provenance](docs/PROVENANCE.md).
- **An agent continuing local tracking work:** copy the standalone [local G2 tracking prompt](prompts/LOCAL-TRACKING.md) to diagnose and improve the existing installation while preserving working behavior.
- **An agent developing the public project:** copy the standalone [project development prompt](prompts/CONTINUE.md) for documented iterations, packaging, portability, and publication. Both prompts discover current state and work at later versions; neither treats a historical artifact as an instruction to reinstall it.

```bash
python3 package/wmrctl.py doctor
python3 package/wmrctl.py profile profiles/hp-reverb-g2-nvidia-ubuntu.json
```

These commands only inspect or validate. They do not install drivers, start VR, record cameras, change the desktop, or upload anything.

## Reference system and scope

| Component | Development reference | Qualification |
|---|---|---|
| Headset | HP Reverb G2, DisplayPort + USB | Display works; tracking unresolved |
| Graphics | NVIDIA RTX 4080 SUPER, driver 595.91.07 with local patches | Reported horizontal tearing resolved |
| Desktop | Ubuntu 26.04.1, GNOME Wayland, kernel 7.0.0-31 | Single development system |
| CPU / memory | Ryzen 7 5700X3D / 32 GB class | Reference only, not minimum requirements |
| Runtime | Monado OpenVR driver + pinned Basalt backend + SteamVR | Current source tests do not imply physical acceptance |
| Game | DiRT Rally 2.0 through SteamVR / Proton | User has driven in VR; long-session tracking not accepted |
| Other WMR headsets, AMD/Intel GPUs, other Linux releases | Profile extension targets | Untested until demonstrated |

Device calibration belongs to each device. Profiles contain capabilities and policies, never copied calibration from another owner's headset. Similar hardware does not establish compatible optics, USB protocols, camera layouts, or controller transforms.

## What this continuation contributes

The source snapshot includes fixes for dropped HMD sensor reports during controller initialization, historical controller pose queries, an iterative filter residual sign, idle sensor reports, world-frame cache consistency, and stale optical masks. Tests establish specific code behavior. They do **not** establish that remaining drift, controller jumps, aim alignment, or micro-jitter are solved.

The [status ledger](docs/current/STATUS.md) records rejected candidates as well as retained fixes. A rebuilt Basalt candidate is explicitly rejected; the existing release backend remains in use. The optional learned gyro-bias consumer and Windows camera-extrinsic override are off.

## Help develop or test

Useful contributions include deterministic sensor-fusion regressions, clock/frame audits, reproducible backend builds, device profiles, and packaging that can roll back cleanly. See the issue templates and [release gates](docs/RELEASE-GATES.md). Report the version and an anonymized symptom description first; do not attach room recordings or full system logs to public issues.

This project preserves the original author's history and credits. [Original project documentation](docs/UPSTREAM-README.md) describes its own results; those claims are not inherited as test results for this fork. Existing upstream automation is retained for provenance, but is not this fork's recommended installer.

## Attribution and licenses

Original integration: [AshishKumar4/Project-VR](https://github.com/AshishKumar4/Project-VR) and [AshishKumar4/monado-wmr](https://github.com/AshishKumar4/monado-wmr). Foundations: Monado / Collabora and contributors, thaytan's WMR work, Basalt and its Monado integration, NVIDIA's open GPU modules, and Valve's OpenVR interfaces. No vendor endorsement is implied.

Existing files retain their original notices and terms. New continuation files are covered by [LICENSE-CONTINUATION](LICENSE-CONTINUATION). Patches retain the target project's license. See [provenance](docs/PROVENANCE.md) before redistribution; do not treat the original README's informal licensing language as permission to relicense third-party files.
