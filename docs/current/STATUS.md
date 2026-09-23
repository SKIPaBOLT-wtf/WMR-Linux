# Current development state

Snapshot: 2026-09-23. Update this file when superseded; historical reports are not deployment instructions.

**Release readiness: experimental source snapshot. Head/controller tracking is not physically accepted. No easy-install consumer binary is qualified.** This publication changes documentation and source packaging, not the live VR installation.

## Preserved behavior

- G2 display and DiRT Rally 2.0 VR have been confirmed by the user on the reference system.
- Horizontal tearing was reported resolved after the NVIDIA presentation workaround. Preserve the version-specific `conceal_vrr_caps=1` setting and SteamVR `disableLinuxWaitForPresent=true` when reproducing that reference; do not prescribe them universally.
- Native display mode is 4320 x 2160 at 90 Hz. The previously verified render scale is 140%; the mode is not a measurement of achieved game frame rate.
- Home controller models/binding routes load, but physical hand/ray alignment is not accepted. Xbox input has worked; it is separate from optical tracking.
- Most recent prior runtime validation stopped SteamVR/game after a normal-launcher check. Always recheck live state before acting.

## Source fixes and evidence boundaries

| Area | Established result | Remaining boundary |
|---|---|---|
| HMD HID reports during controller firmware setup | Full sensor packets are dispatched rather than discarded by short reply reads; bounded deadline/lifecycle handling tested. One live capture: 991.74 Hz, maximum gap 1.99961 ms over 59.579 s | Transport continuity does not establish pose accuracy |
| Controller iterative residual | Corrected fixed-prior IEKF innovation sign with numerical regression | Does not solve every optical ambiguity |
| Idle controller reports | Raw-zero status sentinel separated from real IMU; qualified idle/wake semantics and stale adapter state | Moving-head/controller comfort remains unaccepted |
| Historical controller pose | Historical query no longer borrows present angular rate; bounded residual prediction, checkpoints/epochs | Full real-world timing chain needs continued validation |
| Optical acceptance | Rejected optical estimates do not update accepted gyro reference or retain failed trial state | LED association/mirror ambiguity still relevant |
| World/presentation consistency | Common reanchors and paired presentation transforms handled coherently | Persistent raw VIO drift remains |
| Cached optical geometry | Cache reanchored in CV world with estimator; observation age preserved | Live validation did not exercise the populated stale/reanchor branch |
| Controller feature masks | Existing 150 ms freshness rule used for untracked geometry | Mask timing differs between some offline and live paths |
| HMD history/concurrency | Missing-prefix, first-pose and single-consumer/mutex cases tested; finite fallback and truthful flags | Not a global proof of estimator health |
| Zero visual observations | Explicit successful count of zero invalidates HMD tracking; unsupported/error means unknown | Positive features do not imply stable or accurate tracking |
| Factory prediction calibration | Calibration and integration-order corrections retained | Learned-bias export is still disabled |

Source regression tests establish the named behavior. Prior moving tests still showed large controller jumps. A later lit, hanging-headset recording showed severe raw backend divergence despite positive observations and no controller masks. That uncontrolled recording does not identify a single cause or prove a new patch regressed; it establishes that the main failure remains.

## Current artifact reference

These hashes identify the existing reference installation; they are not downloadable binaries or a claim that a fresh build is bit-identical.

| Artifact | SHA-256 |
|---|---|
| Installed stripped Monado driver | `82d80aa3b3ba174b5066011add02450f2f06e0e30a52fab1f4de794300aa6c4a` |
| Tested unstripped driver | `1c875c3ae329006c4cddbf04d986b7ba42b367f758187b7c92c7fbae961362e3` |
| Retained Basalt release library | `de4f9d30ae0417203796d51ffa66cc56e7c79ec06394a0a60387d68e9516b73c` |

Driver build ID: `011dcb45df91b7c8a039d0bea2501c4f99849be3`. Basalt source: `30ece25f4c7d86e6a9dbee7ff0ebd0b921344a67`, VIT headers `e6db0fb84c69614bc4923fde6c52154c221c1768`.

Normal-launcher loading and persisted settings were verified before publication; a fresh full reboot and separate Steam URI launch were not independently revalidated for this snapshot.

## Feature policy

- Enabled: `WMR_SLAM=true`, `WMR_MAX_SLAM_CAMS=2`, `WMR_AUTOEXPOSURE=true`, `SLAM_SUBMIT_FROM_START=true`, `G2_REQUIRE_VISUAL_OBSERVATIONS=true`.
- Disabled: `WMR_HANDTRACKING=false`, `WMR_CLOCK_WINDOWED=false`, `G2_PREDICT_WITH_VIT_BIAS=false`.
- `WMR_HT1_EXTRINSICS_OVERRIDE` empty/unset; factory calibration retained. `SLAM_CONFIG` unset for the reference path.
- Diagnostics off during normal launch. Backend path resolved locally, not copied from a developer's home directory.

## Rejected / unqualified work

The optional gyro-bias consumer is compiled but off. A vendor backend getter was tested as read-only: candidate and matched source rebuild gave 2,553 identical standard output rows. **Both rebuilds diverged to 481.7 m on the same recorded input where the retained release ended at about 9 cm displacement. Neither rebuilt library is qualified for deployment.** This does not blame the getter; it blocks the rebuild until the discrepancy is explained. Do not ship either artifact.

The Windows HT1 extrinsics trial had mixed paired-replay results and remains off. Calibration was mostly identical; copying Windows settings did not establish a drift cure. Manufacturer grip/aim data and the Home-specific transform are distinct from an earlier subjective six-degree ray trial, which is not physically accepted.

## Next priorities

1. Reproduce and explain backend build divergence using controlled inputs/toolchain/dependency provenance.
2. Diagnose raw head VIO drift with honest visual-quality/observability measures and timing; do not hide it with floor resets or presentation filtering.
3. Validate head-motion/controller coupling through frame and timestamp invariants, then targeted motion tests.
4. Resolve jitter/post-stop prediction without excessive lag; qualify learned bias only with a qualified backend.
5. Validate both controller grip/model/aim mappings and provide reversible floor setup.
6. Turn the development tooling into qualified packages and expand the hardware matrix only after evidence.

See the [roadmap](../ROADMAP.md), [tracking FAQ](TRACKING-FAQ.md), and [continuation prompt](../../prompts/CONTINUE.md). Private raw evidence stays outside the repository.
