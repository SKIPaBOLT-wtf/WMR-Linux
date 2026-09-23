# Tracking evidence and provenance

Reviewed on 2026-09-23. This public-safe summary names engineering records without reproducing private logs, calibration blobs, room imagery, spatial maps, identifiers, or local account paths. Original evidence remains in the local development workspace. The current source and tests should be the public reproduction mechanism wherever possible.

## Local evidence reviewed

| Record | What it establishes | What it does not establish |
| --- | --- | --- |
| `tracking-state-consistency` | Combined driver installed and loaded in a fresh normal launch; exact binary/backend hashes recorded; optional bias consumer disabled; factory camera calibration retained. | Physical tracking acceptance, a full reboot test, or complete controller tracking. |
| `windows-calibration-audit` | Read-only comparison against the same headset's saved Windows calibration; camera/IMU agreement except a small front-camera extrinsic difference and equivalent display normalization; manufacturer controller pose definitions corroborated. | All Microsoft filtering or position math, which driver consumed every registry value, or a causal explanation for drift. |
| `offline-vit-replay` | Production calibration parser parity and deterministic same-input comparisons. The isolated Windows camera extrinsic gave mixed results. Mask timing strongly affects the replay. | Exact reconstruction of live queue timing or mask feedback, external motion ground truth, or a general benefit from the override. |
| `hmd-hid-preserve` | Controller firmware reply handling previously consumed head sensor reports. Production-path fixes and regression tests preserve them; live continuity improved to approximately 1 kHz with a maximum interval near 2 ms in the measured run. | Complete VIO stability, every USB failure mode, or physical comfort. |
| `historical-prior-fix` | Earlier camera queries could use an inconsistent current angular rate; retained history, coherent covariance/gravity metadata, and world-generation checks fix deterministic counterexamples. | Elimination of every controller optical ambiguity or timestamp error. |
| `reference-acceptance-fix` | Rejected optical proposals previously changed orientation-reference/filter state; acceptance and retry rollback now preserve the proper committed state. | Complete moving-controller stability. |
| `world-frame-combined` | Matching controller pose and full rigid presentation correction travel together; pose, velocity, and frame invariants pass controlled tests. | Guaranteed real-world alignment or absence of residual HMD error. |
| `controller-mask-safety` | Stale optical geometry and an unreanchored world cache were real defects. Production-source fixtures verify rigid invariance and freshness boundaries. | A latest live test of stale/initialized cache boundaries: the newest recording had no initialized mask history. |
| `head-stop-audit` | Raw estimate changes and presentation correction both contribute to a recorded post-stop event; floor-offset-only explanations do not explain the ceiling event. | A unique cause of micro-jitter, external physical error, or a correction for long-term drift. |
| `home-pose-audit` | Native controller mesh registration supports a fixed Home-specific frame; subsequent runtime records confirm the selected binding and both controller-type overrides. | Physically accepted hand/thumb/ray alignment or sampled Home action matrices. The early README's “staged only” status is superseded by later runtime records. |
| `pose-bias-prediction` | Optional exact-pose gyro-bias consumption, fallback, frame/sign, and history ownership pass bounded tests. | A deployable or physically accepted jitter fix. |
| `vit-pose-bias-extension` | Read-only bias export and unmodified rebuild produce identical standard estimator output to each other on the qualified test input. | Equivalence to the installed release: both rebuilt libraries diverged badly on input that the installed release handled much better. The deployment bundle is rejected. |

## Observations that constrain the diagnosis

Raw backend divergence remains present with positive visual observations and disabled controller masks. Therefore neither the zero-feature policy nor the stale-mask fix solves the whole head-tracking problem. A prior dark capture was not a reproduction of a lit driving scene. These different conditions must not be pooled as if they were one controlled experiment.

The last capture contains partial diagnostic tails; only complete records support its reported metrics. Lack of an external reference prevents interpreting trajectory differences as exact physical error. Report sample coverage, dropped/truncated data, lighting, headset motion, controller visibility, and whether replay was lockstep or live.

An earlier stationary-headset test looked stable while subsequent normal head movement still caused large controller discontinuities. Physical regression reports override the inference that a stable idle trace means success.

## Primary public references

- [Microsoft: How inside-out tracking works](https://learn.microsoft.com/en-us/previous-versions/mixed-reality/enthusiast-guide/tracking-system) — public explanation of visual/IMU fusion and retained environment data; not source code for the proprietary estimator.
- [Microsoft: Motion controllers](https://learn.microsoft.com/en-us/windows/mixed-reality/design/motion-controllers) — grip versus pointer semantics and optical/inertial/degraded tracking states. Its Windows setup steps are not Linux setup instructions.
- [Valve: OpenVR header](https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h) — standing-space/floor convention, raw/standing transforms, chaperone working copy, and explicit origin reset.
- [Valve: Render Model Reference](https://github.com/ValveSoftware/openvr/wiki/Render-Model-Reference) — grip/tip component contracts; raw device pose is not interchangeable with every application's model frame.
- [Valve: Driver API Documentation](https://github.com/ValveSoftware/openvr/blob/master/docs/Driver_API_Documentation.md) — application-scoped controller emulation and resource-only driver contracts.
- [Basalt for Monado](https://gitlab.freedesktop.org/mateosss/basalt) — the backend project. The local release comparison used source commit `30ece25f4c7d86e6a9dbee7ff0ebd0b921344a67` with pinned dependencies; rebuilding that source has not yet reproduced the installed release's behavior.

The implementation-specific camera/IMU statements were checked against `src/xrt/drivers/wmr/wmr_hmd.c`, `wmr_config.c`, `wmr_source.c`, and `src/xrt/auxiliary/tracking/t_tracker_slam.cpp` in the recorded combined source. These paths should be relinked to the exact published commit when the fork is created. The public Microsoft and OpenVR pages were checked during this audit; no claim is made that they reveal undocumented production internals.
