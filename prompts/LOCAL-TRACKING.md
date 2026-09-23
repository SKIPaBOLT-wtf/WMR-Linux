# Local G2 tracking continuation prompt

Copy everything below into an agent with access to the Linux development machine. This prompt is standalone and remains applicable at later project versions.

---

Continue the local HP Reverb G2 tracking investigation in English. Improve head/controller tracking while preserving working display output, DiRT Rally 2.0 VR, input, and the retained backend. Make evidence-backed changes, complete available validation, and leave a precise handoff. This task concerns the local installation; broader packaging and public project development are supplementary, not a reason to postpone the tracking diagnosis.

## Recover the actual state

Read applicable AGENTS.md files, the integration repository README, docs/current/STATUS.md, tracking FAQ/architecture/evidence, recent iteration records, provenance, release gates, and selected profile. Locate the companion runtime checkout and verify its commit, branch, uncommitted changes, and pairing manifest. Use current records; never reinstall a historical candidate because an old report calls it “latest.”

Locate PRIVATE-HANDOFF.md in the current workspace/repository surroundings or follow WMR_PRIVATE_EVIDENCE if set. Keep its paths and contents private. Use that handoff to locate calibration, recordings, source snapshots, binary manifests, rejected experiments, and rollback records. Avoid broad searches through unrelated personal files. If evidence is unavailable, state the limitation and continue useful independent work.

Read-only checks must establish current hardware/software, runtime processes, installed and actually loaded library hashes, active options, calibration source, and normal launcher route. Distinguish installed, staged, rejected, and obsolete artifacts. Check whether the user is playing or has protected a session. Continue previously authorized work without redundant permission requests, while honoring the latest constraints. A hanging headset and invisible controllers permit some diagnostics, not a physical acceptance test.

## Preserve the baseline

The historical reference used G2, RTX 4080 SUPER, Ubuntu/GNOME Wayland, native 4320×2160 at 90 Hz, a Monado SteamVR driver, and pinned Basalt. Display output and driving in DiRT were demonstrated; a version-specific NVIDIA workaround resolved reported tearing. Verify current values rather than copying this historical configuration blindly. Preserve display/DRM handling, NVIDIA patches, refresh rate, SteamVR presentation options, Home resources, Xbox input, and the successful game-launch path during tracking work.

This headset needs no base stations. Its driver is native Linux; the Windows game may use Proton. Do not mix graphics/AA/monitor changes with sensor-fusion experiments.

Never call GetMirrorTextureGL or acquire GL mirror textures on this reference stack: those probes triggered compositor/Xwayland failures. Use CPU OpenVR APIs and offline tools. Keep diagnostics bounded and disabled during normal launch.

Historically, both the source-rebuilt Basalt control and bias-export candidate failed identical-input qualification against the installed release. The gyro-bias consumer was compiled but disabled; the Windows HT1 extrinsics override was also disabled. Treat those states as rejected/off until newer evidence explicitly qualifies a replacement. Compilation and ABI compatibility alone do not qualify a backend.

## Maintain answers to all eight questions

1. Can Room Setup improve the reported error, or is it already present before standing-space calibration?
2. What evidence explains Windows' better behavior, and what remains unexplained?
3. Which Windows filtering/position mechanisms are publicly documented and actually compared, versus proprietary or assumed?
4. Why do stationary controllers move when the head moves?
5. Which layer creates micro-jitter, exaggerated small movements, and settling after fast stops?
6. How can floor height be set deliberately, reversibly, and persistently?
7. Are camera and IMU observations fused consistently in time, axes, calibration, and uncertainty?
8. How are camera-model inaccuracies, occlusion, dynamic people, and bad optical hypotheses handled?

Do not claim all Microsoft internals are known. The Windows audit compared calibration, resources, and driver metadata; it did not reconstruct the production estimator. Keep authorized Windows volumes read-only and verify mount state before reading. Never import a whole calibration/registry/spatial database or execute Windows binaries as an assumed fix. Read current primary documentation where necessary.

## Investigation sequence

First verify transport and timing: full HID packet dispatch during controller initialization, valid versus idle/status samples, measurement continuity, clock conversion, rollover, image order, exposure timestamp, queue/submission time, out-of-order delivery, and reset/wake behavior. Preserve proven fixes for discarded head IMUs, history boundaries, concurrency/lifecycle ownership, and idle sentinels.

Write the coordinate graph and timestamp contract. Include each camera, head IMU/body, controller IMU/LED/grip/aim, raw estimator world, presentation world, and SteamVR raw/standing/seated spaces. Define transform direction, quaternion order, handedness, and units. “Raw” OpenVR space remains downstream of driver presentation; it is not direct Basalt output.

For coupling, use a consistent capture-time chain:

```text
T_world_controller(t) = T_world_head(t) * T_head_camera * T_camera_controller(t)
```

Test common rigid-world-transform invariance, covariance/velocity transformation, historical priors, and independent head/controller motion. Reanchor caches and estimators once without refreshing observation ages. Keep coherent state through delayed optical replay and world generations. Camera-relative movement caused by head motion does not require controller acceleration; accelerometers cannot measure absolute position or constant velocity. Use synchronized innovations and uncertainty instead of vetoing every optical correction without acceleration.

For head drift, inspect backend pose before prediction, presentation guards, and floor offsets. Check visual distribution/inliers/residuals, feature age, parallax, static-scene support, biases, estimator health, loss/recovery, and actual relocalization/map capabilities. Positive feature counts are not proof of accuracy. Divergence has occurred with positive counts and disabled controller masks; neither zero-feature invalidation nor mask fixes solve everything.

For cameras, audit per-unit intrinsics/extrinsics, WMR distortion conversion, usable radius, camera ordering, shutter/exposure assumptions, and calibration-to-image pairing. WMR_MAX_SLAM_CAMS selects head SLAM cameras; controller tracking uses its separate camera group. Do not enable every camera without qualification or distribute one unit's extrinsics to others.

For controllers, preserve the corrected IEKF residual sign, capture-time history, accepted-optical-only references, rollback of rejected trials, hardware-idle handling, paired presentation transforms, and cache/mask freshness. Test ambiguous LED association, multi-camera consistency, occlusion and reacquisition. The current mask consumer lacks a timestamp, so a producer stall remains an explicit freshness-contract gap.

For jitter, compare calibrated IMU, fused backend pose, local prediction, and submitted pose at a common time. Match learned bias to the exact accepted pose snapshot and body frame; subtract once after static calibration and before world rotation. Resolve acceleration-calibration differences before consuming learned acceleration bias. Do not conceal bad timing with stronger smoothing. Measure noise, lag, overshoot, prediction age, fast-stop settling, and preservation of real small movements.

Keep model/aim alignment separate: validate both hands, grip versus pointer versus Home-native poses, selected bindings/types, actual action matrices, and manufacturer transforms. A subjective angle trial is not calibration evidence; never rotate sensors to correct a rendered ray.

Room Setup establishes floor/origin/bounds; it cannot cure changing VIO drift. A floor utility must back up and preview the standing transform, use a supported API, preserve unrelated bounds/orientation, verify persistence, and provide undo. Seated game recentering is a separate operation. Do not make automatic floor resets a tracking workaround.

## Validate, install, and hand off

Choose one falsifiable hypothesis per iteration. Record baseline commits/hashes/options, complete input provenance, predicted outcome, pass/fail criteria, and rollback. Exercise production code with deterministic negative controls and relevant existing regressions; use sanitizers for lifetime/concurrency changes. Compare identical replay input/calibration/order and explain differences from live queues or mask feedback. Exclude partial tails; do not claim continuity from incomplete captures.

Physical tests must answer a specific remaining decision: natural small head movement, slow sweeps, fast turns/stops, resting controllers viewed by a moving head, independent hand motion, occlusion/reacquisition, and people moving in the room. Never demand a perfectly still wearer. Hanging/idle results do not prove comfort. If the user is unavailable, complete offline work and state the exact physical boundary. User-reported regression overrides favorable uncontrolled metrics.

Install only within current authorization after relevant gates pass. Stop/refuse an active runtime, privately back up exact changed files, use atomic replacement, verify dependency/ABI and loaded hashes, then check a fresh normal launch with diagnostics off. Preserve the retained backend. Rollback must detect later user edits instead of replacing whole Steam configurations. Distinguish fresh-process, Steam-restart, and reboot validation.

Keep private recordings, calibration, registry data, room maps, identifiers and their hashes, account paths, screenshots, and raw logs outside Git. Publish only reviewed code, synthetic fixtures, and sanitized findings if publication is authorized. Update current status and one iteration record, including failures. Report what changed, what was tested/installed, unresolved behavior, current runtime state, rollback, and the next evidence-backed action. Do not declare tracking fixed without the promised physical acceptance.

Start by reconstructing current state and selecting the highest-priority diagnosis that available evidence can advance. The separate prompts/CONTINUE.md covers broader project/package development if needed; this local task does not depend on copying that prompt too.
