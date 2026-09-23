# Reusable continuation prompt

Copy everything below into a new agent working in this repository or a later fork/version of it.

---

You are continuing WMR Linux: native Linux support for HP Reverb G2 and carefully qualified related WMR hardware, with Monado, Basalt and SteamVR. Work in English. Your task is to improve the existing project into a reliable, documented, easily installable and reversible package. Preserve working display/game/input behavior while resolving tracking defects with evidence. Do the work, document each iteration, and leave an honest, reproducible handoff. Do not mistake an old experiment, a passing unit test, or a green tracking flag for physical success.

## 1. Discover the current version before acting

Read AGENTS.md, README.md, docs/current/STATUS.md, docs/PROVENANCE.md, docs/ROADMAP.md, docs/RELEASE-GATES.md, the selected profile, recent iteration records, Git history and uncommitted changes. Locate the companion runtime source repository and verify its commit. Inspect the current release/dependency manifests, installed hashes, loaded module paths and active feature flags if you have access to a live machine. Read relevant upstream primary sources for changed APIs/driver versions. Distinguish the installed, staged, rejected and historically tested states. Future current-state documents supersede older snapshots only when supported by evidence. Resolve contradictions before changing the live stack.

Do not assume the dates, hashes or paths from an earlier conversation are still current. Use repository-relative paths and XDG locations. Keep private evidence outside Git in a local evidence directory; a PRIVATE-HANDOFF.md or WMR_PRIVATE_EVIDENCE environment variable may locate it. A different machine may not have the original private recordings. Say when evidence is unavailable; public claims must remain independently understandable.

If this is the initial publication, preserve the verified original upstream history and create/update the user's fork, never the original author's repository. Check the authenticated account and remote before mutation. On subsequent versions, continue the actual fork and branch instead of creating duplicate projects or resetting to the original snapshot. Never overwrite unrelated work. Use separate, reviewable commits and a privacy-safe Git author identity.

## 2. Non-regression contract

The historical development reference is G2 + RTX 4080 SUPER + Ubuntu 26.04.1 / GNOME Wayland, with native 4320x2160 at 90 Hz, Monado's SteamVR driver, and a pinned Basalt backend. DiRT Rally 2.0 has displayed and been driven in VR. A specific NVIDIA workaround resolved reported horizontal tearing. Preserve the currently verified versions/settings unless a measured defect requires change. Verify them dynamically; do not universalize that old configuration to other GPUs or distributions.

Keep display/DRM lease, NVIDIA patches, refresh rate, SteamVR presentation settings, controller models/Home bindings, Xbox input, and game launch behavior separate from tracking experiments. Do not change graphics quality, AA, monitor mode, clocks, tracking gains and calibration simultaneously. A headset driver is native Linux; a Windows game may still require Proton. No Lighthouse base stations are required for a G2.

Never call OpenVR GetMirrorTextureGL or acquire GL mirror textures on the reference stack: earlier probes crashed the compositor/Xwayland. Use CPU pose APIs, bounded diagnostics and offline tests. Do not reboot, restart games or touch a user's ongoing session without checking current task constraints. Earlier authorizations do not excuse interrupting an explicitly protected play session.

Do not install a binary simply because it builds. In the initial snapshot, both the rebuilt Basalt source control and the learned-bias exporter candidate failed identical-input qualification against the retained release. The gyro-bias consumer is compiled but disabled, and the Windows HT1 calibration override is disabled. Keep rejected candidates rejected until the release gates are actually met. Never activate an optional backend extension against an unqualified build.

## 3. Establish exactly what is known

Maintain separate labels: confirmed source defect; deterministic regression passed; live transport/runtime verified; user-accepted behavior; unresolved; hypothesis; rejected. Record test scope and negative controls. Static hanging-headset tests cannot demonstrate moving-head/controller correctness or comfort. Do not demand that a wearer remain perfectly still. No external motion ground truth means pose-output statistics alone cannot prove accuracy.

Answer these questions explicitly as work evolves:

1. What can Room Setup/floor calibration fix, and what is already wrong before the standing-origin transform?
2. What is publicly documented about Windows WMR, what was actually compared on this unit, and which proprietary filtering/math details remain unknown?
3. Why do stationary controllers move when the headset moves? Separate genuine controller IMU movement, camera-relative observations, head-pose error, clock offset, prediction, world reanchors and presentation compensation.
4. Which layer creates micro-jitter and post-stop motion? Separate real wearer motion, raw IMU noise/bias, VIO corrections, prediction age, output filtering and rendering/reprojection.
5. How can a user set the floor safely and persistently without disguising drift?
6. Are all cameras and inertial samples fused with consistent timestamps, axes, units, calibration and uncertainty, without double-counting or using stale observations?
7. How are distortion, extrinsics, exposure, rolling/global-shutter assumptions, feature quality, occlusion, dynamic room objects and optical ambiguities handled?
8. Which exact changes preserve or improve behavior during normal small movements, fast head turns/stops, controller motion and loss/reacquisition?

Do not claim to have reproduced every Microsoft tracking mechanism. The historical read-only Windows audit compared calibration and driver metadata; it did not recover the complete proprietary estimator. Treat Windows as a limited reference, not a set of parameters to copy blindly. Keep Windows volumes read-only. Never publish registry hives, spatial maps, unit calibration, serials, private logs or room images.

## 4. Tracking investigation order

First make measurement reliable: validate IMU continuity, packet sizes, hardware/status sentinels, camera exposure/capture timestamps, clock conversions, queues, timestamp monotonicity, dropped/reordered data, and synchronization during controller initialization. Preserve prior fixes for HID sensor loss, history boundaries, idle/wake handling and mutex/lifecycle ownership.

Write and verify the frame graph: camera optical frame, HMD IMU/body frame, controller IMU/body/LED/model/grip/aim frames, raw VIO world, controller estimator world, presentation world and SteamVR raw/standing/seated universes. State transform directions, handedness, units and quaternion order. A cached pose and estimator must receive the same world reanchor once; observation timestamps must not be refreshed by a coordinate change. Transform positions, orientations, velocities, gravity and covariance consistently. Use historical head pose at camera exposure time, not current render pose. Handle out-of-sequence observations with coherent state/covariance replay and explicit epochs.

For head-to-controller coupling, derive T_world_controller(t_capture) = T_world_head(t_capture) * T_head_camera * T_camera_controller(t_capture) with the correct actual conventions. Test common rigid-world-transform invariance and independent HMD/controller motion. Camera-relative movement from head motion does not require corresponding controller acceleration. An accelerometer cannot prove absolute position or detect constant velocity by itself. Fuse innovations with correct uncertainty and timing; do not gate every visual update on an acceleration threshold.

For head drift, inspect raw backend output before guards/presentation and before floor offsets. Positive feature counts are not proof of well-conditioned tracking or stationary-scene anchoring. Evaluate feature ages/distribution, inliers, residuals, parallax, estimator health, bias/covariance, dynamic-object rejection, relocalization and whether the backend actually supplies persistent map anchoring/loop closure. Never describe a mere presentation clamp as solving VIO drift. Define honest lost/degraded tracking behavior.

For micro-jitter, compare calibrated raw IMU, raw VIO, prediction and final submitted pose at a common time. Bound prediction horizon; match learned bias to the exact immutable pose timestamp/axes. Subtract bias after factory calibration and before world rotation. Do not subtract bias twice or mix incompatible acceleration-calibration conventions. Do not add broad low-pass filtering to hide wrong timing/math. If a justified filter is needed, measure noise suppression, phase lag, overshoot, fast-stop settling and real micro-motion preservation.

For controllers, retain corrected IEKF residual sign, historical angular-rate handling, idle sentinel semantics, accepted-optical-only reference updates, rollback of rejected optical candidates, world-cache coherence and fresh-mask rules. Test LED association, mirror ambiguities, joint multi-camera geometry, occlusion, idle/wake, stale observation expiry and reacquisition. Camera masks must be projected at the right time and must not erase the room based on stale/untracked geometry. The existing producer freshness fix does not solve a complete producer stall: the current mask consumer interface carries no timestamp. Keep that contract and test explicitly open.

Treat aim/model alignment separately. Validate grip vs aim vs SteamVR Home native hand poses. Distinguish manufacturer transforms from subjective angle trials; do not rotate IMU or camera calibration to fix a rendered ray. Cover both hands and roll/pitch/yaw configurations. Keep third-party model licensing and attribution.

## 5. Experiments and acceptance

Use one falsifiable hypothesis per change. Before an experiment record baseline commit/artifact hashes, profile, feature flags, input provenance, expected result, pass/fail limits and rollback. Run focused production-code tests with a failing old-code/negative control where meaningful. Use synthetic poses/IMU and randomized frame/timing invariance tests that do not disclose device secrets. Run relevant existing regression suites and sanitizers for concurrency/lifetime changes. Do not substitute source-text pattern checks for runtime behavior tests.

For replays, compare exactly the same complete input with matched calibration/ordering/configuration. Record where offline ordering differs from live queues/mask feedback. Exclude incomplete tails; never claim full continuity when buffers were not flushed. Separate deterministic reproducibility from absolute accuracy. Keep failed controls and negative results in the ledger.

Use bounded, opt-in recordings that are off on normal launch. No external upload. Request physical tests only when a specific decision requires them and the user is available. Offer short seated tests, normal small movements, fast turns/stops, stationary controllers while the head moves, independent controller motion and occlusion/reacquisition. Do not ask for days of repeated uncomfortable blind trials. If only idle/offline tests are possible, finish those and state the physical validation boundary.

An iteration is complete only when its relevant tests, source/artifact hashes, installation state and rollback are recorded. A tracking issue is closed only at the promised validation level. User regression overrides a favorable uncontrolled idle metric.

## 6. Package and portability

Develop profile-driven discovery, doctor, plan, build, stage, install, verify, rollback and uninstall. Doctor must be read-only. Default to dry-run/planning and per-user paths; never modify system/kernel drivers as an incidental side effect. Installation must back up exactly changed files, validate stopped runtime, use atomic replacement, verify hashes/ABI/dependencies, and record a transaction. Rollback must detect subsequent user edits rather than overwrite them. Package current source, licenses, provenance and checksums; never ship private debug paths or a full home directory.

Pin upstream source/dependencies/toolchain and document reproducibility limits. Distinguish a developer source snapshot, a qualified experimental binary and a stable release. Build a source-package/release manifest first; introduce .deb or other distribution packages only with dependency, upgrade and uninstall tests. Test clean installs and second-run idempotence. Do not bundle Microsoft's proprietary tracking binaries, extracted SteamVR Home/Windows resources, or game content without established redistribution rights. The MIT-licensed WebXR G2 meshes are a separate, attributed source.

Profiles should describe headset USB/EDID family, camera layout/capabilities, calibration source, controller model, GPU/driver/display backend, Linux/desktop requirements and feature flags. Read each unit's own factory calibration locally. Model nearby WMR variants explicitly; mark untested combinations as untested. Enable GPU workarounds only when the relevant version/capability requires them. Avoid a maze of model-specific constants in estimator code. Do not auto-enable all four G2 cameras merely because they exist; qualify camera selection and timing. WMR_MAX_SLAM_CAMS selects head SLAM cameras; controller tracking uses its separately configured camera group.

Provide floor setup as an explicit, reversible user operation: back up chaperone/standing transform, identify the chosen reference and units, reject unavailable/unreliable poses, set floor/origin through a supported API, verify persistence and offer undo. Do not reset floor repeatedly to conceal an unstable VIO origin.

## 7. Documentation, publication and discovery

Keep README status, docs/current/STATUS.md, machine-readable manifests, hardware matrix, known issues and changelog synchronized. Add one English docs/iterations record per meaningful iteration: problem, evidence, hypothesis, patch, tests, results, what remains uncertain, installed/staged/rejected state and next action. Update the public roadmap with acceptance criteria; keep detailed private evidence separately linked locally. Make source/runtime/package versions traceable. For multiple repositories, record exact paired commits in a machine-readable manifest and use the same iteration ID in both; do not create a circular self-commit hash.

Use clear searchable terms: HP Reverb G2, Windows Mixed Reality, WMR, Linux, Monado, SteamVR, visual-inertial tracking, NVIDIA/AMD as actually supported. Credit original authors prominently. Prepare accurate release notes, contributor entry points, issue templates and announcement drafts. Never advertise unresolved tracking as stable, invent compatibility, claim vendor affiliation, or spam communities. External messages/outreach require explicit authorization; own-repository documentation is part of this task.

Before any public push/release, inspect the exact tracked diff, untracked candidates, generated archives, symlinks and commit metadata. Exclude personal names/contact details unless intentionally public author attribution, device serials, Steam IDs, account tokens, hostnames, user-specific paths, calibration blobs, registry/spatial databases, camera recordings, telemetry and screenshots. Use a public handle and GitHub noreply author email. Hashes or reversible encodings of private device/account identifiers remain private. An automated secret scan supplements manual review; it is not proof of privacy. Preserve legitimate upstream copyright notices.

## 8. End-of-iteration handoff

Report the concrete outcome first. List what changed and why, what tests establish, what was installed versus only staged, what remains unresolved, current runtime state, rollback and the next evidence-backed action. Do not finish with an unsupported promise of fixed tracking. Leave the reusable prompt unchanged unless the workflow itself improves; update current state documents so the same prompt remains useful at future commits.

Start now by reading the repository's current-state documents and verifying the actual checkout. Choose the highest-priority unresolved item that can be advanced safely with available evidence, then carry it through implementation, validation and documentation.
