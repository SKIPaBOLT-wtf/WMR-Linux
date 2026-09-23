# Tracking engineering and acceptance checklist

This document describes the next engineering work, not completed features. Update it as evidence changes. Keep display recovery, the qualified backend binary, the successful game-launch path, and the NVIDIA-specific tearing workaround independent of tracking experiments.

## Coordinate and timing contract

Use `T_A_B` to mean a rigid transform that converts coordinates from frame B into frame A. Name every frame: head IMU/body, each physical camera, controller IMU, controller LED model, controller grip, controller aim, raw estimator world, presentation world, and SteamVR standing space. State units, axis handedness, quaternion order, and active/passive convention beside each interface.

At camera exposure time `t`, the optical chain is conceptually:

```text
T_world_controller(t) = T_world_head(t) * T_head_camera * T_camera_controller(t)
```

The actual implementation may use inverse/basis-converted forms, but all inputs must describe the same time and world generation. A common world transform `D` applied to both camera and controller must leave their camera-relative geometry invariant:

```text
inverse(D * T_world_camera) * (D * T_world_controller)
    == inverse(T_world_camera) * T_world_controller
```

Test full rotations and translations, velocities, and covariance cross terms. Translation-only cancellation tests are insufficient. Do not compare generation counters belonging to independent devices as if they were one shared epoch.

Record hardware measurement time, exposure time, host arrival, optical submission, estimator output, prediction base, prediction target, and presentation time separately. Preserve source timestamps through HID routing and camera queues. Verify clock conversion, packet rollover, monotonicity, reordering, reset, and wake transitions. Do not tune a timing offset against an uncontrolled idle recording.

## Separate the pipeline into observable layers

1. Device calibration and transport: sensor packets, image order, clocks, valid sample flags, and dropped measurements.
2. Raw head VIO: fused pose/velocity, current visual observations, residuals, bias estimates, map/reset epoch, and visual support.
3. Local prediction: exact accepted base pose, calibration/bias conventions, covered IMU interval, integration horizon, and fallback.
4. Presentation: world correction, decay, reanchor, device-paired transforms, and SteamVR standing offset.
5. Controllers: capture-time prior, optical association, PnP/LED acceptance, out-of-sequence replay, IMU arbitration, masks, and loss/recovery.
6. Application: selected grip/aim/native-model component, bindings, action matrices, frame pacing, and reprojection.

The OpenVR “raw” universe is still downstream of this driver's presentation stage. It is not direct Basalt output. A valid pose flag is not a confidence score. Feature count is not geometric observability or proof of a correct map.

## Prioritized unresolved work

| Priority | Question | Required evidence |
| --- | --- | --- |
| P0 | Why can lit head VIO diverge before local prediction? | Identical-input comparisons, confirmed image/IMU ordering and calibration, observation geometry/residuals, reset/loss state, and independently qualified backend builds. |
| P0 | Why does moving the head still produce controller jumps? | Simultaneous head raw/presented and controller raw/presented/camera-relative traces; synchronized innovations and accepted/rejected optical updates; stationary controllers viewed by a naturally moving head. |
| P1 | What produces small head jitter and post-stop settling? | Layer-by-layer angular/position residuals and latency; prediction base age; pose-matched learned gyro bias only after backend qualification; controlled slow/fast motion and stops. |
| P1 | Why are hand models and pointing rays high/outward? | Actual application action matrices and chosen model/type, rigid model registration, manufacturer grip/aim definitions, and physical alignment. Keep static offsets separate from dynamic tracking. |
| P1 | What happens with walkers, occlusion, weak features, and exposure changes? | Static-background support, outlier rejection, truthful degraded flags, bounded loss behavior, reacquisition/relocalization without false world jumps. |
| P2 | How should floor and seated origin be configured? | A reversible user action with backup, preview, stable tracking checks, transform invariants, and restart validation. |
| P2 | Can nearby hardware profiles work without code edits? | Capability detection plus a tested matrix; per-unit calibration remains local. No assumption that all WMR camera/LED layouts or GPU workarounds are interchangeable. |

## Camera model and sensor calibration review

Audit the production parser, WMR distortion equation, backend RT8 conversion, distortion-center terms, tangential terms, metric radius, pixel normalization, and image ROI. Quantify residuals across the usable image, particularly at the edge. The source's comment that some terms tend to be small is not a measured error bound for every device.

Verify camera-to-IMU transform direction and the four-camera image/extrinsic ordering. Preserve hardware model gates; never apply one unit's Windows-updated extrinsics to another unit. Revalidate any additional-camera profile end to end.

Keep static calibration distinct from learned residual bias. Current local head prediction and backend acceleration calibration do not have identical matrix conventions. Do not enable learned acceleration-bias subtraction until these conventions agree. A bias belongs to the immutable pose snapshot from which prediction starts; rejected or reordered poses must not replace it. Subtract it once in the correct body frame, before world rotation.

## Optical/inertial arbitration

Use uncertainty and time-consistent innovations, not a rule that requires acceleration for every optical displacement. Account for gravity and the fact that a camera correction may remove integrated drift. Rejected optical hypotheses must leave the nominal state, covariance, reference orientation, cache, and replay history unchanged except for explicitly documented diagnostics. Partial acceptance must survive a failed speculative retry.

Keep hardware idle status distinct from physical zero measurements. Wake at the first real measurement timestamp; never integrate a long sleep interval using one fresh sample. Preserve matching head/controller presentation state without feeding presentation smoothing back into the raw estimator.

The controller mask fix currently expires producer-side geometry using the existing 150 ms evidence rule and reanchors the cache. The consumer interface has no timestamp, so a total producer stall still needs a separately designed stale-data contract. Test it explicitly rather than assuming producer freshness solves that case.

## Micro-jitter and filtering acceptance

Measure position/angular noise, frequency content, latency, overshoot, slow-motion onset, rapid stop settling, and long-term drift independently. Retain actual sample intervals and confidence/loss annotations. A lower stationary RMS achieved by delayed motion is not automatically a better result.

Use natural small head movement, slow sweeps, fast turns and stops, both controllers resting on a support, both held, one occluded, reacquisition, and an independently moving person. A wearer must never be required to remain perfectly still. Hanging-headset data are transport/backend evidence with their own motion and scene limitations, not a substitute for these tests.

## Floor utility contract

The existing launcher has no dedicated floor command. A future utility can use CPU-only OpenVR APIs:

1. Check that the runtime and current tracking space are available; do not silently start a game or reset its map.
2. Export the current chaperone configuration and record a local backup.
3. Read a valid standing-to-raw transform, then preview a user-requested vertical adjustment while preserving rotation and horizontal placement.
4. Commit only the intended working change; reread and verify it. Provide an exact rollback and avoid overwriting later unrelated settings.
5. Keep standing floor calibration and a game's seated recenter distinct. Reject automatic “drift correction” by continuously moving the floor.

`ResetZeroPose` changes the origin and is defined by Valve as a user-triggered action; it is not a floor estimator. `IVRChaperoneSetup` provides a working copy and explicit commit. See [Valve's API contract](https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h). No floor mutation was performed for this audit.

## Release gates

- Production-path regression tests must include negative controls that fail in the earlier implementation.
- Identical-input replay must compare the exact shipped backend and any source rebuild. A compiling library with a matching version string is insufficient.
- Keep an immutable known-working display/game baseline and an atomic, hash-checked rollback. Change one causal subsystem per experiment where possible.
- Verify the actually loaded binary, environment, bindings, and calibration; configuration text alone is insufficient.
- Distinguish a fresh-process test, Steam restart, and OS reboot. Report which happened.
- Require explicit physical acceptance for moving-head comfort and controller alignment before describing those features as fixed.
- Do not use `GetMirrorTextureGL` or acquire GL mirror textures on the reference system: that diagnostic path has triggered compositor/Xwayland failures. CPU OpenVR queries suffice for these checks.
- Never ship raw room images, spatial maps, per-unit calibration, device identifiers, user configuration backups, or private logs. Publish synthetic fixtures, code, generalized profiles, and reviewed aggregate findings instead.
