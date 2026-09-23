# Tracking status and practical questions

Reviewed on 2026-09-23. This is an experimental native Linux WMR tracking stack. Display output and game launch have been demonstrated on the reference machine. Head and controller tracking have **not** passed physical acceptance. A passing unit test, a valid-pose flag, or a stable unattended capture does not establish comfortable tracking during normal head movement.

## Would SteamVR Room Setup correct these inaccuracies?

Room Setup is useful for the standing origin, facing direction, floor height, and play-area bounds. It does not calibrate the camera/IMU fusion, correct a bad camera timestamp, or remove ongoing estimator drift. The G2 uses inside-out tracking; the wizard's generic base-station illustration does not mean this headset needs base stations. [Microsoft's controller tracking description](https://learn.microsoft.com/en-us/windows/mixed-reality/design/motion-controllers#controller-tracking-state)

The observed ceiling-height incident already contained a large displacement before SteamVR's standing-space transform. Restarting the tracking session restored the expected height; this demonstrated recovery, not drift prevention. Repeatedly adjusting the floor would hide the symptom temporarily. There is no need to repeat the wizard as a tracking experiment.

## Do we know why the Windows installation avoids this drift?

**Not yet.** Public Microsoft documentation describes visible-light feature tracking fused with high-rate IMU data and retained environment information. It does not disclose the exact production estimator, all filter parameters, controller arbitration rules, or the complete relocalization implementation. A software fork must not describe that documentation as a full reconstruction of Microsoft's tracking system. [Microsoft's inside-out tracking explanation](https://learn.microsoft.com/en-us/previous-versions/mixed-reality/enthusiast-guide/tracking-system)

A read-only comparison of the reference headset's Windows and device-supplied calibration found matching camera intrinsics and IMU calibration. One front-camera extrinsic differed slightly; display values also used a different but mathematically equivalent normalization. Testing only that extrinsic on identical input gave mixed results, so the optional override remains disabled. The comparison did not identify the cause of Linux drift. The private calibration, registry hives, room database, and logs are not publication assets.

## Have all confirmed Windows filtering and position mechanisms been accounted for?

No completeness claim is justified. The useful confirmed public contract includes distinct grip and pointer poses, brief inertial continuation after optical loss, and degraded controller position states. Prolonged optical loss can produce a body-relative approximate position while orientation continues from the controller's sensors; this is different from reliable world-locked tracking. Those semantics are a comparison checklist, not evidence that this Linux implementation matches Microsoft's behavior. [Microsoft's pose and tracking-state documentation](https://learn.microsoft.com/en-us/windows/mixed-reality/design/motion-controllers)

Work on the Linux path has addressed concrete temporal-prior, optical-acceptance, idle-packet, coordinate-frame, sensor-delivery, and pose-history defects. Remaining work includes quantified tracking confidence, loss/recovery behavior, prediction consistency, and moving-head validation. Unknown proprietary details must stay explicitly unknown.

## Why can turning the headset move controllers that are physically still?

Headset cameras measure a controller relative to a moving camera. Its world pose therefore depends on both that relative measurement and the headset pose **at the image exposure time**. Head-pose error, camera/IMU timing error, a stale estimator prior, or inconsistent world-frame changes can move both hands together even if neither controller moves physically.

Several relevant defects have been reproduced and corrected in bounded tests:

- Historical camera queries used the latest angular rate to reconstruct earlier controller motion. Retained capture-time estimator state now supplies a coherent prior.
- Head presentation corrections and controller poses could come from unrelated snapshots. The rigid correction now travels with its matching pose.
- Cached controller optical poses were not transformed when the estimator changed world frame. The cache is now reanchored with it.
- Old controller geometry could keep masking head-tracking images long after trustworthy optical evidence expired. Masks now use the existing freshness rule.
- Rejected optical estimates could still alter the gyro reference or leave speculative filter changes behind. Acceptance is now transactional.

These are demonstrated source defects, not a complete explanation of every physical jump. A recorded moving-head trial still contained large controller discontinuities after earlier fixes. The newest unattended capture did not exercise the initialized controller-cache path. The question remains open at system level.

## Has micro-jitter been removed?

No. Some mechanisms are now isolated, but the reported small involuntary-looking motion and fast-stop settling have not passed a wearer test. Raw fused pose, local prediction, presentation correction, and rendered timing must be measured separately.

The backend learns gyro bias, while the ordinary pose interface does not expose that bias to the short-horizon predictor. An optional pose-matched bias interface and consumer have been tested. However, both a source-rebuilt backend control and its extended counterpart failed identical-input qualification against the installed release. The new backend is **rejected for deployment**, and the consumer remains disabled. It is not a jitter fix available to users yet.

Do not add a dead zone or stronger smoothing merely to make a stationary trace look good. Such changes can suppress legitimate small motion and add onset/stop lag. Any filter needs a stated purpose, noise model, latency budget, and controlled motion tests. A wearer cannot and need not be perfectly motionless.

## Can I set the floor level?

Yes. SteamVR Room Setup is the existing user-facing calibration route. Once tracking is stable, use its height calibration for the chosen play-area mode. That changes the coordinate reference, not sensor calibration. The current custom launcher has no dedicated floor command.

OpenVR also exposes a standing-origin transform through `IVRChaperoneSetup`; a future utility can back up, preview, apply, and restore a floor adjustment. This API was checked read-only; no floor setting was changed. OpenVR defines standing-space Y=0 as floor level and separates the standing transform from raw device coordinates. [Valve's OpenVR API](https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h)

A cockpit's recenter action is useful for the seated viewpoint but is not a substitute for stable tracking or a room-floor definition. A floor tool must preserve rotation, horizontal origin, and existing bounds unless the user explicitly changes them. It must not quietly follow drifting head height.

## Are cameras and accelerometers fighting each other?

They are intended to constrain one estimate together. In this stack, Basalt combines camera observations with head IMU measurements; a separate controller estimator combines controller IMU and optical LED observations. It is not correct to say accelerometers are unused.

There are several opportunities for inconsistency: different clock domains, image exposure versus delivery time, mismatched static calibration, stale world-frame generations, asynchronous optical updates, or applying a bias twice. The already corrected HID routing defect discarded head IMU packets during controller initialization; a later live sample demonstrated uninterrupted approximately 1 kHz measurement delivery through that path. That removes one transport defect, not all fusion uncertainty.

Controller acceleration is valuable evidence, but cannot be an unconditional approval gate for every camera position update. An accelerometer measures specific force rather than position; constant-velocity motion does not require persistent acceleration, and a visual correction can legitimately repair accumulated inertial error. The correct comparison uses synchronized state, gravity, uncertainty, and measurement innovations.

## Are camera inaccuracies compensated correctly?

Some calibration and rejection paths are checked, but complete correctness is not established. The driver uses device-supplied intrinsics/extrinsics. The reference profile currently uses two cameras for head SLAM, while the controller path can consume the headset's full configured tracking-camera group. Enabling additional cameras is not automatically an improvement: image order, extrinsics, exposure timing, and backend behavior must be qualified together.

Known gaps include WMR-to-backend camera-model approximations, unproven effective exposure timing, dynamic people in the scene, stale-mask consumer behavior if the producer stalls, and the meaning of positive landmark counts. The installed policy invalidates head tracking when a successful current-frame observation query reports zero observations. Positive counts still do not guarantee a correct solution: substantial raw backend divergence has occurred with positive observations and controller masks disabled.

Moving people must not become the room's apparent fixed reference. The next work must test robust static-scene support and loss/relocalization behavior rather than assume that more features, stronger smoothing, or a room wizard can solve this.

## How to read the evidence

“Source defect reproduced” means a deterministic counterexample fails in the earlier code and passes in the changed production path. “Runtime route verified” means the intended library, setting, binding, or data path was actually used. “Physical acceptance” requires the real behavior during normal use. The first two do not imply the third.

See [the engineering checklist](TRACKING-ARCHITECTURE.md) for the remaining work and [the evidence map](TRACKING-EVIDENCE.md) for provenance and limitations.
