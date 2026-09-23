# Profiles

The G2/NVIDIA/Ubuntu profile describes a development reference, not a certified compatibility claim or an installation recipe. It contains no unit calibration or serials. `wmrctl profile` validates its structure without applying it.

Nearby devices need explicit capability checks: USB protocol, firmware, camera count/layout and exposure timestamps, optics/distortion model, factory-calibration parsing, IMU axes/scale, LED geometry, controller grip/aim transforms, display timings and transport. Similar appearance or shared WMR branding is insufficient.

Create a new profile from measured capabilities. Inherit only demonstrated common behavior. GPU/display workarounds must be conditional on driver version and capabilities; an NVIDIA workaround must not run on AMD/Intel. Linux/desktop differences belong in platform adapters and dependency detection, not scattered estimator constants. A future installer should fail clearly on an unsupported combination and offer read-only diagnostics.

A profile can move from `untested` to `development-reference-not-tracking-accepted`, then `experimental-qualified`, then `stable-qualified` only with linked evidence satisfying the release gates. The current validator accepts the schema and the first two states only; qualified states need release-manifest support before implementation.
