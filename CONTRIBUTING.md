# Contributing

Start with the current status and roadmap. Explain a concrete problem and expected behavior. Prefer a small patch with a deterministic reproducer and regression against the old behavior. Report the validation boundary: source tests, live packet/runtime checks and physical acceptance are different claims.

Keep coordinate frames, timestamps, units, uncertainty, prediction and world resets explicit. Do not add smoothing constants as substitutes for diagnosis. Separate graphics, tracking, model alignment and packaging changes. Include rollback and update the iteration ledger.

Public issues must not include room images, raw telemetry, full logs, serials, Steam account IDs, registry/calibration data or secrets. Use the minimal issue template first. Maintainers should request a sanitized summary or synthetic reproducer before any private recording. No automatic telemetry/upload is acceptable.

Retain upstream notices. New continuation files use LICENSE-CONTINUATION; modified runtime/NVIDIA/mutter code retains its existing terms. Generated controller assets need source/license provenance. Do not redistribute proprietary Windows binaries.

Use GitHub's noreply commit email if you do not want your personal email in public history. Inspect the staged diff and generated artifacts before pushing. Contributions do not imply vendor endorsement.
