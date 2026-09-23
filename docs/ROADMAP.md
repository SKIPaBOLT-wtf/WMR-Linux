# Development roadmap

Each issue needs a reproducer, bounded hypothesis, baseline, acceptance criterion, regression scope and documented rollback. This order reflects current evidence, not a guarantee of one root cause.

| Priority | Work item | Acceptance criterion |
|---|---|---|
| P0 | Explain source-rebuilt Basalt divergence | Matched baseline reproduces retained release behavior within stated tolerances on controlled complete inputs; toolchain/dependency/config difference identified; no cherry-picked passing input |
| P0 | Raw visual-inertial head drift | Stable room-relative behavior during normal movement and stops, sensible degraded states, bounded long-session drift against an independent reference; positive feature count alone fails acceptance |
| P0 | Head-motion/controller coupling | Historical transform/timing invariants pass; stationary controllers stay fixed under head motion and independently moving controllers follow; no common-frame jumps or double reanchors |
| P1 | Micro-jitter and stop overshoot | Per-layer diagnosis, bounded prediction/bias correctness; reduced noise without unacceptable lag/overshoot or suppression of real movement |
| P1 | Controller optical robustness | LED ambiguity/multi-camera/occlusion/idle/reacquisition tests plus moving physical acceptance for both hands |
| P1 | Grip, aim and Home poses | Manufacturer versus app-specific transforms documented; rays/hands match both physical controllers through roll/pitch/yaw; subjective angle trial not treated as calibration |
| P1 | Floor/origin UI or CLI | Supported API, trustworthy reference, backup/undo, persistence test; cannot mask upstream drift |
| P1 | Reproducible build and transactional installer | Pinned sources, clean-host build, staged manifest, ABI/hash checks, idempotence, upgrade and rollback with subsequent-edit detection |
| P2 | Additional WMR/GPU/Linux profiles | Capability audit and measured tests per combination; unsupported states accurately shown |
| P2 | Qualified experimental release and discovery | Release gates passed, searchable README/topics, contributor tasks and honest demonstration; no broad stability claim before evidence |

Potential bounded contributor tasks: synthetic transform-invariance tests; timestamp-unit documentation; profile schema validation; transaction rollback tests; clean-build dependency pinning; privacy-safe issue triage. Do not mark estimator fixes “good first issue” unless a deterministic reproducer makes the scope suitably narrow.
