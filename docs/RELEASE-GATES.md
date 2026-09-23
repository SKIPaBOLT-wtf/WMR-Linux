# Qualification and release gates

1. **Provenance:** source/dependency commits and licenses recorded; runtime and package revisions cross-linked. Preserve upstream attribution. Rebuilt backend behavior must be qualified independently of a read-only ABI extension.
2. **Privacy:** inspect exact diff/archive/commit authors. No room data, raw telemetry, serials, Windows registry/maps, unit calibration, accounts, tokens, private paths or build debug strings. An automated scan is a supplement.
3. **Build:** clean build and relevant production-code regression suites; targeted negative controls; sanitizers for memory/concurrency changes. Record toolchain, flags, artifacts and limitations. Do not state portability from one host build.
4. **Tracking:** synchronized frame/time contracts, quality/degraded behavior, repeatable controlled input comparison, head/controller independent motion, fast stops, occlusion/reacquisition and realistic small movements. Include long enough tests to detect slow drift. Static or zero-motion tests alone fail this gate.
5. **Presentation/game:** physical display, actual refresh and frame timing, tearing, restart/reboot, Home/input, and DiRT Rally 2.0 route checked separately. Refresh rate is not achieved FPS.
6. **Package:** offline plan, atomic transaction, clean install, idempotence, upgrade, verify actual loaded libraries, rollback/uninstall and user-edit conflict detection. No implicit kernel/desktop replacement.
7. **Documentation:** versioned status, hardware matrix, known failures, reproduction, install/rollback and next steps match artifacts. No “stable” or “plug-and-play” labels while tracking remains unaccepted.

A source snapshot can be published before physical qualification when its limitations are prominent. An experimental binary still needs installation/rollback and regression gates; it must list missing physical coverage. Stable releases need all applicable gates and actual user acceptance.
