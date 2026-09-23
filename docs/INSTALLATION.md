# Installation, staging and rollback

There is no qualified consumer installer yet. The current deliverable is a source continuation with read-only doctor, profile and Basalt-linker checks. Existing `infra/install.sh`, `g2ctl heal` and other upstream scripts can make extensive system changes; they are retained as upstream history, not this fork's quick-start path.

## Available now

Run `python3 package/wmrctl.py doctor` to obtain a small allowlisted local report. It reads OS/kernel, GPU model/driver and installed library hashes. It does not enumerate serials, users, Steam accounts, processes, room data or environment variables. It performs no uploads or writes. Read the report before sharing: even allowlisted hardware/version information may identify a configuration.

Run `python3 package/wmrctl.py profile profiles/hp-reverb-g2-nvidia-ubuntu.json` to validate the profile. It does not apply environment variables. Runtime source/build instructions are in the companion repository. Build into a new staging directory; do not run an unreviewed install target against the live machine.

For a staged reference Basalt library, run `python3 package/wmrctl.py basalt-linker /path/to/libbasalt.so`. It exits 0 only when ELF `.comment` identifies mold 2.40.4; GNU ld and unknown linker metadata fail the gate. This is a read-only identity check. A matching complete-input replay and later installation/physical gates remain required. The upstream Basalt CMake build only uses mold when it is found, so check every artifact rather than assuming the requested build preset selected it.

## Intended transactional package interface

`doctor` (read only), `plan`, `build`, `stage`, `install`, `verify`, `rollback`, `uninstall`. Only doctor, profile validation and the Basalt-linker check exist in this snapshot. A plan must identify every destination, dependency and system-level operation. Default installation should be per-user and dry-run; kernel/desktop patches require their own separately reviewed operation.

A package manifest must include source commits, patches, dependency and toolchain versions, ABI, every file hash/mode, license, hardware qualification and test results. Binaries built with private workspace paths must be rebuilt with debug-prefix maps before public release. Do not package the current private build directory.

Installation must stop or refuse an active runtime, save a private transaction backup, atomically replace only specified files, register the driver through supported tooling, and verify actual loaded paths/hashes at a fresh startup. It must not overwrite the entire Steam configuration or replace another user's backend silently. Optional settings require explicit profile selection and a recorded original value.

Rollback must restore only transaction-owned files whose current hashes still match the installed transaction; surface subsequent edits instead of overwriting them. Uninstall must undo driver registration and restore only settings owned by the package. Never delete game data, global Steam settings, existing user calibration, or unrelated runtimes.

The reference machine already has an installation and backups. Use its private handoff and dynamically verified records for local rollback. No new runtime was installed by this project-bootstrap task.
