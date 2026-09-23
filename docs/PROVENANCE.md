# Source provenance and publication boundary

Integration ancestor: [AshishKumar4/Project-VR](https://github.com/AshishKumar4/Project-VR), commit `d92aa7f1877ae8f77aff81e476dd90e881e652b2`. Runtime ancestor: [AshishKumar4/monado-wmr](https://github.com/AshishKumar4/monado-wmr), branch `g2-linux-integration`, commit `431ee47b89565ce6cbf57ea43f3dad6d0361ec44`. Both original Git histories are preserved. This continuation's working branch is `g2-stability`.

The runtime companion import manifest identifies 40 exact source/resource files carried from the tested local snapshot. Generated Python bytecode was excluded. Additional regression harnesses use synthetic inputs, not private sensor captures. See its `doc/g2-stability/PROVENANCE.md`, `import-manifest.json` and build/test guide. `project-manifest.json` pairs this integration tree with its runtime commit; the containing integration commit is obtained through Git to avoid a circular hash.

Basalt is a separate dependency from [mateosss/basalt](https://gitlab.freedesktop.org/mateosss/basalt), release source `30ece25f4c7d86e6a9dbee7ff0ebd0b921344a67`. The retained binary hash is documented for reference. No rebuilt or private binary is published here; the known failed rebuilds are not a release candidate.

NVIDIA display continuation patches are separately recorded under `patches/continuation/`. They are source diffs, not automatic installation instructions. Existing system modules/initramfs and Steam settings are not repository contents.

## Licensing

Original Project-VR documents its own code as offered in the “MIT spirit” but has no separate standard license file at the pinned revision. Its existing files retain their original statements; this continuation does not retroactively relicense them. Clarifying upstream licensing is a distribution follow-up before bundling those tools into a consumer package. Newly authored continuation files use LICENSE-CONTINUATION.

Monado is primarily BSL-1.0 with per-file exceptions and third-party notices; the optional VIT header retains BSD-3-Clause. NVIDIA/mutter patches follow their target files' licenses. Preserve notices and audit packages rather than labeling the whole stack MIT.

Public G2 controller models derive from the MIT-licensed immersive-web/webxr-input-profiles assets, pinned to commit `f4992299601614adbfefd398dc8e281556bb7444`. They are not extracted proprietary Windows or SteamVR Home assets. Their source hashes, conversion description and license are retained in the runtime fork.

## Privacy boundary

Published material is reviewed source, synthetic tests, generic policies, aggregate findings and documentation. Excluded: room imagery, raw camera/IMU/pose recordings, registry hives, spatial maps, unit calibration, device/account identifiers (including hashed encodings), personal contact details, credentials, full logs, private paths, compiled local debug binaries and configuration backups.

The private read-only Windows comparison establishes limited calibration/driver facts. It does not reveal Microsoft's complete tracking implementation. Numerical source/artifact hashes are public provenance, not hashes of private device identities.

Public Git attribution for new commits uses the owner's handle and GitHub noreply address. Existing public upstream author and copyright notices remain intact.
