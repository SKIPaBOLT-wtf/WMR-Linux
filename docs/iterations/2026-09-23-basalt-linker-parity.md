# Basalt linker parity — 2026-09-23

## Decision

The unmodified source rebuild failed the retained-release comparison: 481.6975 m versus 0.09054 m net displacement on the same complete 84.95 s replay. Hypothesis: a build-environment difference, rather than the optional getter, caused the divergence. Baseline: WMR-Linux `6c8d125fe30f3489aeddcd331778efb2a7810090`, monado-wmr `296c50e8822a047bc960fa31699eb6f99699a62c`, Basalt source `30ece25f4c7d86e6a9dbee7ff0ebd0b921344a67`, installed backend SHA-256 `de4f9d30ae0417203796d51ffa66cc56e7c79ec06394a0a60387d68e9516b73c`. Acceptance was full pose, velocity and feature-row parity on the original failing replay, with identical input and calibration. Rollback was to discard private relinked artifacts and leave the installed release untouched.

## Result

- Preloading the release's fmt 10.1.1 into the GNU ld rebuild selected that library but left all 2,553 output rows unchanged. This disproved fmt runtime selection as the cause on this input.
- Relinking the **same compiled unmodified Basalt objects** with mold 2.40.4 produced SHA-256 `a8da63a2839c725cb56863e2c890ece86b4cdb616440b777dbd802c054a1a944`. Its 2,553 output rows matched the retained release exactly. The GNU ld artifact matched only the first seven rows and diverged to 481.6975 m. Thus linker selection is the decisive build difference on this replay; the underlying ELF-level mechanism remains unknown.
- Relinking the optional getter candidate with mold produced SHA-256 `39c0da83c3518e4f7fd95d230b0741c22f9b12ae05418a0b2a0f1f5f882c293d`. Its 2,553 ordinary rows matched the release exactly, and all 2,553 exported bias timestamps matched their pose timestamps. This is source-output parity, not a physical learned-bias qualification.
- The read-only `wmrctl basalt-linker` gate accepts the mold 2.40.4 artifact and rejects the GNU ld artifact. The upstream Basalt build selects mold only if present, so silent fallback was possible. Seven existing package-tool tests pass. The gate checks linker metadata; replay remains a separate requirement.

No live driver, backend, settings, game path or Windows data changed. Installed driver SHA-256 remains `82d80aa3b3ba174b5066011add02450f2f06e0e30a52fab1f4de794300aa6c4a`; installed backend remains the release above. The two mold artifacts are private test outputs, not an installation stage or release binaries. The GNU ld artifacts remain rejected. No physical trial was run, so head/controller comfort and raw VIO drift remain unaccepted. A clean source build and another complete input are still required before considering any backend replacement.

Paired runtime source is `86658feb8ccefed3d58f0cf7b8804c43e933720d` (documentation update only); resolve this record's integration commit with `git rev-parse HEAD`. To roll back source tooling, revert the integration changes from this iteration; live rollback is unnecessary. Next action: diagnose raw head VIO drift on matched timestamps and raw estimator output, including visual observability, before presentation or controller effects.
