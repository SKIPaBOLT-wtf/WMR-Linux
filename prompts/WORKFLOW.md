# Which prompt to use, and when

Start **two independent tasks now** if you want parallel progress. Give both agents access to this repository and the paired `monado-wmr` checkout. Each agent follows [OPERATING-RULES.md](OPERATING-RULES.md) and works in its own branch/worktree. They do not edit the same files concurrently. The tracking task alone owns the live VR installation.

## Task A: tracking, start now

Give the agent this exact message:

> Work in the existing WMR-Linux and paired monado-wmr checkouts. Follow `AGENTS.md` and `prompts/OPERATING-RULES.md`; use `prompts/LOCAL-TRACKING.md` for detailed tracking guidance. Start with the highest-priority unresolved tracking diagnosis that available evidence can decide, including the retained-versus-rebuilt Basalt divergence where relevant. Preserve working display, game and input. Complete offline diagnosis, a bounded fix and regressions before asking for one specific seated headset test. Keep new runtime changes staged until their relevant gates pass; install reversibly only when the user is available for validation. Record results and the next decision in current status and one iteration note. Do not reopen settled tests without a changed premise.

The agent can inspect source, matched recordings, timestamps, frame math and synthetic tests while the headset is hanging. **You are needed** for a short moving-head/controller trial after a candidate is safely installed. Compilation, an idle headset or CI cannot prove comfort, drift or controller accuracy. If a candidate feels worse, report that and let the agent roll it back before another trial.

## Task B: package and portability, may run alongside A

Give a separate agent this exact message:

> Work in a separate worktree of WMR-Linux. Follow `AGENTS.md` and `prompts/OPERATING-RULES.md`; use the packaging, documentation, privacy and release sections of `prompts/CONTINUE.md`. Own only source-level packaging, profiles, installer transactions and their tests. Begin with the read-only doctor/profile foundation and implement the next small reversible step toward plan, stage, install, verify and rollback. Do not change the live VR installation or tracking algorithms, and do not claim that tracking is accepted. Coordinate shared status/manifest edits with the tracking task before merge. Keep the package experimental until hardware and clean-install gates pass.

This task needs no headset for source work or isolated installer tests. **You are needed later** for an actual install/rollback and full VR/game check before a usable binary release. Related headsets, GPUs and Linux versions need their own qualification; a configurable profile is not proof of compatibility.

## After an iteration

Do not wait for every tracking defect to disappear before packaging work. Do not start a fresh project fork or paste both long prompts into one task. After either agent finishes an issue, have the next task read the updated status, pairing manifest, latest iteration and open issues, then continue with the same task-specific message. Merge reviewed branches only after their checks pass and they do not conflict with the verified baseline. Use a short physical trial whenever a tracking change requires it; reserve the full display, tracking, controller, input, game, restart and rollback trial for release qualification. A source snapshot can remain public while those gates are open.
