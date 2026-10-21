# ScienceWorld BEACON-aligned OMR, 2026-10-21

## Baseline continuation correction

After fresh submissions169823/169824 were accepted, the user clarified that the
old baseline can be retained. This supersedes the BOTH-from0 choice below.
Returned old baseline latest is update30 (COMPLETE checksum
26ee44cf840926069f4ca3f606ba233e6bbcb56f555830cfe8779af0fcb51a66); progress32
is not a complete checkpoint. Only pending169823 may be withdrawn. Full169824
continues from0 at frozen cee87234ea1ff3b52ad1c7740cb08354d2d2758e.

resume_scienceworld_baseline.py queues one named testqueue/18h continuation
of omr-v2-scienceworld-baseline-s0-fullbatch-20261019. It retains the original
execution-profile receipt and checkpoint until the native complete-generation
writer commits a newer one. The baseline-only compatibility bridge checks exact
source/checksum/update, reproduces ddae412's signature, compares effective configs
(including materialized150-step schedulers), manifest, schedule and protected
Git trees. Only disabled OMR profile IDs and runtime/monitoring fields differ;
Full checkpoints or scientific changes are refused. Model, optimizer, dataloader
and RNG restore through the production loaders. Update31 is the next optimizer
update. A durable continuation origin supports later strict same-source resumes.

Legacy SwanLab used log-sequence SDK steps; this continuation opens a linked
segment to avoid mixing axes or overwriting unsaved updates31/32. The SDK axis
starts at30, followed by31. Later continuous policy-update runs reuse their ID
with resume=allow when safe. No automatic child or old checkpoint deletion is
requested. The new baseline source differs from Full only in explicit resume
and monitoring orchestration; protected baseline learning code is identical.

The correction passed42 targeted CPU tests, including real old/new configuration
composition, the production checkpoint loader/writer with model/optimizer/data
fixtures, RNG and cursor restoration at30, subsequent same-source resume at35,
OmegaConf struct mode, SwanLab axes, pending-only cancellation and quota retry.
The Slurm script syntax check passed. Native GPU restoration and remote run
readback await user Tunnel execution; these local fixtures are not GPU evidence.

## Source and decision

The supplied milestone revision is implemented on an independent branch from
ddae412e6195d2f86dace7f5ce2630d155ac0c01, the previous frozen ScienceWorld source.
Later WebShop repairs are not incorporated. The user explicitly requests both
ScienceWorld arms start again from the initial model. Consequently no old model,
optimizer, archive or counts are migrated; old checkpoints/logs/payloads remain
on disk. The user-returned scoped live queue has no matching jobs.

The native wrapper still generates subgoal_completed from current_score>prev_score
and MiGPO's dynamic path still reads that field. These source files are unchanged.
OMR's original candidate trigger already required that flag; the removed extra
semantics were dynamic score targets, threshold refresh, and score-category D.

## Profiles and collection contract

| Profile | Value |
|---|---|
| signal_profile_id | scienceworld-beacon-boundary-v1 |
| statistics_profile_id | beacon-progress-after-start-v1 |
| novelty_profile_id | beacon-boundary-condition-v1 |

ScienceWorldBeaconTrace validates the exact native boolean, concrete task and
strict Goal label. Every real new flag has an occurrence record, including a
terminal action. Only nonterminal flags are candidates, including action H.
Already-done/padding slots and replayed prefixes contribute no new occurrences
or statistical trials. Raw scores are evidence, never new threshold targets.

The two future questions are sw_beacon_progress_after_start_v1 and Goal. B asks
whether any new action in this suffix has a native flag. No new action means
unknown; a valid executed suffix without flags means false. Each trajectory adds
one Bernoulli trial regardless of occurrence count. Goal does not imply B.
Capturing a child does not reset the parent's positive trial. Child trials begin
independently; prefix occurrences/action positions are separate diagnostics.
completed_event_ids is empty for the two future questions. StateEntry validates
this contract; new birth hashes include all three profile fields.

Discovery keys hash kind, strict TaskSpec key, raw post-action observation and
sorted native legal actions. No score threshold, history, step, update, time or
rollout/checkpoint ID is included. This is a visible-condition proxy, not a
semantic subskill or full-world identity; rendering/alias differences can still
change keys. Exact history deduplication remains separate. NativeReplay verifies
the native action receipt, and usable context is checked before novelty. Novelty
is recorded before buffer/capacity decisions, excludes the restart condition
itself and uses the controller's frozen pre-collection seen registry. Registry
updates happen only at commit. Archive limits/turnover, candidate RNG, mixture
sampling and L/D formulas/priors/decay are unchanged.

## Legacy classification and persistence

events.py retains score_event/score_reaches/observed_scores as legacy reference
and test code. New ScienceWorld Full explicitly selects the new trace and cannot
read that threshold path. full_protocol.py only updates new_thresholds in the
legacy non-ScienceWorld branch; ScienceWorld's threshold refresh loop is removed.
New checkpoint/commit records contain profiles instead of thresholds. Generic
event_ids, coverage_key and seen_categories interfaces carry the new questions
and condition keys. Loading a new-profile checkpoint rejects missing/mismatched
profiles, old score event counts and legacy category keys. No ignore-signature
or implicit statistical migration is introduced. Old Goal evidence stays in the
old runs. Complete-generation/model/optimizer/data/RNG/payload checks remain.

## SwanLab

New ScienceWorld runs use SDK step=actual policy update, while a local sequence
orders log records. Initial validation is step0. After restoring25, validation
logs at25 and the next training update at26, even in a new monitoring run.
Continuous online state uses the known run ID with resume=allow. Rollback beyond
the saved checkpoint, offline history or missing run ID opens a new linked
segment. Tracking/lineage writes use scoped open_dict under OmegaConf struct
mode. No old metrics are overwritten to hide rolled-back updates. Actual remote
reuse and upload still require runtime receipts.

API reference: https://docs.swanlab.cn/api/py-init.html (resume allow/must/never).

## Deployment

submit_scienceworld_beacon.py submits exactly two ordinary independent jobs:
omr-scienceworld-baseline and omr-scienceworld-full. Each requests
testqueue,18h,2H200,32CPU,256GiB. Both use the same frozen new revision and run tag
beacon-boundary-fresh-20261021, existing Qwen2.5-1.5B-Instruct/runtime/assets,
B16xG8,H30,150 updates, unchanged validation/save frequencies, perturn-legacy-v1
and sync-v1. Baseline environment/reward/optimizer behavior is unchanged.
There are no arrays, smoke jobs, cancellations or automatic continuation jobs.
Quota retry retains accepted IDs; old active ScienceWorld jobs and unrecorded
state in fresh output paths block duplicate submissions.

Journal: analysis/beacon-omr/training-submissions/beacon-boundary-fresh-20261021/jobs.json.
Runs: analysis/beacon-omr/omr-v2-scienceworld-{baseline,full}-s0-beacon-boundary-fresh-20261021.

## Validation and pending real evidence

Related CPU suites cover signal edge cases, terminal/H/prefix/padding behavior,
8-trial Bernoulli counts, frozen discovery independent of capacity, profile/state
isolation, actual collector restore/checkpoint paths, monitoring update axes,
standalone quota retry and legacy benchmark regressions. A fixed-input test runs
the real MiGPO functions and compares rewards, segment IDs and advantage tensors
with unchanged masks. Environment wrappers, MiGPO, learner workers and per-turn
rollout engine have no diff against ddae412.
Across the targeted suites,81 distinct tests passed; the final collector rerun
also passed after removing an unnecessary optional-module import from a fixture.

Full's formal allocation writes historical-beacon-signal-audit.json. This bounded
read-only reader checks actual old latest/COMPLETE metadata and samples committed
raw flag rows with source/update references, terminal cases and missing-field
gaps. It never loads model files or reuses old counts. Legal-action evidence may
be unavailable retrospectively, so historical condition keys are not invented.
The reader is tested locally; actual historical/native/GPU/SwanLab receipts and
new submission IDs have not yet been returned. Synthetic tests are not proof of
native simulator behavior or algorithm efficacy. No checkpoint migration report
is claimed because the user chose fresh training.
