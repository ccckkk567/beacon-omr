# Full-batch release — 2026-10-19

Base: repair branch `codex/omr-repair-20261019`,
`e3e36b444989420b7b52a0c6cc47362db5b11ba6`.
Release branch: `codex/omr-restore-fullbatch-20261019`; the full deployment SHA
for training is `ddae412e6195d2f86dace7f5ce2630d155ac0c01`, bound into each
new submission, profile, launch and resolved config. A later controller-only
fix is selected separately with `--controller-revision`. Upstream reference remains
`c88af45736b0a178106bc14ea89603ff02d8b222`.

## Finding and narrow change

The four previous repair submissions already explicitly selected
`perturn-legacy-v1 / sync-v1`. The two protected running baselines use the older
full-batch source. There is no evidence that active-only was enabled in these
formal jobs. No collector/helper rollback was necessary.

The launcher now checks both `execution-profile.json` and `repair-lineage.json`
for matching source/execution/writer bindings before runtime creation or learner
imports. A conflict fails rather than silently letting the latter file overwrite
the former. Missing historical bindings retain legacy/sync defaults; old active
and bounded profiles remain readable. Source/profile resume signatures remain
strict. No historical record is rewritten.

New authorized formal submissions explicitly use `perturn-legacy-v1`; writer
selection is copied independently from each old run (currently recorded sync).
The ordinary standalone submitter and its quota/uncertain-outcome journal are
reused. There are no new continuations, arrays, smoke jobs or node exclusions.

Collector, terminal cache, active reward/length filtering, MiGPO, OMR candidate
buffer/dedup/admission/archive, ALFWorld identity/acceptance, ScienceWorld root
contracts, per-turn FSDP/vLLM sync/sleep/wake, cost/audit writer, SwanLab, task
splits, horizons and learning settings are unchanged. Model checkpoint rotation
still commits the new complete generation before removing the old latest;
active/latest OMR payload references remain protected.

## Actual jobs and deployment decision

The user requested cancellation of exactly four newer jobs, then explicitly
requested resubmission. Returned `sacct -X` shows each CANCELLED by UID 173020;
the exact-ID live query is empty. This later scheduling authorization explains
resubmission despite those jobs already selecting legacy. The default rule
against restarting other already-legacy jobs still applies.

| Condition | Previous job/source | Observed state | Action |
| --- | --- | --- | --- |
| ALFWorld baseline | 167234_0 / 98d7a69 | Last returned RUNNING | KEEP |
| WebShop baseline | 167237_1 / 98d7a69 | Last returned RUNNING | KEEP |
| ALFWorld full | 167422 / d9e9d2f | CANCELLED | New standalone if no update/checkpoint |
| WebShop full | 167425 / d9e9d2f | CANCELLED | Same |
| ScienceWorld baseline | 167426 / d9e9d2f | CANCELLED | Same |
| ScienceWorld full | 167427 / d9e9d2f | CANCELLED | Same |

Existing baseline children 167235_0 and 167238_1 are preserved. No other job is
cancelled by this release. All old outputs and frozen checkouts are retained.

`experiments/omr/submit_fullbatch.py` checks fresh queue/accounting plus original
submission receipts. It audits all six run directories, source/profile bindings,
launch/resolved paths, optimizer receipts, latest/checkpoint data and at most four
bounded generation-record samples. An existing update, partial/complete model
checkpoint, conflicting source or active old target blocks fresh submission.
There is no cross-version/profile migration or actor-only pseudo-resume.

After `--apply`, detailed inventory is at:
`analysis/beacon-omr/training-submissions/fullbatch-20261019/inventory.json`.
Accepted new IDs are in the four adjacent `benchmark-method.json` journals;
rerunning the same frozen command retains accepted jobs after quota rejection.
Each new run uses tag `fullbatch-20261019`, 2 H200 / 32 CPU / 256 GiB / 18h,
an explicit `omr-{benchmark}-{method}` name and its own SwanLab run. Credentials
remain runtime-only. New IDs and GPU execution receipts are pending the user's
Tunnel execution; no submission is claimed here.

Actual HPC `launch.json`, `resolved-config.json`, completed update/latest and
partial-done generation records have not yet been returned for these four jobs.
Current profile findings come from frozen submission bindings/source, not an
observed HPC partial-done turn. The inventory labels absent/unsampled evidence;
it does not convert configuration into a runtime acceptance claim. On the next
normal formal rollout, verify full `original_slots`, positive legacy-done count,
zero avoided-done count and unchanged effective rows when a trajectory naturally
terminates early. No extra GPU probe is needed.

## Local validation

CPU fixtures use the actual collector, DataProto padding/scatter, native-terminal
cache, observer, collation/masking and BEACON advantage functions. Lengths
`[1,3,2]` produce RPC widths `[4,4,4]`: 12 model requests, 6 active requests,
3 DP padding and 3 legacy-done requests, zero avoided-done requests. Native
steps/rewards/lengths remain `[1,3,2]`, six valid samples, no fourth RPC. All
original rows preprocess/decode and retain original seeds, IDs and real sync
receipts. Done rows generate deliberately invalid responses but cannot mutate
native environments or Full OMR events/candidates/outcomes.

Active/legacy effective actions, rewards, terminal labels, masks and nonzero
episode/segment/joint advantages match in controlled fixtures. Tests cover early
success/failure, all-done, H truncation and evaluation tails with real-only result
denominators. Historical active tests remain `[4,2,2]`.

All 80 related combined CPU regression cases passed, including Full OMR
terminal filtering and actual runtime evidence/conflicts.
Coverage includes profile composition for all three benchmarks and
both arms, writer independence, source conflicts, standalone retry/quota guards,
candidate/archive rules, identity/root contracts, real sync guards, bounded
writer durability and latest-checkpoint state. These are fixtures on local
Python/Torch, not native simulator, CUDA, two-H200 or completed training evidence.

## Submission-journal path correction

The user's first `ddae412` apply exited during inventory with FileNotFoundError,
before training checkout creation or any sbatch call. The queued-repair run tag
is `pool-fix-20261019`, but its actual submission journal is
`repairs/queued-pool-fix-20261019/jobs.json`. The reader incorrectly treated the
run tag as the journal directory. ALFWorld's independent receipt directory was
already correct. Historical files must stay at their original paths.

The controller now uses an explicit run-tag/journal-directory mapping. Fixtures
use the historical controllers' actual layouts independently of the reader;
the returned failure was reproduced before the fix. All 17 targeted submission
tests pass, including correct receipt paths, zero-update/checkpoint protections,
writer preservation, partial quota retries and protected jobs. The retry uses
the corrected controller with `--revision ddae412e6195d2f86dace7f5ce2630d155ac0c01`
and a separate controller SHA. It retains the same new-run tags and idempotent
journals; no source migration, history rewrite or extra GPU probe is involved.
