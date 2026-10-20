# WebShop fresh start, 2026-10-20

The user explicitly replaced the checkpoint-continuation plan with fresh
Baseline and Full training and requested deletion of the previous checkpoints.
They report cancelling the pending Full job. Baseline168529 at bb049a6 reached
checkpoint restoration but failed when adding omr.resume_lineage to a structured
OmegaConf configuration. No new policy update is shown in that returned log.
The prior scheduler-value comparison had passed. The migration-only lineage
failure is not patched in this release: new runs have no migration request and
do not enter that branch.

## Execution scope

start_webshop_fresh.py reads the original retry receipt from
analysis/beacon-omr/repairs/webshop-scheduler-20261020/jobs.json. It checks the
bound source, exact command and both accepted job IDs; Full's ID is read from
that real receipt rather than guessed. Baseline is bound to the observed168529.
A fresh successful squeue query must show no active known old writer or child,
including COMPLETING. The user-reported cancellation is not treated as proof
that cleanup has finished. Both old run locks must be acquired normally.

Only these two checkpoint stores are deleted, relative to the configured root:

- analysis/beacon-omr/omr-v2-webshop-baseline-s0-update-fix-166698/checkpoints
- analysis/beacon-omr/omr-v2-webshop-full-s0-fullbatch-20261019/checkpoints

Before deletion, the old owner, exact latest35/latest10 pointers, COMPLETE hashes
and source commits must match. Both stores are validated before removing either.
The controller records the previous metadata and an inventory of file paths and
sizes outside the deletion targets, then unlinks only recorded files and removes
empty recorded directories. Links/redirected paths, changed contents and unknown
directories are rejected. Interrupted cleanup resumes from the inventory; a
retired store that reappears is preserved for inspection. Model weights are not
read, hashed or copied by the login-node controller.

Old experiment logs, rollouts, cost ledgers, archive restore payloads and audit
reports are retained. Other benchmarks and unrelated jobs are outside the scope.
The deleted optimizer/model checkpoint state will no longer be resumable.

## New training pair

Both ordinary jobs use a new source revision and new run tag
input-v2-fresh-20261020. Their names are omr-webshop-baseline and omr-webshop-full.
Each requests testqueue,18h,2H200,32CPU,256GiB. The existing initial model,
runtime/assets and shared input-v2 repair are reused. There is no migration
request or old checkpoint path. Baseline and Full share B16,G8,H15,7000/512 token
limits, fixed150 updates, seed0 and the existing native full-batch per-turn
execution and synchronous writer. Existing initial validation and save/evaluation
frequencies are unchanged. Both start at policy update0 and create fresh SwanLab
runs through the existing launcher; Full starts with an empty new archive.

Outputs are analysis/beacon-omr/omr-v2-webshop-{baseline,full}-s0-input-v2-fresh-20261020.
Operation, deletion and submission receipts are under
analysis/beacon-omr/repairs/webshop-fresh-20261020. The existing quota-safe submission
helper preserves accepted IDs across retries. A newly running job/checkpoint is
never included in old-checkpoint deletion. No arrays, smoke jobs or automatic
continuations are submitted.

## Validation and evidence limits

Seven focused local tests passed: exact deletion scope and unrelated-record
preservation, single submissions on retry, active-job rejection, changed-pointer
rejection before any deletion, interrupted cleanup, quota retry after a new
checkpoint appears, redirected-path rejection, and composed input-v2 profiles
returning from the actual checkpoint-load method at update0 without migration.
These are CPU fixtures. Actual HPC deletion, submitted IDs and GPU training
progress have not yet been returned.
