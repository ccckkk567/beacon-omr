# One-hour native training smoke jobs

The user requested short jobs because the 18-hour formal jobs were taking too
long to schedule. This is an explicit exception to the earlier request to stop
standalone preparation checks: these jobs execute actual training, not another
environment preflight. The user approved exercising checkpoint saving and then
immediately deleting the smoke checkpoints, retaining the evidence.

Each of the six independent conditions requests one hour, two H200, 32 CPUs and
256 GiB on testqueue. They have no dependency on the formal array or continuation
jobs. Shorter limits can fit backfill windows, but do not guarantee an immediate
start; see [Slurm scheduling](https://slurm.schedmd.com/sched_config.html).

The new controller directly imports and executes the unchanged frozen source
`c193e617acf03d94e18cad023934e65e64ff6923`. `OMR_SOURCE_ROOT` and the reused
runtime's source hook select that same checkout for Ray workers. No package
installation, model download, synthetic action or training monkeypatch is used.
Both controller and training checkouts must be clean and match their revisions.

The explicit diagnostic overrides are two updates instead of 150, no initial or
final evaluation, a separate SwanLab project `beacon-omr-smoke`, separate task
manifest files and a smoke-specific protocol signature. B16, G8, per-benchmark H,
prompt/response limits, reward, optimizer configuration, two-GPU worker topology
and the scheduled training tasks remain the original recipe. These diagnostics
are not formal learning-curve results. The optimizer's total-step horizon follows
the two-update diagnostic budget.

## Submit

From a clean checkout of the published controller revision:

```bash
python3 -I -S -B "$CONTROL/experiments/omr/smoke_training.py" submit \
  --root "$ROOT" --controller-revision "$REV"
```

Array indices 0/1/2 are ALFWorld/WebShop/ScienceWorld baseline; 3/4/5 are full.
Each invocation creates a new smoke array. Do not repeatedly submit while the
same short test is queued. The existing formal array 166362 stays unchanged.
The first submission was rejected by `QOSMaxSubmitJobPerUserLimit` and produced
no smoke job. The user then authorized cancelling the six continuations
166364_0 through 166369_5 to free submission slots for all six short tests.
Cancellation and a successful smoke submission still await returned evidence.

Logs: `logs/beacon-omr/smoke-<array>_<index>.out`.
Outputs: `analysis/beacon-omr/smoke-<array>/<benchmark>-<method>/attempts/<job>-<index>/`.
The startup prints the exact `REPORT` path. A five-minute warning requests a
normal update-boundary stop; ending with only one update is not a passed test.

## Acceptance and immediate checkpoint cleanup

`smoke-report.json` requires two native optimizer-version receipts, finite
gradient norms, updated weights in the second rollout, a complete update-2
checkpoint containing both ranks' model/optimizer/extra state, and successful
online SwanLab SDK shutdown. It reports nonzero-gradient update counts separately.
Online SDK success is not claimed to be cloud server readback.

The ordinary native save path writes and hashes the full model checkpoint. After
the training writer finishes and its private Ray cluster shuts down, the
controller saves `checkpoint-save.json` (actual manifest, sizes, hashes), deletes
only the smoke run's owned checkpoint directory and records
`checkpoint-cleanup.json`. It then summarizes the training logs. No extra model
reload or large tensor allocation is used to inspect saved metadata. The recorded
checkpoint path is historical once deletion succeeds; the smoke run cannot resume.

Deletion checks the exact smoke path under the assigned work root, array and
condition identity, a smoke ownership marker and the checkpoint's own owner.
Redirected paths and mismatched ownership are refused. Formal checkpoints,
model/data caches, logs and OMR interaction records are retained. Catchable save
failures retain their failure status and clean owned partial checkpoint files.
A scheduler SIGKILL or process-wide OOM can prevent cleanup code from executing;
the surviving logs/receipts must then establish what remains before targeted cleanup.

Passing the smoke test establishes execution of these paths for these two
updates. Full OMR reports `restore_path=not_exercised` when its actual sampled
tasks never restart from an admitted state; it does not invent a replay test.
Full evaluation, checkpoint reload, remote metric readback and long-run stability
remain outside this short test. GPU memory/RAM failures must come from actual
HPC output; local fixtures cannot establish that a save fits the node's memory.

Six local fixtures cover unchanged training shapes, separate monitoring and
storage, short independent submission, successful save evidence, missing-rank
rejection, owned partial-save cleanup and refusal to delete formal/foreign files.
