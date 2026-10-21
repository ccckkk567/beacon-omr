# ScienceWorld baseline: replace cancelled continuation, preserve restored steps

## Returned HPC evidence

- Parent 167699, `omr-scienceworld-baseline`: TIMEOUT after 18:00:20.
- Old child 168029, `omr-scienceworld-baseline-resume`: CANCELLED, elapsed 00:00:00.
- Latest: update 30, generation `global_step_000030`, COMPLETE SHA256
  `26ee44cf840926069f4ca3f606ba233e6bbcb56f555830cfe8779af0fcb51a66`.
- User requested exactly one replacement on h200p, 72 hours, two H200. Preserve
  other jobs, including WebShop child 168994 and ALFWorld jobs. No new child chain.

## Monitoring behavior

Prefer the persisted SwanLab ID and sequence. An existing checkpoint rollback
opens a linked run as before. Missing monitoring IDs now create a new run rather
than blocking model restoration. For this explicitly authorized continuation,
an online `resume=must` initialization failure gets one `resume=never` attempt
without an ID and with SDK `reinit=True`; its status records the parent ID and
exception type. Both failed attempts remain an error, not a silent offline run.

Cloud step is `max(previous_log_sequence + 1, actual_policy_update)`. A new run
therefore logs restored validation at update 30, if performed, then training at
31 onward. It never restarts this restored curve at 0 or 1. A continued run
retains its larger cloud sequence, and `axis/policy_update` remains the exact
training-update axis even when several records share a policy version.

Official SDK behavior for `id`, `resume`, and `reinit`:
https://docs.swanlab.cn/guide_cloud/experiment_track/resume-experiment.html
https://docs.swanlab.cn/api/py-init.html

## Narrow source compatibility, not a new optimization protocol

The checkpoint binds its signature to the original source
`ddae412e6195d2f86dace7f5ce2630d155ac0c01`. A new actual source revision is needed
for the monitoring fix. The new entrypoint uses the original revision only for
the scientific protocol signature, while recording the new real revision in
`source_commit` and `omr.monitoring_resume.actual_code_revision`.

This is allowed only with an explicit request under the exact baseline run,
matching the returned update-30 pointer and COMPLETE hash/source/ownership.
Git differences must stay within the exact monitoring, bootstrap, controller,
test and documentation allowlist in `science_baseline_resume.py`. The saved
execution profile remains perturn-legacy-v1/sync-v1. Configuration composition
is compared against the real original `compose_config`: the scientific config
and signature match, apart from declared source/monitoring metadata/options.

No changes to training.py, the optimizer, scheduler, actor, native environments,
collectors, OMR Full events, task manifest or schedule are authorized here.
Native checkpoint signature, actor/optimizer/extra/data state and RNG restore,
ledger hashes and log-watermark checks remain in place. Existing frozen HPC
checkouts and old checkpoints are never edited by this controller.

## Submission and recovery

Run `science_baseline_resume.py submit --root ROOT --revision FULL_SHA --apply`
from the new frozen checkout. It checks the cancelled child via accounting,
rejects any live baseline writer/continuation, and journals exactly one named
normal h200p/72h job with two H200, 32 CPUs, 256 GiB. Retry uses the journal;
uncertain submission is never blindly repeated. Since the old parent/child are
terminal, no dependency on another running benchmark is added.

The GPU guard refuses missing, incomplete, wrong-source or changed u30 state.
The original native trainer then performs full checkpoint verification and
restores update 30; its next training update is 31. Default pre-train validation
and the per-turn sync/sleep-wake behavior are unchanged. New checkpoints record
the real new source revision and keep the original scientific signature; future
continuations must explicitly retain that compatibility provenance.

Persistent request and submission journal:
`analysis/beacon-omr/resume-chains/science-baseline-requeue-20261021/`.
Job stdout: `logs/beacon-omr/scienceworld-baseline-resume-JOBID.out`.
Inspect the new attempt's `restored-state.json`, `resume-lineage.json`,
`swanlab/omr-transport.json` and `swanlab/omr-metrics.jsonl` for actual restoration,
cloud ID, first cloud sequence and policy update. Cloud read-back, native GPU
restoration and the replacement submission are pending until returned by HPC.

## Local validation

22 tests passed: source/config signature equivalence against the original Git
object; wrong-scope, actor/environment-change and altered-checkpoint rejection;
one-job/idempotent submission and protected unrelated jobs; monitoring old-run,
fallback and restored-step behavior; existing strict resume and completion
tests. SDK/Slurm transports are fixtures, not evidence of completed HPC training.
