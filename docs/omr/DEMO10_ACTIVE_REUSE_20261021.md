# Demo10 active reuse v1

This experiment is new. It does not replace, migrate, cancel, or modify any
historical experiment. All six conditions start from the locked base model.

## Integration and scientific comparison

Base: `aa891ae4a34d95bb10a379e86530d21c9bd66021`, which contains WebShop input-v2
and fresh-start fixes. Imported: `cee87234ea1ff3b52ad1c7740cb08354d2d2758e`, the
ScienceWorld native BEACON boundary profiles and policy-update SwanLab axis.
The monitoring conflict was resolved by retaining both WebShop's legacy migration
branch and ScienceWorld's continuous policy-update support. Demo entry points
never invoke any legacy migration or retirement controller.

All six jobs use the same final revision, Qwen2.5-1.5B-Instruct snapshot, B16/G8,
150 updates, two H200,32 CPUs,256GiB, legacy per-turn full-batch execution and
sync writer. Horizons remain ALF30/Sci30/Web15. Rewards, segment advantages,
optimizer, state priority/staleness/coverage mixture and L/D formulas are unchanged.
The only native PPO edit gates the historical extra update2 checkpoint off for
Demo10; regular saves are every5 updates and at boundary stops/final update.
Evaluation remains root-only at0,15,...150 on the original validation split;
final test averages the existing three single attempts, not pass@3.

Baseline has16 root groups. Active Full has eight predetermined root slots and
eight archive slots. The independent immutable mask varies by update. After
freezing the archive, task selection is uniform without replacement over eligible
demo instances; fewer than8 eligible tasks are filled through balanced random
permutation cycles. Inner checkpoint selection bypasses the old alpha coin but
uses its original distributions and reservations. Only a committed previous
update can contribute states. Failed restores fall back once for the entire G8
group on the selected task. No extra groups or retries compensate for failures.

Task RNG, candidate RNG, inner sampler RNG, archive/statistics, model/optimizer,
data cursor, mask/bank hashes and policy update are checkpointed. The task planner
cursor must equal the model cursor on restore. New evaluation guards restore
Python/NumPy/Torch host RNG and reject archive/planner mutation. Generation uses
the existing split-specific per-request seeds. No importance weights enter loss.

## Frozen subset preparation

The user confirmed all three existing parent files on2026-10-21:

- `manifests/update-fix-167125/alfworld.json`
- `manifests/input-v2-fresh-20261020/webshop.json`
- `manifests/fullbatch-20261019/scienceworld.json`

Paths are relative to `$ROOT/analysis/beacon-omr`. Derivation reads only the
parent train split; validation/test retain exact content and order. Ceil10%,
seed20261021, native family quotas, stable instance hashes and identity checks
are implemented in `demo_manifest.py`. Missing/constant WebShop categories use
a reported single bucket. When category count exceeds budget, WebShop uses
proportional quotas without mandatory category coverage; this fixed policy does
not consult outcomes. ALF/Sci refuse impossible coverage. Subsets smaller than16
fail rather than silently sampling with replacement or increasing the subset.

CPU preparation reads existing package metadata and hashes actual model/tokenizer
files; it installs nothing. NumPy/OmegaConf are imported from existing package
files with a CPU-compatible Python module; no `.pth` execution, GPU model import
or simulator action is needed. Root schedules retain the original NumPy
RandomState implementation. Artifacts publish atomically under:

`$ROOT/analysis/beacon-omr/manifests/demo10-active-v1/`

Each benchmark directory contains parent_manifest.json, demo_manifest.json,
subset_membership.jsonl, subset_report.md, root_schedule.json, branch_masks.json
and FROZEN.json. The parent includes data-version/deduplication provenance.
`integration_manifest.json` adds package locks, model hashes, profiles, full
source diff summary and six config hashes. Actual counts/hashes are produced on
HPC, not inferred from fixture tests. Native training rebuilds the full parent
catalog and compares it before loading the derived subset.

## Submission and old-job protection

`submit_demo10.py queue-preparation` submits one CPU ioqueue job (2CPU,8GiB,30min).
After freezing and validating artifacts, that job records six complete dry-run
commands/config comparisons and submits six ordinary testqueue18h jobs. No
arrays, dependencies between conditions, automatic resumes or GPU smoke jobs.
The controller only uses read-only squeue plus new sbatch submissions; it never
calls scancel, scontrol update/hold/release, or any legacy recovery script.

Every new run is named
`omr-demo10-<benchmark>-<baseline|active-full>-s0-20261021`, with independent
outputs and SwanLab project `beacon-omr-demo10`. A kernel submission lock and
durable intents prevent duplicates. Quota rejections retain accepted IDs;
unknown submission outcomes require inspection. Repeating `submit` only fills
missing conditions. The current old-job snapshot includes168994,168028,
167235_0,169826,169824; they are protected, as are all other existing jobs.

Shared filesystem free space must exceed288GiB for six latest+staging stores.
This is a filesystem check, not a claim about project/user quotas. Existing
checkpoints, archives, environments and source checkouts are never deleted.
Each new run keeps only its latest model generation after atomic two-rank
verification/publication. Restore payload GC still retains all live/archive and
latest-checkpoint references. Lightweight scientific logs remain.

## SwanLab, acceptance and reporting

All six runs log SDK steps at actual policy update. Resume reuses the same online
run when compatible; rollback, offline state or missing ID opens a linked new
segment starting at the restored update. No old curves are merged.
`configs/demo10_metrics_schema.json` maps panel keys and detailed record paths.
Allocation records include failed attempts; terminated allocations without final
receipts are explicitly flagged as needing Slurm accounting, never reported as
zero-cost. CPU contract tests do not establish GPU/native simulator acceptance
or remote SwanLab readback. Actual formal-job receipts remain required.

After training begins, run `demo10_report.py --root "$ROOT"` with isolated Python
to read submission IDs, progress, latest checkpoints, SwanLab transport URLs and
the first committed8:8 native collection for each active arm. Missing evidence is
reported as null; a successful reset alone does not count as an optimizer update.

Explicit future continuation (only after that condition leaves the queue):

```bash
python3 -I -S -B "$CONTROL/experiments/omr/submit_demo10.py" resume --root "$ROOT" --revision "$REV" --condition scienceworld/active_full
```

It restores this new run's complete latest checkpoint only, uses the identical
frozen source/config, submits one standalone18h job, and never creates children.
No continuation is submitted by the initial six-job request.
