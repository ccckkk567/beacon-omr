# Repair execution, 2026-10-19

Branch: `codex/omr-repair-20261019`, reviewed base `311a19f`, failed training
source `98d7a695f4a4b333e2984b731330d088612cf81b`. Separate local worktree;
no running or frozen HPC checkout was edited. Orion is reachable only through
the user's VS Code Tunnel. All HPC execution below remains pending.

## Implemented changes

| Commit | Purpose |
| --- | --- |
| 803f820 | Native load-bound ALFWorld identity on reset/step; strict receipts; in-job native acceptance |
| d37d0d3 | Candidate buffer (16 per task), exact dedup before payload persistence, category-balanced admission ordering, protected bounded archive turnover |
| e5e16e0 | ScienceWorld native root contract separated from render diagnostics; exact replay history divergence still falls back |
| 8eeba08 | Optional active-only preprocess/pad/scatter with original-slot seeds and real per-turn synchronization receipts |
| 28aa470 | Generation costs before environment calls, phase timing, failed-attempt records, optional bounded writer and checkpoint log watermarks |
| deployment commit | Standalone named formal jobs, default read-only targeted recovery, no automatic resume; final regression refinements |

No change to `fsdp_vllm.py`, worker generation synchronization, sleep/wake,
TP/FSDP policy, `free_cache_engine`, memory fractions, CUDA graphs, B/G/H,
reward, prompts, optimizer or task schedules. No smoke/Pilot/ablation jobs.

Immediate Full repair uses **perturn-legacy-v1 / sync-v1**, matching the old
baseline's common generation/I/O profile. Active-only and bounded logging are
available for explicitly coordinated future starts via `submit_training_jobs.py`
profile options. They are not silently enabled for this repair. Candidate/pool
rules are versioned `bounded-turnover-v2`; old results are not rewritten.

## Latest user override: normal jobs, no automatic continuations

Every new condition is one independent `sbatch`, with no `--array` or
`--dependency`. Names are `omr-<benchmark>-<baseline|full>`; logs use
`<benchmark>-<method>-<jobid>.out`, attempts use `<jobid>-single`.
Resources stay 1 node / 2 H200 / 32 CPUs / 256 GiB / 18 hours / testqueue /
account prjsnnn0000269. The user will decide later when to request continuation.
Legacy array launch compatibility remains only for historical workflows;
do not use old paired/array submission scripts for new work.

## Protection and targeted repair

Latest user-reported queue, not a live query by the assistant:

| Jobs | Reported state | Action |
| --- | --- | --- |
| 167234_0, ALFWorld baseline | RUNNING 2:23:12, GDS9B74 | KEEP |
| 167235_0, its existing continuation | PENDING dependency | KEEP |
| 167234_3, ALFWorld Full | FAILED 1:09:09, NonZeroExitCode | Inspect evidence; restart separately only if no updates/latest |
| 167236_3, broken Full continuation | PENDING Priority, dependency cleared | Exact pending-only withdrawal after binding checks; no new continuation |
| 167237_1/4, WebShop | PENDING | KEEP |
| 167238_1 / 167239_4 | PENDING dependency | KEEP |
| 167240_2/5, ScienceWorld | PENDING | KEEP |
| 167241_2 / 167242_5 | PENDING dependency | KEEP |

`repair_failed_full.py` defaults to read-only inventory. It verifies original
submission receipts, live owner/account/script/source, exact Full element and
failed run evidence. `--apply` permits only the verified pending old child to be
withdrawn. An already running child is preserved and reported. Query failure is
not absence. Old optimizer receipts or model checkpoint files stop replacement;
no cross-version migration is attempted. Only one new `omr-alfworld-full` job
is submitted, with quota-safe intent/accepted-ID journaling. Other jobs and all
old evidence remain. No node exclusions or machine-wide cleanup.

New run tag: `receipt-fix-20261019`. New SwanLab run carries source,
parent_failed_run, parent_attempt and execution/writer profile. No new actual
job ID or SwanLab link exists until the user returns execution output.

## Evidence levels and acceptance still needed

**82 targeted local CPU tests passed**, using production wrapper methods,
DataProto, collector loop, archive/controller, model-checkpoint orchestration
and logger. Simulator transitions, Ray transport, Slurm and two-rank model
files are fixtures. This is not native simulator or GPU-update acceptance.

Covered: reset-only extras, None/list shapes, explicit conflicting game files,
double reset, replay and continuation; late candidates and cap protection;
global/task/category turnover and quarantine; ScienceWorld render versus exact
restart; active counts 128/127/3/2/1/0 and actual collector early-termination flow;
byte-bound/backpressure/error propagation; immutable chunk/checkpoint restoration
and corruption rejection; generation costs surviving environment failure;
standalone job naming, no arrays/resumes, quota retries and live-state races.

In the repaired formal allocation, a single independent training-task simulator
instance check runs before long validation, using worker_python and the production
factory. It records actual TextWorld/wrapper/worker paths, identity fields,
reset, native steps, capture/replay and continuation. No acceptance trajectory
enters training/archive/evaluation. `ALFWORLD_RECEIPT_ROOT_CAUSE.md` distinguishes
the source-supported explanation from missing HPC raw evidence.

Still needed: native acceptance receipt; first real Full optimizer update;
the existing checkpoint boundary saving both ranks plus consistent algorithm
state; naturally observed candidate service/reuse if any. Actual resume is
deferred to a later user-requested continuation. No live training is killed
to manufacture resume evidence.

## Metrics and durability

`perf/train/*` and `perf/eval/*` use driver monotonic wall time. Generation RPC
timing is inclusive of sync/wake/generate/sleep; observer includes capture.
Nested times must not be added as if disjoint. Learner timing remains under
`timing_s/*`; `perf/checkpoint/save_wall_s` records full model-save wall time.
Phase heartbeats have wallclock/monotonic/attempt/update/local_step and are
rate limited. They indicate activity, not a completed update.

`rollout/model_requests` includes all actual engine requests and DP padding.
For active-only with evaluation instance padding skipped:
`model_requests = active_requests + generation_padding_requests`.
Legacy additionally reports `legacy_done_model_requests` and
`eval_padding_requests`; neither is hidden as saved compute. Seeds, original
slots, compact indices, generated receipts and response tokens are local audits.
`used_training_steps` excludes training copies; failed unverified steps never
receive invented reward labels. `cost/train/*` and `cost/eval/*` are separate
and include failed attempts. Old `cost/cumulative_*` aliases retain their legacy
definitions; allocation totals are known completed receipts, not a fabricated
complete total for interrupted/missing allocation records.

Archive metrics include observed/exact-duplicate/buffered/replaced/rejected
candidates, admissions/untried-cap rejection, active coverage and retirements;
cohort tables retain first-trial service and pending/right-censored states.
L/D/wait summaries and channel counts are low cardinality; per-state views,
event N, full selection probabilities and errors stay in local records.

Optional bounded writer: immutable bytes, 64 MiB pending-byte limit, one writer,
backpressure, atomic closed gzip chunks, flush after 30 seconds or 8 MiB, mandatory
collection/exception/checkpoint drain, visible queued versus durable watermarks.
A hard kill can lose nondurable tail. Checkpoints reference only drained chunks
with hashes, retain the old complete latest until new publication, and preserve
active/latest restore dependencies. Exactly one complete model generation is
retained after successful publication. No HPC checkpoint deletion was performed.
