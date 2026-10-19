# Execution status — 2026-10-19

## Current repair branch and standalone submission policy

The user now authorized upgrading the remaining queued conditions to the repair
release: WebShop Full 167237_4 and ScienceWorld baseline/full 167240_2/5, with
pending old children 167239_4/167241_2/167242_5 withdrawn. All three new jobs
must use training d9e9d2f, matching ALFWorld Full 167422, as independent named
jobs with no new resumes. Running ALF/Web baselines and 167422 stay protected.
The targeted controller validates fresh states and original receipts, then
materializes and cancels only the six authorized pending elements, children
first. The user confirmed successful replacement using controller
`f0e883612326e767828d815b7d49f4a8d418d6b1`: WebShop Full **167425**,
ScienceWorld baseline **167426**, ScienceWorld Full **167427**. All three train
at `d9e9d2f3dc27c7c8617a00ab9ed92e370139cc8a`, independently, with no new
continuations. The six listed old elements were reported withdrawn. New tag:
`pool-fix-20261019`; journal `repairs/queued-pool-fix-20261019/jobs.json` under
`analysis/beacon-omr`. No execution/optimizer/checkpoint results returned yet.

| Benchmark | Baseline | Full |
| --- | --- | --- |
| ALFWorld | 167234_0, protected old source 98d7a69 | 167422, repaired source d9e9d2f |
| WebShop | 167237_1, protected old source 98d7a69 | 167425, repaired source d9e9d2f |
| ScienceWorld | 167426, repaired source d9e9d2f | 167427, repaired source d9e9d2f |


The user confirmed successful submission of **167422**, named
`omr-alfworld-full`, as one standalone formal job with no continuation.
Training/controller source remains frozen at
`d9e9d2f3dc27c7c8617a00ab9ed92e370139cc8a`; later documentation commits must
not migrate this run. Resources: 2 H200 / 32 CPUs / 256 GiB / 18 hours.
Run tag: `receipt-fix-20261019`. Log: `logs/beacon-omr/alfworld-full-167422.out`.
Submission journal: `analysis/beacon-omr/repairs/receipt-fix-20261019/jobs.json`.
Old child 167236_3 is FAILED; its guard reports blocked because no complete
latest checkpoint existed. No new running, optimizer, checkpoint or native
acceptance evidence for 167422 has been returned yet. Preserve other jobs.

See [repair execution report](REPAIR_EXECUTION_20261019.md). Local CPU fixtures
pass; native ALFWorld/GPU acceptance is pending. The authorized repair rollout
above is submitted; running baselines and unrelated jobs remain protected.
New submissions are independent named normal jobs, no arrays, no automatic
continuations. Preserve per-turn sync/sleep/wake and default legacy/sync profile.

## Previous frozen submission: six formal tasks and continuations

The user returned successful submissions from controller
**1d30ff32a790993b2b4738b32303e14d2282f1a4**, with all training frozen at
**98d7a695f4a4b333e2984b731330d088612cf81b**:

| Benchmark | Arm | Formal task | Own-parent continuation |
| --- | --- | --- | --- |
| ALFWorld | baseline | 167234_0 | 167235_0 |
| ALFWorld | full | 167234_3 | 167236_3 |
| WebShop | baseline | 167237_1 | 167238_1 |
| WebShop | full | 167237_4 | 167239_4 |
| ScienceWorld | baseline | 167240_2 | 167241_2 |
| ScienceWorld | full | 167240_5 | 167242_5 |

Each task requests 18 hours, two H200, 32 CPUs and 256 GiB. The three formal
arrays are independent with two concurrent arms each; each continuation has
`afterany` on only its own formal task. Resume requires a complete latest
checkpoint from that new logical run. No smoke jobs or node exclusions.

Cancellation was requested for 167126_0, 167127_3, 167129_2, 167130_5,
167125_0, 167125_3, 167128_2 and 167128_5. These targets were withdrawn or
already exiting before replacement; the output does not establish that node
cleanup finished. Old files remain preserved and ancient 166362_2 is untouched.
The journal remains `training-submissions/first-update-167125-166698-167128/`.
Formal tags are `update-fix-167125`, `update-fix-166698`, `update-fix-167128` for
ALFWorld, WebShop and ScienceWorld respectively. Do not submit another chain or
change the frozen source merely because a newer documentation commit exists.
This is submission evidence only: new startup, optimizer, checkpoint and SwanLab
execution results have not yet been returned.

## Earlier recovery: historical-attempt source check repaired

The user ran controller **5bad4e6** with training **98d7a69**. It stopped before
any cancellation or submission at WebShop baseline's historical attempt
**166511-1**, which used **c193e61**. The controller incorrectly required every
historical attempt to match the current **d6dff44** WebShop source.

The repaired controller distinguishes the current progress-selected attempt and
live Slurm jobs (still pinned to their exact expected source) from retained
historical attempts (valid source identity and matching benchmark/method/output
directory, but not necessarily the same revision). It records historical source
identities in the retirement receipt. Optimizer receipts across **all** attempts
and shared checkpoint contents still block replacement. No old files are removed.
The same failure was reproduced locally before the fix; **20 local recovery and
submission tests now pass**, including mixed-version history, current-source
mismatches, historical optimizer/checkpoint evidence and idempotent submission.
Use the newer controller with the same training revision, options, journal and
output tags. The previous failed invocation created no new job chain.

## Earlier recovery pause: an old ALFWorld job started

The user executed recovery at **98d7a69**. It stopped in the initial evidence
check, before any cancellation or new submission. The follow-up queue shows
**167125_0 RUNNING at 33:11**, with attempt **167144-0**, source **5dd260d** and
progress `starting`. This status is written before initialization and is not
updated until a completed optimizer update; it does not identify the current
initialization/evaluation/rollout stage. ALFWorld **167125_3** and ScienceWorld
**167128_2/5** remain pending, as do their four continuations. Old WebShop
continuations **166699_1/166700_4** are absent from the returned active queue.

Use the newer controller's explicit `--replace-started-zero-update` option,
keeping training frozen at **98d7a695f4a4b333e2984b731330d088612cf81b** via
`--revision` and selecting the controller with `--controller-revision`.
It still targets only the same ten exact old IDs. It validates every present
job's owner, account, script and frozen source, plus all old attempt sources,
progress, optimizer receipts and checkpoint contents. Unexpected updates,
checkpoints, output bindings or query failures stop replacement. The two
reported missing WebShop continuations may be absent on a fresh all-state query;
other missing jobs need this controller's own cancellation record.

Continuations are withdrawn first. A matching zero-update parent that moves
from pending to running can be withdrawn after fresh validation. Cancelled,
terminal or COMPLETING jobs can be replaced into the existing separate output
tags without waiting for node cleanup. No old files are deleted, no smoke jobs
are submitted, no nodes are excluded, and ancient **166362_2** is not a target.
The same submission journal preserves accepted jobs across quota retries.
**18 local recovery/submission tests pass**, including the returned queue,
start races, cancellation/query failures, new update evidence and retry safety.
Slurm transport is substituted; the revised controller has not yet run on Orion.

## Combined startup optimization, awaiting formal replacement

The user confirmed **d386a1d recovery has not been executed**, then requested
that repeated initialization and unnecessary imports also be fixed. Use the
combined newer frozen revision with the same first-update recovery controller,
old target IDs and new output tags described below. No new job IDs exist yet.

OMR WebShop now runs one CPU Ray preparation task inside the allocated GPU job.
It constructs products/prices/goals, parses each distinct title with the same
full `en_core_web_sm` pipeline, hashes index/source provenance, and builds the
restart manifest once. All 128 training and 16 evaluation actors receive the
same result ObjectRef as a top-level argument, following [Ray's documented
argument/lifetime semantics](https://docs.ray.io/en/latest/ray-core/objects.html).
This avoids repeated construction, not all per-process data deserialization.
Mutable sessions, goal overrides, histories and RNGs remain private; native
Lucene handles are still opened independently in each process.

Text environment imports no longer load Selenium or image-only Torch. Root
seeding still seeds Python/NumPy/native state, and Torch if already loaded,
without importing Torch just for CPU simulation. Unused cleantext/BM25 imports,
the duplicate human-attribute JSON read, repeated restart-manifest construction,
and the arbitrary 14.4-second WebShop startup sleep are removed. Actor method
futures still wait for their actual constructors to finish.

Pinned Pyserini 0.17's [search package initialization](https://github.com/castorini/pyserini/blob/pyserini-0.17.0/pyserini/search/__init__.py)
also imports dense encoders. The shared OMR path now uses a version-guarded narrow
bridge to the same Java `io.anserini.search.SimpleSearcher`, preserving the
[original string search/document calls](https://github.com/castorini/pyserini/blob/pyserini-0.17.0/pyserini/search/lucene/_searcher.py)
and raw stored document field. It uses the existing installed jar/index and
does not change ranking, analyzer settings, query text, k, or returned order.
Non-OMR callers retain the regular Pyserini API.

**36 targeted local tests passed.** Fresh-process checks forbid unneeded imports,
exercise native construction/search/click/purchase/reset with private sessions,
compare cached/uncached rewards (including repeated title nouns), check a single
preparation ref is used by both vectors, and validate the narrow Java call
contract. Four golden product/price/goal/RNG digests were independently compared
against the frozen d386a1d native constructors for synthetic/human goals with and
without goal limits. Fixtures substitute NLP token production, JVM and Ray
transport; they do not establish native Java/GPU acceptance or elapsed-time gains.
Earlier first-update, restore and guarded submission tests also pass in this set.
`webshop.static_catalog` / `webshop.create_actors` stages and
`OMR_WEBSHOP_STATIC` timings allow the next formal run to measure the effect.

## Current: first-update failures repaired; formal replacement prepared

The previously interrupted pending refresh succeeded with controller **175a616**:
ALFWorld formal **167125_0/3**, resumes **167126_0/167127_3**; ScienceWorld formal
**167128_2/5**, resumes **167129_2/167130_5**. Their training source is **5dd260d**.
The queue before the first recovery attempt showed all eight pending, plus old
WebShop resumes **166699_1/166700_4**. The newer returned state is recorded above.
Ancient **166362_2** remains COMPLETING and is not a target.

Both WebShop formal jobs **166698_1/4**, training source **d6dff44**, failed with
zero completed optimizer updates:

| Arm | Completed before failure | Direct cause |
| --- | --- | --- |
| Baseline | Initial evaluation and first 15-action training rollout | `collate_fn` stored boolean padding metadata in an object array; direct Torch conversion failed in `compute_response_mask` |
| Full | Initial evaluation | `_group_render` shallow-copied a vector owning live actors; the temporary copy's destructor closed/killed the shared actors before event-context collection |

The first cause applies to both arms on all three benchmarks. The second was
reproduced with both production WebShop and ScienceWorld vector destructors.
Neither traceback establishes GPU/RAM exhaustion or a wall-time limit.

The mask now normalizes padding metadata to a boolean vector and checks its row
count before conversion. Rendering previews isolate manager memory and metadata
without copying the actor owner. ALFWorld preview renders local admissible commands
without overwriting its live vector cache. Root/history agreement checks and the
real owner's normal cleanup remain intact. Experiment profiles and batch sizes
are unchanged; the common source also includes the earlier startup import fixes.

**58 targeted local tests passed**: 40 training/masking/restart/advantage fixtures
and 18 submission/recovery fixtures. The new tests first reproduced the original
object-array exception and premature actor destruction. They use real DataProto,
collation, padding, mask, manager rendering and vector destructor methods, with
Ray transport/native engines substituted. Local CPU tests use Torch 2.10 and
TensorDict 0.6.1, not the HPC Torch 2.6 runtime; GPU acceptance is still required.

`recover_first_update.py` checks the ten pending IDs above, withdraws continuations
before parents, and submits three independent two-element formal arrays plus six
own-parent continuations. New outputs/SwanLab identities use `update-fix-167125`
(ALFWorld), `update-fix-166698` (WebShop), and `update-fix-167128` (ScienceWorld).
All use one repaired frozen revision, 18 hours, two H200, 32 CPUs and 256 GiB.
There is no smoke submission, node exclusion or wait for ancient cleanup.
Previous run files remain intact. A target that starts, changes identity, or has
optimizer/checkpoint evidence stops replacement. Quota retries reuse accepted IDs
in `training-submissions/first-update-167125-166698-167128/`.
New submission output and a successful GPU optimizer update have not been returned.

The sections below are chronological records of earlier states and decisions.

## Pending refresh interrupted after array cancellation

The user ran controller **5dd260d6bf8bda538c16b28ccc4428e087d55e6d**.
It requested cancellation of 166854_0, 166855_3, 166750_2, 166751_5 and
166853_0, then stopped on a nonzero `scancel --state=PENDING 166853_3` reply.
No replacement was submitted. Its receipt already records cancellation intent
for those six elements and verified descriptors for all eight targets.

The follow-up queue shows array records **166853, 166854, 166855, 166750 and
166751 CANCELLED with zero runtime**; `scontrol` reports 166853_3 not found.
Only old ScienceWorld **166749_2/5** remain pending among the eight targets.
WebShop **166698_1/4** are still running (1:04:07 / 58:06), with their resumes
166699_1/166700_4 pending. Old 166362_2 remains COMPLETING and is not a target.

This is consistent with [Slurm's documented unsplit-array cancellation
behavior](https://slurm.schedmd.com/scancel.html). The controller now updates an
individual element's comment to materialize its job record, revalidates it, then
uses the pending-only cancellation filter. Nonzero cancellation replies retain
stderr and trigger a fresh queue read: only a cancelled/absent intended target
can be reconciled; running, pending, unknown states or failed queries stop
submission. Old training-attempt evidence still blocks replacement.

An optional `--controller-revision` allows this controller repair to continue the
existing retirement receipt while `--revision` remains the original **5dd260d**
training revision. Baseline/full, output tags, resources and continuation paths
remain bound to that training revision. All old files and accepted submission
records are retained. **16 targeted local tests passed**, including the returned
partial-cancellation state, benign and unsafe races, scheduler-error reporting,
quota retries and matching formal/resume bindings. These use a simulated Slurm;
new submission IDs and GPU execution evidence are still pending.

## Latest decision: continue with formal jobs only

The user requested cancellation of all project `omr-smoke` jobs and no further
smoke submissions. Preserve the formal training jobs and their continuations;
do not duplicate them. The returned queue now contains no smoke jobs.
Older recovery instructions below that include
smoke are superseded by this decision and must not be rerun unchanged.

The latest ScienceWorld baseline smoke **166748_2**, physical attempt
**166830-2**, source **80dd118**, failed on GDS9B74 with the same **300-second
rank-zero registration timeout**. No optimizer updates, actions, checkpoint or
SwanLab run were recorded. TaskRunner startup stages place 503.74 seconds
between `processor.load` and `workers.import_classes` (including the intervening
vLLM version-check import), and 429.37 seconds in placement/registration before
failure. These measurements identify expensive stages, not the worker-side
root cause. Formal training uses the same startup path, so withdrawing smoke
does not establish that the failure is fixed.

## Startup import repair and pending formal-only refresh

The returned TaskRunner self-stacks now identify the slow import at
`main_ppo.py:89`: the early vLLM version check imported `vllm_utils`, DeepSeek/MoE
modules and Torch compiler initialization even with `lora_rank=0`. The controller
now checks installed distribution metadata only when LoRA is requested and marks
that stage separately from processor loading. A sampled `filelock` audit hook is
not evidence of a lock deadlock; no package versions or locking policies change.

Simulator submodule imports also executed the environment package's eager
manager import, pulling Torch into CPU actors. Public manager exports now load
on demand; ALFWorld's image-only Torch/torchvision imports are deferred to the
image helper. The text simulation, batch/group sizes, seeds, rewards and training
algorithm are unchanged. The measured HPC speedup and the separate rank-zero
registration failure remain unverified.

The latest queue contains **WebShop 166698_1/4 running** with resumes
166699_1/166700_4 pending. Preserve them. **ALFWorld 166853_0/3** and
**ScienceWorld 166749_2/5**, together with their resumes 166854_0/166855_3 and
166750_2/166751_5, are pending. `refresh_pending_training.py` validates and
withdraws only these eight pending elements, then uses isolated output tags
`startup-166853` and `startup-166749` for four formal tasks and four continuations.
All retain 18 hours, two H200, 32 CPUs and 256 GiB; both formal arms and benchmarks
queue independently. No smoke or node exclusions are added. A target that starts
running or changes identity aborts replacement. Accepted submissions survive
quota retries. Old 166362_2 remains outside this controller and may still be
COMPLETING. No old logs, checkpoints, manifests or environments are removed.

**32 local regression tests pass**, covering real production controller execution
up to GPU imports with vLLM unavailable, simulator class imports/serialization
with Torch forbidden, existing reset/startup tests, pending-only cancellation,
start races, target binding, quota retries and own-parent continuation dependencies.
Ray scheduling and simulator engines are test doubles; these are not GPU results.
New submission records will be under
`training-submissions/startup-166749-166853/`; no replacement receipt yet.

## Current priority: ALFWorld explicit reset API repair

Baseline smoke **166613_0**, physical attempt **166727-0**, source **c685614**,
reached `trainer.fit` after model initialization, then failed at its first
`AlfworldWorker.reset_explicit`: `TextworldBatchGymEnv` has no `.unwrapped`.
The previously returned inventory identifies **TextWorld 1.7.0**. Its
[batch class](https://github.com/microsoft/TextWorld/blob/1.7.0/textworld/gym/envs/textworld_batch.py)
is returned directly rather than inheriting Gym's environment API.

The fix accepts the direct batch object or a Gym wrapper, selects the explicit
game on the actual iterator owner, verifies one game per worker and the returned
`extra.gamefile`, and restores the seeded iterator even when reset fails.
Both baseline roots and Full capture/replay use this same worker method. The
seed, task schedule, reward, batch, training algorithm and existing packages are
unchanged. The report contains **zero updates/actions and no checkpoint**;
deleting its empty checkpoint directory did not test a real save. SwanLab run
`y0whj971` was created online and the SDK reported 17 uploaded records, but the
smoke training/monitoring acceptance flags remain false.

**43 targeted local tests passed** (21 reset/import/full-protocol and 22 recovery/
submission/resume tests). Five reset tests were additionally run against the
unchanged TextWorld 1.7.0 batch class and shuffle utility obtained from its
official tagged source, reproducing the old AttributeError first. Only the
underlying game engine/import dependencies were substituted; this is not a
native PDDL task or two-GPU training result. Tests cover direct/wrapped reset,
task identity, iterator restoration on failure, batch manager output and Full
prefix replay/continuation through the production worker.

The latest queue shows **166613_3** smoke and **166629_3** formal running,
**166630_0** pending on priority and **166631_3** pending on dependency;
166613_0/166629_0 are absent. `recover_alfworld.py` checks only these six exact
IDs, requests withdrawal with continuations first, and submits the same-size
baseline/full smoke/formal/resume matrix into new **restart-166629** outputs.
The formal run and SwanLab name suffix is `-restart-166629`; both arms share
`manifests/restart-166629/alfworld.json`. Its new journal is
`training-submissions/alfworld-restart-166629/`. Old files remain intact, new
jobs do not wait for node cleanup, no node is excluded, and partial quota
retries keep accepted IDs. **Replacement ALFWorld job IDs remain pending.**

Preserve WebShop 166697_1/4 (both running), 166698_1/4 and 166699_1/166700_4,
plus the newly queued ScienceWorld jobs listed below.

## ScienceWorld independent restart: new jobs queued

The user now authorizes an alternative to waiting for the old COMPLETING job.
`isolate_scienceworld.py` prepares one new generation tagged **restart-166362**,
with one frozen source for controller, baseline/full and continuations:

- Two independent one-hour smoke jobs, two independent 18-hour formal jobs,
  and one 18-hour continuation after each formal parent. Every job requests
  two H200, 32 CPUs and 256 GiB. Per the user's clarification, node selection
  stays with Slurm: **JDS9B74 is not excluded**. Smoke and formal jobs have no
  dependency on each other or on old job 166362_2.
- Formal outputs are `omr-v2-scienceworld-{baseline,full}-s0-restart-166362`;
  their common manifest is `manifests/restart-166362/scienceworld.json`.
  Fresh SwanLab experiment names use these new identities. Continuations
  retain normal monitoring lineage and accept only their new formal run's
  complete checkpoint. Old training output is neither adopted nor removed.
- The old recovery must have submitted nothing. Under its original lock, the
  controller backs up `retirement.json` to `retirement-before-isolation.json`
  and marks the former request superseded. Frozen d89c798 then refuses to run
  that request again. Submission IDs are journaled at
  `training-submissions/scienceworld-restart-166362/jobs.json`; quota retries
  retain accepted jobs, and uncertain submissions block automatic retries.
- Existing ALFWorld and WebShop jobs, environments, models and old ScienceWorld
  files remain intact. The old cancelled baseline may remain COMPLETING on
  JDS9B74; other still-active old ScienceWorld jobs block this transition.

**46 targeted local tests passed**, covering the ledger transition, duplicate
submission prevention, partial quota retries, actual launch/config composition,
new checkpoint selection, paired protocol invariance, startup and monitoring.
Both changed batch scripts pass Bash parsing. These checks do not establish
native GPU execution. The user's later queue contains **166748_2/5** and
**166749_2/5**, plus **166750_2/166751_5**, all pending. Per the supplied
controller's submission order these are smoke, formal and continuations
respectively. Preserve them; no new execution logs have been returned.
After removing node exclusions as requested, **19 affected controller tests
passed again**, including assertions that submissions do not select or exclude
any node.
The node cleanup cause and Ray rank-zero registration failure remain unresolved;
the restart includes the previously added startup diagnostics.

## Earlier investigation: old ScienceWorld cleanup on JDS9B74

The user first requested diagnosing the persistent COMPLETING state. The
original recovery submitted **no new ScienceWorld jobs**; its attempts remained
blocked by **166362_2**, physical job **166572**, JDS9B74. The new approach above
supersedes retrying that recovery command.

At **12:16:15 +08**, 7m07s after the recorded EndTime of **12:09:08**:

- JDS9B74 reports **MIXED+COMPLETING**, CPULoad **602.63** against 112 effective
  CPUs, CPUAlloc=56, AllocMem=753664 MiB, FreeMem=712138 MiB, allocated GPUs=6.
  The allocation-statistic reductions from the earlier snapshot match 32 CPUs,
  256 GiB and two GPUs; this does not prove all job processes have terminated.
- Controller records still list **166362_2.extern**, **.batch** and **.0** as
  RUNNING. These are controller records, not a fresh process inspection.
- `sstat --allsteps --jobs=166572` prints **No steps running for job 166572**,
  returns zero, and provides no statistics. The matching
  [23.02.8 source](https://github.com/SchedMD/slurm/blob/slurm-23-02-8-1/src/sstat/sstat.c)
  can emit this message on allocation-lookup failure before querying node
  statistics, as well as on an empty step list. Do not infer successful node
  communication, absent processes or a proven epilog hang from this output.
- Configuration: KillWait=30s, MessageTimeout=10s, UnkillableStepTimeout=300s,
  SlurmdTimeout=300s, ProctrackType=proctrack/cgroup, node
  Epilog=/cm/local/apps/cmd/scripts/epilog, EpilogSlurmctld=null. These timers do
  not sum to a guaranteed cancellation deadline; epilog timeout was not queried.
- Training output still ends at startup configuration/resource-check warning.
  No termination traceback, D-state process, epilog execution trace or kernel
  filesystem/GPU error has been obtained. Earlier step launches to JDS9B74
  timed out, so node cleanup/communication warrants investigation without a
  confirmed root cause.

The user confirmed only a login-node terminal is available. Further decisive
evidence needs node-side job cgroup PIDs/states/wait channels, slurmstepd/slurmd
termination logs since 12:09, and whether this job's epilog began and finished.
Do not repeat failed diagnostic allocations, attach with elevated privileges,
kill unrelated jobs or restart shared node services. The latest queue has
repaired ALFWorld formal baseline **166629_0**, both ALFWorld smoke arms
**166613_0/3** and WebShop baseline smoke **166697_1** running; preserve them.

## Latest ScienceWorld evidence and startup diagnostics

Old ScienceWorld baseline smoke **166370_2**, physical attempt **166696-2**,
has now failed at c193e61. Ray started at 11:48:06; rank-zero registration waited
from 11:59:43 for 300 seconds and raised `RayTaskError(TimeoutError)` in
`RayWorkerGroup._init_with_resource_pool`. The named-actor snapshot contained
only `2yW4AcWorkerDict_0:0`, without its registration-center actor. The report
records **zero optimizer updates/actions and no checkpoint**. Checkpoint cleanup
marked deleted does not establish a successful checkpoint save. Warnings about
128 through 314 Python workers suggest startup pressure but do not establish
why rank zero failed to register. The later lock-close `OSError: [Errno 77]`
is secondary; neither an OOM nor an algorithm failure is established.

The user executed recovery controller **d89c79828be5f46f0262f8ba11bdb55cba26ffa4**.
It requested cancellation of **166514_2, 166517_5, 166362_2, 166362_5** and
stopped because **166362_2/5** were still exiting. **No replacement jobs were
submitted.** The controller did not request cancellation of either smoke in this
invocation. Preserve the existing retirement ledger under
`training-submissions/scienceworld-fdbf5569b82d9c1afd1ad2578c914adf00d0f417/`.
The isolated restart described above replaces this pending recovery request.
The new source improves startup observability; registration timeout is now a
known failure site, but its underlying cause has not been fixed or verified.

The latest returned queue confirms formal baseline **166362_2 at 1:01:51** and
full **166362_5 at 16:44**, both still `starting` at c193e61 with **no complete
checkpoint**. Old smoke **166370_2/5** is RUNNING at 14:43 on GDS9B74; resumes
**166514_2/166517_5** are pending. The attempt to read Ray logs in step
166362.2 timed out before launch; no Ray scheduler state was obtained.

Recovery now uses `recover_scienceworld.py`, targeting only those six IDs.
It validates ownership, original script/source and array index before cancelling
continuations, smoke and then formal parents. It distinguishes retained terminal
records from active/COMPLETING states, checks for optimizer/checkpoint evidence
before and after cancellation, and verifies the old writer locks are released.
It preserves all attempts/files and records the transition. Any new progress
blocks a source change. New submission uses the existing quota-safe journal:
two independent one-hour smoke and two 18-hour formal arms, plus one 18-hour
continuation per formal arm. All freeze training source **fdbf5569b82d9c1afd1ad2578c914adf00d0f417**,
which includes self-reported startup stages/stacks. No recovered job IDs or GPU
results have been returned yet; the original startup root cause remains unknown.

The same queue confirms WebShop smoke **166697_1/4**, formal **166698_1/4**,
resumes **166699_1/166700_4**, and the existing repaired ALFWorld jobs. Recovery
preserves them. **19 local controller/startup tests passed**, covering exact
cancellation scope/order, terminal retention, completing state, progress races,
writer locks, wrong source binding, quota retries and per-arm dependencies.

At **11:54:29 +08**, the diagnostic step on H64Y574 ran successfully. Full
TaskRunner **2268434** was alive (RNl, elapsed 08:13, CPU 1.5%, RSS 941192 KiB).
`py-spy` was denied permission; no stack was collected and no elevated access
will be attempted. The worker's own stdout shows **3322 train / 1661 test**
variations loaded, with no Python exception in the returned stderr. These are
catalog messages emitted before asynchronous actor initialization completes;
they do not demonstrate model loading, all simulator readiness, or an update.
The next requested evidence is this run's own Ray scheduler/debug logs.

The new source adds explicit TaskRunner startup stages (environment actor
creation, tokenizer/processor, worker imports, manifest/lock, Ray resources and
placement/registration, reference/actor model initialization). Stage records and
process-owned stack dumps go under each attempt's `startup/` directory. A slow
stage triggers a self-dump every 180 seconds using CPython's watchdog, requiring
no ptrace or elevated privileges. The watchdog is cancelled before `trainer.fit`
and on initialization failure; exception messages/locals/credentials are not
included. Initialization completion is not optimizer success. Existing frozen
jobs remain on their submitted source. This change improves observability; the
underlying ScienceWorld startup delay is not yet diagnosed or claimed fixed.

**48 distinct targeted local tests passed** across startup, training/full
protocol, ALFWorld import boundary, smoke and monitoring tests. The startup
fixture exercises CPython's real timed stack dump and cancellation in a child
process, and executes the production worker-init method with injected placement
and model failures. These checks do not emulate Ray scheduling or GPU training.

## Latest execution: duplicate WebShop goal identities

WebShop baseline attempt **166511-1**, source c193e61, exited before model/worker
initialization or optimizer updates. The first exception is
`Duplicate/overlapping concrete task identities in source pools` in
`task_manifest.split_catalog`. Complete generated goal dictionaries can repeat;
the previous code incorrectly assumed one unique content identity per list row.
The later `OSError: [Errno 77] File descriptor in bad state` occurred while the
launcher closed its lock after the failed child. Its filesystem cause is not
established; it is separate from the first, reproducible manifest failure.

The fix retains full-content goal identities and the original test identities.
It keeps the first native index per identity, excludes all training copies of
test goals, and deduplicates remaining train goals before the dev holdout. It
never changes the native goal array, merges merely equal text, salts IDs with an
index, or disables disjointness checks. Baseline and full restart catalogs share
this rule. The saved manifest includes raw/canonical counts, excluded occurrence
counts, cross-split counts, full input digest and duplicate source indices. This
is an explicit content-unique task-pool adaptation, not a claim of unchanged
native row-sampling weights. Actual HPC counts remain to be returned.

All **77 targeted tests passed**: 39 identity/catalog/replay/training/core tests,
plus 38 submission/resume/monitoring/smoke/full tests. A fixture executes the
original synthetic goal generator with duplicate product options to reproduce
the source failure mode; this does not establish which product rows duplicated
in the HPC data. No repaired GPU updates have been evidenced.

The paired submitter now supports one selected benchmark and optional smoke.
For WebShop it queues smoke `1,4%2` (one hour), formal `1,4%2` (18 hours), and one
18-hour `afterany:<new-formal>_<index>` continuation each. All use one repaired
source. There is no smoke-to-formal dependency. Accepted IDs survive quota
rejections; uncertain outcomes block blind retries.

Returned queues contain ALFWorld arrays 166613/166629 (indices 0,3) and
continuations 166630_0/166631_3; preserve them. Old WebShop 166370_1, 166370_4
and 166362_4 are confirmed **CANCELLED**; 166438_1/166516_4 had already left the
targeted remaining-ID query. The supplied retirement guard wrongly treated
terminal records retained by `squeue --states=all` as active. Direct submission
was then supplied. The latest queue confirms repaired WebShop formal pair
**166698_1/166698_4** pending; new smoke/resume IDs have not been returned.

ScienceWorld baseline **166362_2** was RUNNING at **46:58** on JDS9B74. The
diagnostic `srun` failed to launch step **166572.2** with a socket send/receive
timeout; no process status or stack was collected. This does not establish the
training root cause or prove a node-wide outage/OOM. Full **166362_5** was
RUNNING at **1:51** on H64Y574. Next evidence is current shared training
logs/progress for both arms and scheduler-reported node state, without another
remote diagnostic step. No ScienceWorld optimizer update has been evidenced.
A WebShop manifest failure is not evidence about ScienceWorld.

At **11:51:40 +08**, the returned shared-file/node inspection shows:
- JDS9B74 is MIXED, with **CPULoad 621.54**, 112 effective CPUs, 88 allocated
  CPUs and 712726 MiB node-wide free memory. This is evidence of high system
  load, not proof of its source, a GPU problem, or a per-job memory outcome.
- Baseline log mtime is **11:18:11**, over 33 minutes unchanged; progress is
  still `starting`. The last emitted warning comes from the resource check.
- Full log mtime is **11:46:24**, about five minutes unchanged; progress is
  `starting`. Its tail ends inside config printing, so it does not establish
  the same stopping point as baseline. TaskRunner PID is **2268434**, physical
  allocation **166362**, attempt **166362-5** on H64Y574.

The next bounded stack inspection targets the full run on H64Y574. Do not
attribute the load to this job or other users without process evidence, assume
node-wide free memory excludes a job-limit OOM, or change the training protocol
based only on the load snapshot. Existing submitted jobs/source bindings remain.

## Earlier execution: ALFWorld import boundary failure

The first returned ALFWorld baseline log (attempt 166509-0, training source
c193e61) fails in `make_envs` before optimizer updates or model initialization:
the learner imports `alfworld.info` through the vendored simulator package even
though `worker_python` selects the separate ALFWorld runtime. The source fix
defers the same factory import until the catalog or simulator actor calls it.
It does not change packages, algorithms, rewards or training shapes.

All 26 targeted import-boundary, smoke, bootstrap and training-protocol tests
passed, including real manager imports/worker serialization with simulator
imports blocked in the learner. The smoke batch script passes Bash syntax checks.
Actual fixed-source GPU execution remains unverified.

The latest user queue shows WebShop baseline 166362_1 running; formal indices
2–5 and all six smoke jobs pending. All replacement continuations now exist:
166437_0, 166438_1, 166514_2, 166515_3, 166516_4, 166517_5.
Recovery targets only pending ALFWorld jobs 166362_3, 166370_0, 166370_3,
166437_0 and 166515_3, followed by a fixed-source ALFWorld smoke pair (indices
0 and 3, independently scheduled). Withdrawal and new short-job IDs are still
pending. WebShop/ScienceWorld retain their running/queued c193e61 jobs.
Both ALFWorld arms must be rerun under source
`c6856146300b1d6a9f8706cc8a272016b133e400`. The user subsequently requested
queueing these two formal jobs and their corresponding continuations immediately,
without a smoke dependency. The paired submitter prepares indices 0,3 concurrently
and one 18-hour continuation after each parent. All four use the fixed training
source. Its journal preserves accepted IDs across quota retries and blocks
ambiguous retries; 15 targeted submission/resume/monitoring tests passed locally.
The subsequent queue contains the new ALFWorld arrays/continuations listed above.
No checkpoint is migrated; prior attempt evidence remains intact.

## Submission history

The user requested direct progress into training, without more standalone
preflight/continuation jobs. **Baseline and Full OMR training are now connected
for all three benchmarks.** The user reports that the earlier v1 baseline array
was submitted and subsequently paused by the user. The v2 array **166362** was
reported submitted with all six elements pending on testqueue. No HPC optimizer
update has been returned yet. The user returned successful submission receipts
for six dependent 18-hour continuations: **166364_0, 166365_1, 166366_2,
166367_3, 166368_4, 166369_5**, corresponding to parent indices 0–5.
These six continuations were subsequently cancelled at the user's request to
free submission slots for smoke tests. See `RESUME_CHAIN.md` for the historical
mapping. The 11 targeted resume-chain/monitoring fixtures passed locally;
no resumed GPU updates have been evidenced.

The user requested short real-training jobs while the formal array queues.
One-hour, two-update diagnostics for all six conditions are prepared, using the
same frozen c193e61 training source, full B16/G8/H shapes and separate outputs.
The user approved saving then deleting the smoke checkpoints after recording
the save evidence. Six targeted local fixtures passed. Smoke array **166370**
has now been submitted successfully with controller
`f130cfe90c1b5e22a6ee1960f73cfa82997813cc`; no smoke GPU execution results have
been returned yet. See `SHORT_TRAINING.md`.
The first smoke submission was rejected by Slurm with
`QOSMaxSubmitJobPerUserLimit`; it created no smoke job and ran no training code.
The user then explicitly requested cancelling the six continuation jobs
166364_0 through 166369_5 and submitting all six smoke conditions in their place.
The returned script output confirms the six continuation IDs left the queue
before successful submission of smoke array 166370. Formal array 166362 remains
authorized and unchanged. The old continuation IDs are historical receipts,
not an active continuation chain; do not duplicate the submitted smoke array.

The user now requests resubmission of the six cancelled continuations. The
explicit `resume_chain.py submit --replace-cancelled` path preserves the old
ledger, verifies cancellation records, and records new IDs separately. Definite
quota rejection can be retried without duplicating accepted jobs. The first
replacement attempt at c7e2f59 stopped before sbatch because sacct returned no
old-job records; no replacement job was submitted. The user already supplied
successful cancellation output, so the new explicit confirmed-ID option binds
that evidence to the original ledger and checks live queue absence across all
states. All 20 targeted resume/replacement/monitoring tests passed. The returned
c86ad2b submission output confirms **166437_0** (ALFWorld baseline) and
**166438_1** (WebShop baseline). Index 2 was initially rejected by the submit-job
quota; the subsequent queue shows 166514_2, 166515_3, 166516_4 and 166517_5,
confirming the remaining four submissions. The ALFWorld import failure and
targeted recovery above supersede the earlier all-pending status.

See `TRAINING.md` for actual submission/resume commands. Each array element uses
two H200 and trains one benchmark/arm for 150 updates; all six elements may run
concurrently (twelve H200 total), subject to Slurm allocation, and fail independently.
Fixed instance manifests, held-out validation, final three-seed
test, explicit same-root groups, strict ScienceWorld success, online SwanLab,
durable records and atomic full baseline checkpoint/resume are connected to the
existing BEACON optimizer loop. Full also connects group restore/fallback,
next-update archive admission, q/D and sampler state, and payload-safe resume.
See `FULL_OMR.md` for native contracts and the local/native evidence boundary.
Protocol v2 has common horizon/history/seed fixes: do not compare v1 baseline
results against v2 full, or silently migrate checkpoints between them.

## Completed locally

- Independent BEACON clone and experiment branch, pinned upstream provenance.
- Public GitHub origin and login-node Git clone/GPU-inventory submission script.
- Common milestone/action/timing/padding fixes described in `SOURCE_AUDIT.md`.
- Standalone archive, statistics, sampler, controller and payload store.
- Standalone atomic latest-only checkpoint transaction with failure tests.
- Single-writer SwanLab integration and explicit failed-run shutdown.
- Local suite before targeted preparation: 35 passed, 1 skipped (Windows cannot create the symlink fixture).
  Source compilation and submission-script syntax checks passed. The follow-up
  job reruns these fixtures in the existing HPC stack; GPU acceptance is separate.

## Returned GPU evidence and next dependency

The user returned preflight job **165837**, using frozen source `f7ccbe7`:

- ALFWorld and ScienceWorld source imports, GPU math and FlashAttention passed.
- Native ALFWorld reset/action replay and ScienceWorld reset/action replay passed.
  ALFWorld has 3,553 eligible train gamefiles with matching milestone identities.
- Two-process NCCL passed in the ScienceWorld runtime; all 36 core fixtures passed.
- The existing Qwen2.5-1.5B-Instruct tokenizer loaded. Full model revision/hash and
  weight loading still need acceptance; no replacement model download is needed.
- WebShop dependency checks failed, and the bounded asset scan found no complete
  product/attribute/human-attribute/index binding. Native interaction has not run.

The experiment scope is **ALFWorld, WebShop and ScienceWorld only**. At the user's
request, AppWorld installations are now excluded from inventory, candidate reuse,
WebShop environment discovery and probe execution. Old reports are retained as
historical evidence, not as a request to repair or test AppWorld.

The WebShop preparation used the already validated ScienceWorld stack as its
dependency base. The committed
WebShop profile uses `use_small=true` (1,000 products); this is the upstream profile,
not an instruction to download the full catalog. Preparation and checks run on GPU
compute nodes. Preserve the existing ALFWorld/ScienceWorld evidence without rerunning
it merely because the candidate list changed. See `SITE_PREPARATION.md` for limits.
The assistant has no Orion connection; the user executes remote jobs via Tunnel.

The targeted WebShop preparation is now implemented; see `WEBSHOP_PREPARATION.md`.
It creates an independent venv reusing the ScienceWorld stack, downloads only pinned
small-catalog/NLP assets, builds and checks Lucene, tests purchase/reset receipts,
checks persistent Ray source imports, verifies/loads existing Qwen weights and
exercises SwanLab offline logging.

Latest returned preparation **165873**, source `b019cfc`, completed with
`prepared_and_tested` and no error. Native WebShop interaction, Ray imports,
existing pinned model weight loading with BF16/FlashAttention, and real SwanLab
offline logging passed. The stage list confirms that the v3 environment and index
were reused. Online SwanLab remains unverified. This is preparation acceptance,
not an optimizer update or a complete training-readiness result.

The superseded standalone continuation job would check WebShop mid-episode replay and online SwanLab
upload/readback without installing packages, downloading assets, building an index
or reloading the policy model. The replay adapter uses explicit content-bound tasks,
executes the real command prefix and verifies every transition plus final session,
browser, detector, native RNG and manager input history. A fresh rollout receives
15 new actions. Its local tests include recursive prefixes, independent state,
corruption, live divergence and terminal exclusion. See `CONTINUATION_CHECKS.md`.
Those APIs are now connected to Full OMR group planning/archive admission;
native training evidence for the integrated path is still pending.

Earlier preparation **165860**, source `71f4083`, passed package
installation, preserved-package checks, `pip check`, WebShop dependencies, HF
catalog preparation, Lucene build/catalog checks, native search/click/purchase/reset
and Ray source provenance. It stopped in the monitoring adapter: SwanLab 0.10.2's
`run.url` property raises in offline mode, even when accessed with `getattr` and a
default. The existing-model hash/load check comes after monitoring and has not run.

The adapter now reads cloud URL properties only in online mode. Offline attempts
retain their run IDs and store `url: null`; logging and explicit success/failure
shutdown are still required. Five monitoring regression tests pass, including
an offline URL property that throws and an online URL retention check. The exact
preparation monitoring function also passes with real SwanLab 0.10.2 in an isolated
local Windows environment: both normal and intentional-failure attempts initialize,
write metrics, retain run IDs with null URLs, and finish with the expected status.
This is an offline SDK check, not GPU or online acceptance. Runtime requirements
and both index-provenance source files are unchanged, so the retry
reuses the prepared v3 environment, pinned assets and existing index. GPU retry
165873 passed; online monitoring verification remains pending.

Returned preparation job **165846**, source `6546d98`, stopped at dependency
resolution with a BLIS/Cython source-build error. Java/base checks, creation of the
isolated v1 venv and retrieval/verification of the English model wheel completed;
dependency installation, dataset/index preparation and subsequent native checks
did not run. False acceptance flags in this report mean not yet verified, not
additional simulator failures. The fix pins the spaCy/thinc/BLIS wheel combination
and requires binary packages except pure-Python Gym.

Returned retry **165848**, source `3000f2f`, again stopped during dependency
resolution. It now identifies the conflict: smart-open 8.0.3 is pinned in the base,
but weasel 0.4.1–0.4.3 require smart-open below 8; older weasel releases reject the
base's Typer 0.27.3. The user supplied all 216 base package versions. The next recipe
pins weasel 0.4.3 and allows only smart-open 7.7.1 to shadow its base counterpart
inside a small v3 venv. It checks installed dependents before installation, installs
only approved hashed artifacts without uninstalling base files, and verifies reused
package versions/locations afterward. Base packages, cache and HF assets are reused;
v1/v2 markers remain intact. Its GPU result is recorded above. No formal training or
optimizer update has run.

Before this fix, local checks covered 45 fixtures (44 passed; the Windows symlink fixture is
skipped), including persistent child-runtime isolation, rejection of dependency
replacement, asset hash failures, native terminal/reset behavior and SDK naming.
All 12 targeted preparation fixtures now pass, including a real offline pip
installation into an isolated fixture venv that preserves every base file, exact
overlay allowlisting, dependent checks and post-install provenance checks.
The earlier no-install pip resolution for CPython 3.12 / manylinux x86-64 selected
42 wheels for the pinned spaCy/thinc/BLIS stack with NumPy 1.26.4; the real pip
requirements parser also confirms binary-only BLIS/thinc and the Gym source
exception. That check covered the NLP wheel combination, not the installed HPC constraints.
A new no-install resolution uses the 216 returned HPC version constraints (with
the one explicit smart-open exception) and the full WebShop wheel requirements.
It resolved 99 wheels; smart-open 8.0.3 to 7.7.1 is the only changed base version.
The projected new-artifact list passes the same version/hash guards used by the
preparer. This cross-platform check excludes unchanged source-only Gym and does
not emulate Linux dependency markers or replace native GPU-node acceptance.
Submission scripts pass Bash syntax checking. These are not HPC acceptance results.

## Remaining execution / acceptance

1. Obtain the already submitted v1 baseline array ID/status. Preserve its results;
   it is not the v2 formal comparator. Do not cancel unidentified jobs.
2. Submit the six v2 baseline/full conditions from one frozen source revision
   through the user's Tunnel session, independently and concurrently.
3. Obtain actual native replay, two-H200 optimizer, weight synchronization, resume and SwanLab
   evidence from training. Local fixtures do not establish GPU execution success.
4. Collect the six learning curves, final evaluation tables and mechanism plots.
   Runtime failures or no improvement must be reported from measured results.

| Environment | Baseline | Full | Job ID | Training updates | SwanLab URL | Latest model |
| --- | --- | --- | --- | ---: | --- | --- |
| ALFWorld | v1 submitted (user report); v2 not submitted | v2 not submitted | Pending | Unknown for v1 | Unknown | Unknown |
| ScienceWorld | v1 submitted (user report); v2 not submitted | v2 not submitted | Pending | Unknown for v1 | Unknown | Unknown |
| WebShop | v1 submitted (user report); v2 not submitted | v2 not submitted | Pending | Unknown for v1 | Unknown | Unknown |

There are no measured learning curves, success rates, GPU-hours, or model outputs
yet. New-run/resume commands are in `TRAINING.md`; upstream example launchers are
not the OMR execution protocol. Full remains unlaunched.

## Local checks

```bash
python -m unittest discover -s experiments/omr/tests -v
python -m compileall -q agent_system/milestone_reuse experiments/omr migpo
```

Tests require NumPy/Torch/OmegaConf for numerical and composition fixtures; simulator/Ray imports are
not required. The inventory is standard-library-only and runs on the GPU node.

After the continuation changes, the complete local suite ran 62 tests: 61 passed
and the Windows symlink fixture was skipped. Both new submission scripts pass
Bash syntax checks. No returned GPU continuation or online readback result yet.

After baseline integration, the local suite ran **74 tests: 73 passed, one Windows
symlink fixture skipped**. New coverage includes real OmegaConf composition for
all profiles, disjoint stratified manifests, deterministic resumed task schedules,
native terminal freezing, strict/reference ScienceWorld separation, evaluation
tail denominators, and the production checkpoint subclass with simulated rank
failure plus Python/NumPy/Torch/dataloader state recovery. Training submission
scripts pass Bash parsing. These fixtures do not simulate a native GPU update.

After Full OMR integration, **89 local tests ran: 88 passed and the Windows
symlink fixture was skipped**. New coverage exercises all three native-interface
restart paths, the production full observer and manager, next-update admission,
whole-group fallback/quarantine, dynamic event denominators, prefix exclusion,
full archive/model checkpoint rollback, WebShop history preservation, and native
worker policy/weight-sync receipt guards. Both training scripts passed Bash syntax
checks. No Full OMR GPU execution or final experimental result is claimed.
