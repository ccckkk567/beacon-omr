# Execution status — 2026-10-19

## Latest execution: ALFWorld import boundary failure

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
Both ALFWorld arms must be rerun under the same new source. Their formal
replacements and continuation dependencies will be arranged after smoke output;
no existing checkpoint is migrated and no prior training result is overwritten.

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
