# Execution status — 2026-10-18

The user requested direct progress into training, without more standalone
preflight/continuation jobs. **The three baseline training entry points are now
implemented.** Full OMR remains under integration and is explicitly rejected by
the baseline launcher. No HPC optimizer update has been returned yet.

See `TRAINING.md` for actual submission/resume commands. Each array element uses
two H200 and trains one benchmark for 150 updates; elements run one at a time and
fail independently. Fixed instance manifests, held-out validation, final three-seed
test, explicit same-root groups, strict ScienceWorld success, online SwanLab,
durable records and atomic full baseline checkpoint/resume are connected to the
existing BEACON optimizer loop. The previous continuation job is superseded.

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
These APIs are not yet connected to trainer group planning/archive admission.

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

## Remaining implementation / acceptance

1. User submits the three baseline training jobs through Tunnel; handle any actual
   training error per benchmark without blocking the other two.
2. Complete native ALFWorld/ScienceWorld restart adapters and connect all three
   adapters to group planning, admission and controller commit after optimizer updates.
3. Extend the integrated baseline checkpoint to full archive/statistics/payload
   dependencies and full-only scientific records. Do not relabel baseline as full.
4. Obtain actual two-H200 optimizer, weight synchronization, resume and SwanLab
   evidence from training. Local fixtures do not establish GPU execution success.

| Environment | Baseline | Full | Job ID | Training updates | SwanLab URL | Latest model |
| --- | --- | --- | --- | ---: | --- | --- |
| ALFWorld | Not run | Not run | None | 0 | None | None |
| ScienceWorld | Not run | Not run | None | 0 | None | None |
| WebShop | Not run | Not run | None | 0 | None | None |

There are no measured learning curves, success rates, GPU-hours, or model outputs
yet. New-run/resume commands are in `TRAINING.md`; upstream example launchers are
not the OMR execution protocol. Full remains unlaunched.

## Local checks

```bash
python -m unittest discover -s experiments/omr/tests -v
python -m compileall -q agent_system/milestone_reuse experiments/omr migpo
```

Tests require NumPy/Torch for numerical BEACON fixtures; simulator/Ray imports are
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
