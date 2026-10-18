# Site preparation evidence — 2026-10-18

The user returned GPU inventory job **165833**, run from commit `3510627` on
`testqueue`. The raw report is retained privately, outside version control. This
is inventory evidence, not permission to launch the formal six runs.

## Confirmed

- Two NVIDIA H200 GPUs were allocated, with driver 580.126.20. Existing ALFWorld,
  ScienceWorld and AppWorld environments see both devices through Torch.
- The main and overflow locations are writable. The reported main filesystem has
  approximately 737 GiB free; user/project quota remains unverified.
- A credential file exists and is readable; its contents were not read by the
  inventory or copied into this repository.
- Java 11.0.21 is available. ScienceWorld 1.2.3 has a bundled JAR with SHA256
  `f4e1fdce0cdabe1bab7128a7e78ea820210d9db3c28d7272bcbbc3f3a1c5f0d7`.

| Existing environment | Relevant installed packages | Decision before runtime checks |
| --- | --- | --- |
| `gigpo-scienceworld-py312-v1` | Python 3.12.5, Torch 2.6.0/CUDA 12.4, vLLM 0.8.2, Transformers 4.51.1, TensorDict 0.6.2, ScienceWorld 1.2.3, SwanLab 0.10.2 | First candidate for ScienceWorld package reuse; do not downgrade solely because README mentions Python 3.10 |
| `gigpo-appworld-py312-v1` | Python 3.12.5, Torch 2.6.0/CUDA 12.4, vLLM 0.8.5, Transformers 4.51.1, TensorDict 0.6.2 | Candidate dependency base; ALFWorld/ScienceWorld packages absent |
| `gigpo-alfworld-py312-v1` | Python 3.12.5, Torch 2.8.0/CUDA 12.8, vLLM 0.11.0, Transformers 4.57.3, TensorDict 0.10.0, ALFWorld 0.4.2/TextWorld 1.7.0 | Reuse simulator assets if validated; policy stack differs from BEACON's declared constraints |
| `dep-prep-py312-v1` | Actual Python 3.14.6, no Torch | Name is not evidence of Python version or readiness |

The three ML environments passed their existing `pip check`. This checks installed
distribution metadata; it does not establish compatibility with the new checkout.
No environment in the first inventory reports Pyserini or legacy Gym. This does
not rule out a virtual environment inside an old project; the follow-up now scans
bounded `envs` and `code` paths and checks those interpreters explicitly.

## Changes motivated by this report

The inventory showed `agent_system` namespace paths from both this checkout and
old projects. A regular `agent_system/__init__.py` now prevents that namespace
merge. The next diagnostic also runs each selected interpreter with `-I -S -B`,
explicit site-packages directories and source-origin checks, without executing
old editable-install `.pth` hooks. This mechanism is diagnostic only: inheritance
into persistent Ray workers is still an acceptance item.

ScienceWorld's vector wrapper only uses the `Env` base class. It now imports that
class from the already installed Gymnasium, without adding a legacy Gym package
or changing native ScienceWorld step/reset signatures.

The original bounded asset scan did not follow directory symlinks and returned
no matching asset metadata despite finding data/cache directories. This does not
prove assets are absent. `assets.py` now follows directory aliases, detects cycles,
and records depth/entry/time limits and errors. It reports config/tokenizer/shard
presence, ALF logic and gamefile counts, milestone key coverage, and WebShop files
and index directories. Counts are discovery evidence, not formal task manifests.

## Next GPU job

`submit_preflight.sh` submits one two-H200 job to `testqueue` using the previous
inventory JSON as input. It does not install packages, download models, read keys,
or activate/edit the existing environments.

The job attempts actual BEACON imports, BF16 matrix multiplication and
FlashAttention on each GPU, and a two-process NCCL all-reduce. It also attempts
ScienceWorld reset/step/replay and ALFWorld independent reset/replay on one
test-only instance when an eligible data root is found. It records the installed
ScienceWorld step source, native completion status, score and JVM cleanup. No gold
actions are requested. An available Qwen tokenizer is read offline as a loader
check; this does not select or certify the required model revision.

WebShop is now included explicitly. Every discovered interpreter gets real
dependency imports, including Gym, Pyserini/JNI, spaCy's `en_core_web_sm`, and the
vendored simulator. A complete adjacent product/attribute/human-attribute/index
layout is required before interaction. Optional explicit native asset paths allow
reuse without copying files or changing the old checkout. In a passing runtime,
two independent servers execute reset/search/product-click with different worker
seeds and the same manifest seed; native session transitions, goal identity and
manifest hashes are checked. These are test-only dataset-query fixtures, not
training or evaluation policy trajectories. Missing dependencies/assets produce
explicit blockers; skipped interaction is never reported as passed.

The default submission checks all three environments. Pass `webshop` as the fourth
argument to `submit_preflight.sh` for a supplement if the earlier checks already
ran. Wait for a running earlier job to finish before pulling into its checkout.

## Checkout normalization and frozen submission

The first submission stopped at its clean-checkout check. Four reported modified
Search-R1 files have CRLF Git blobs despite the repository's `*.py eol=lf` rule.
A fresh local checkout reproduced these modifications, plus `docs/conf.py`, without
any user edits. All five blobs are now normalized to LF; their bytes otherwise
remain unchanged. The user's original HPC modifications are not discarded or
assumed to be exclusively line-ending changes.

`submit_frozen_preflight.sh` fetches a specified published commit and creates a
separate detached worktree under `code/beacon-omr-runs/`. The original checkout's
working files and branch remain intact. The ordinary submission helper validates
the frozen revision and clean state in that new directory before submitting all
three environments' checks. Worktrees use the same Git object store and existing
external model/data paths; models and datasets are not copied.

The local suite has 35 passing tests and one symlink fixture skipped because the
Windows process cannot create that link. The follow-up job also reruns the core
fixtures in the existing HPC dependency stack, including that symlink test.

Failed checks remain failed in the JSON; simulator fixture results do not enter
training archives. Local tests do not establish that this GPU job has passed.
Formal readiness still requires returned native interaction evidence, exact model
provenance, complete task manifests, integrated restart adapters, actual policy updates,
checkpoint/resume, and SwanLab upload/readback.

Implementation references: [Python command-line isolation](https://docs.python.org/3/using/cmdline.html)
and [ScienceWorld Python interface](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py).
Installed source and the pinned JAR remain authoritative for runtime acceptance.
