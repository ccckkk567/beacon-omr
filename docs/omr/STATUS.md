# Execution status — 2026-10-18

The OMR implementation is in progress. **Do not launch the six formal runs yet.**

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

The next preparation targets missing WebShop dependencies and its HF-hosted assets,
using the already validated ScienceWorld stack as the dependency base. The committed
WebShop profile uses `use_small=true` (1,000 products); this is the upstream profile,
not an instruction to download the full catalog. Preparation and checks run on GPU
compute nodes. Preserve the existing ALFWorld/ScienceWorld evidence without rerunning
it merely because the candidate list changed. See `SITE_PREPARATION.md` for limits.
The assistant has no Orion connection; the user executes remote jobs via Tunnel.

The targeted WebShop preparation is now implemented; see `WEBSHOP_PREPARATION.md`.
It creates an independent venv reusing the ScienceWorld stack, downloads only pinned
small-catalog/NLP assets, builds and checks Lucene, tests purchase/reset receipts,
checks persistent Ray source imports, verifies/loads existing Qwen weights and
exercises SwanLab offline logging. Its HPC result has not yet been returned.
Local checks now cover 45 fixtures (44 passed; the Windows symlink fixture is
skipped), including persistent child-runtime isolation, rejection of dependency
replacement, asset hash failures, native terminal/reset behavior and SDK naming.
Submission scripts pass Bash syntax checking. These are not HPC acceptance results.

## Remaining implementation / acceptance

1. Select compatible existing environment/model/data paths; prepare missing assets
   through Hugging Face and configure dependencies on GPU compute nodes only.
2. Implement native per-environment restart adapters and explicit task manifests;
   integrate same-start groups, history/detector state and H new actions.
3. Connect controller commits, complete records, cost ledger, atomic checkpoint and
   full resume to the native trainer. No replacement PPO implementation.
4. Verify real environment interactions, strict success, milestone-to-advantage,
   two-H200 optimizer/rollout synchronization, checkpoint interruption/resume, and
   SwanLab online metrics or verified offline sync.
5. Freeze configs/commit, then submit ready environments' baseline/full pairs.

| Environment | Baseline | Full | Job ID | Training updates | SwanLab URL | Latest model |
| --- | --- | --- | --- | ---: | --- | --- |
| ALFWorld | Not run | Not run | None | 0 | None | None |
| ScienceWorld | Not run | Not run | None | 0 | None | None |
| WebShop | Not run | Not run | None | 0 | None | None |

There are no measured learning curves, success rates, GPU-hours, or model outputs
yet. Formal new-run/resume/evaluation commands will be delivered with the verified
integration; upstream example launchers are not the OMR execution protocol.

## Local checks

```bash
python -m unittest discover -s experiments/omr/tests -v
python -m compileall -q agent_system/milestone_reuse experiments/omr migpo
```

Tests require NumPy/Torch for numerical BEACON fixtures; simulator/Ray imports are
not required. The inventory is standard-library-only and runs on the GPU node.
