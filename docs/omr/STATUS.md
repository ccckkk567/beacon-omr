# Execution status — 2026-10-18

The OMR implementation is in progress. **Do not launch the six formal runs yet.**

## Completed locally

- Independent BEACON clone and experiment branch, pinned upstream provenance.
- Public GitHub origin and login-node Git clone/GPU-inventory submission script.
- Common milestone/action/timing/padding fixes described in `SOURCE_AUDIT.md`.
- Standalone archive, statistics, sampler, controller and payload store.
- Standalone atomic latest-only checkpoint transaction with failure tests.
- Single-writer SwanLab integration and explicit failed-run shutdown.
- Local suite: 30 passed, 1 skipped (Windows cannot create the symlink fixture).
  Source compilation and submission-script syntax checks passed. The follow-up
  job reruns these fixtures in the existing HPC stack; GPU acceptance is separate.

## Current external dependency

The user returned inventory job 165833: two H200 GPUs are visible and existing
ScienceWorld/AppWorld dependency stacks are candidates for reuse. See
`SITE_PREPARATION.md` for evidence and limits. The assistant has no Orion connection.

The submission helper now clones/checks the small code repository on the login
node and submits inventory directly to `testqueue`, without an `ioqueue` job or
CPU-job dependency. Environment imports and GPU checks still run inside Slurm.

Environment setup/reuse decisions require the real GPU-node package/import paths,
model assets, Java/JAR and available storage. Inventory does not install packages,
load a large model, read credential values, or modify old experiments.

The next command submits `preflight.sbatch` for symlink-aware asset discovery,
isolated package/source imports, GPU kernels and NCCL, and bounded test-only native
simulator interactions. Its returned report is needed before finalizing reuse.

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
