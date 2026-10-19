# OMR source audit — 2026-10-18

Upstream: https://github.com/ZJU-REAL/BEACON

Pinned source: `c88af45736b0a178106bc14ea89603ff02d8b222`.
The supplied ZIP independently matches SHA256
`aed4d0b2eb6267a5355609d0ee652407e283b6c4294c4452936490b489badca7`.
This checkout was cloned from Git, not imported from the ZIP.

## Common-base changes implemented

| Finding | Source and change | Evidence / remaining boundary |
| --- | --- | --- |
| ScienceWorld milestone lost before MiGPO | `agent_system/multi_turn_rollout/rollout_loop.py:388` explicitly propagates `subgoal_completed`, pre/post score | Numerical dynamic-event fixture passes; real collector-to-GPU update still pending |
| ALF missing executed actions / trial ID | `env_manager.py:143,166`; collector `rollout_loop.py:355` records commands separately from response tokens | Parser matches the actual `migpo/alfworld.json` key shape; live gamefiles not yet inspected |
| NumPy truth-value error | `verl/trainer/ppo/ray_trainer.py:374` uses explicit None fallback | ScienceWorld and WebShop numeric fixtures use the same event route |
| ScienceWorld shared split mutation | `sciworld/envs.py:207` copies kwargs before split extraction | Explicit train/dev/test manifests and actual load receipts still require integration |
| Reset score incorrectly zero | `sciworld/envs.py:157` uses reset score as `prev_score`; preserves reference reward formula | Strict completion predicate must be established from installed JAR/package |
| WebShop worker-specific goals | `webshop/envs.py:40` separates fixed manifest seed from worker seed, seeds complete price/goal construction, records actual goal content identity | Need live all-worker digest and actual split overlap checks; integer indexes alone are not acceptance evidence |
| Repeated complete WebShop goals | `webshop_goal_catalog` keeps one canonical index per full-content identity; native test identities take precedence over train copies; training and restart adapters share the pools and record duplicate indices/counts | The returned 166511-1 failure exposed the previous uniqueness assumption. Five new fixtures cover exact duplicates, original goal generation, canonical targets, worker permutation mismatch and replay identity. Repaired native counts/updates are pending |
| WebShop terminal session lost | Native text step preserves a copied purchase receipt and defers reset; explicit reset discards the previous mutable session | Native search/click/purchase/reset passed in GPU jobs 165860 and 165873 |
| WebShop assets tied to checkout | Environment construction accepts explicit shared product/attribute/human-attribute/index paths | Small-catalog data and one Lucene index prepared outside Git; reuse passed in GPU job 165873 |
| Physical batch reorder changed segments | `migpo/core_migpo.py:38` sorts by local step, precomputes segment IDs, passes them into advantages | Exact numerical permutation fixtures pass |
| Padding altered group evidence | `multi_turn_rollout/utils.py:100,129` marks copies; MiGPO excludes copies, trainer/actor masks their policy/KL/entropy contributions | Numerical advantage fixture passes; distributed optimizer equivalence remains a GPU acceptance item |
| Restarts could re-award prefix milestones | `match_milestones` accepts initial offset; reward and advantage share precomputed segments | Offset fixture passes; real restart adapters are not integrated yet |
| Destructor could mark failed jobs successful | `verl/utils/tracking.py:117`, `main_ppo.py` explicitly finish with real exit status | SwanLab simulated SDK fixture passes; installed SDK and remote metric read-back pending |
| WebShop debug file collisions | `migpo/webshop_milestone_detector.py:98` uses process + UUID; run log directory can be supplied | Detection logic unchanged; verbose volume tuning pending |
| Ray overrode thread controls | `main_ppo.py` passes one OMP/MKL thread and disables tokenizer parallelism | Actual Ray worker resource validation pending |
| WebShop lacks explicit continuing-state restoration | Opt-in worker task/replay API verifies native state, RNG, detector and manager input history; ordinary reset/step remains available | Local replay tests pass; native continuation GPU probe and trainer integration pending |

## Algorithm components implemented, not yet connected to the rollout loop

`agent_system/milestone_reuse/` contains immutable task/start identities, a bounded
archive, protected first-trial admission, decayed q/r/L/D statistics, independent
restart RNG, frozen-view priority/staleness/coverage sampling with reservations,
update-boundary commits, and content-addressed restore-payload persistence.

Core fixtures cover per-checkpoint L normalization, unknown-label handling,
single lazy decay, no source-tail evidence inheritance, concurrent discovery
order independence, actual marginal selection probabilities, first trials through
any channel, next-update activation, and serialization/RNG continuity.

## Checkpoint and monitoring status

`experiments/omr/latest_checkpoint.py` implements and tests an owned, locked,
same-filesystem generation transaction: write → validate → fsync → commit pointer
→ remove old generation. Injected write failure leaves the previous latest usable.
It verifies file hashes and keeps archive payload collection separate.

**This coordinator is not yet wired into RayPPOTrainer._save_checkpoint/_load_checkpoint.**
Do not use upstream `max_actor_ckpt_to_keep=1` as a substitute. FSDP rank files,
loader state, algorithm state, task sampler state and run lineage must be committed
together before formal training is enabled.

The existing Tracking integration now initializes one SwanLab writer, stores local
metrics before upload, redacts credential fields, records transport status, and
finishes explicitly. The rollback decision helper starts a new attempt when logs
are ahead of the model. Its integration with model resume is still pending.
SDK behavior was checked against official init/resume documentation and current
upstream Run.finish; the installed HPC version remains authoritative:
- https://docs.swanlab.cn/api/py-init.html
- https://docs.swanlab.cn/guide_cloud/experiment_track/resume-experiment.html
- https://github.com/SwanHubX/SwanLab/blob/main/swanlab/sdk/internal/run/__init__.py

## Initial audit gaps (historical; current resolution below)

- The ALF TextWorld limit and ScienceWorld `env_step_limit` still need a separate
  replay-prefix allowance plus H new actions without resetting native world time.
- Existing WebShop rendering has a long-text fallback that drops history. Record
  the actual renderer behavior and resolve the common input profile before formal
  runs; do not silently change only full.
- Actual task manifests, explicit parent task planning, group-wide restore/fallback,
  initial detector/history restoration and strict evaluation are not implemented.
- Existing evaluation grouping uses training G in the collector. Formal evaluation
  needs root-only, fixed-manifest G=1 and complete multi-batch coverage.
- Complete scientific step/rollout/group records, actual cost ledger, final test
  seeds, real optimizer/rollout weight synchronization and signal-stop integration
  remain to be wired and verified.

## Acceptance boundary

24 local unit/numerical fixtures passed with Python 3.12 and Torch 2.10 CPU tensor
execution on Windows. This is **not** the training dependency profile and is not
evidence of Orion simulator/GPU readiness. The user has since returned Slurm
inventory job 165833, confirming two H200 devices are visible through existing
Torch installations; see `SITE_PREPARATION.md`. Actual environment interaction,
training updates, SwanLab uploads and model checkpoints remain unverified.

## Protocol v2 integration

The historical integration gaps above are now connected in the trainer. The
baseline/full arms share explicit manifests, root-only evaluation, terminal
freezing, complete interaction/cost logs, native optimizer/rollout version
receipts and the latest-only checkpoint transaction. Full adds independent native
replay, whole-group fallback, milestone-offset/history restoration, the frozen
sampler view and post-optimizer q/D/archive commit. See `FULL_OMR.md`.

Common v2 changes are deliberate and require a fresh matched pair: artificial
ALF/ScienceWorld limits no longer terminate an H-step collector early; ALF loads
the explicit game with its explicit seed; WebShop retains history=2 above the
old character cutoff and obeys the common token-overflow error. Old v1 baseline
results cannot be treated as the v2 comparator. The actual reward equations and
per-benchmark optimizer profiles are unchanged.

ScienceWorld replay checks native task/variation, score/completion and legal-action
receipts, preserving the original API order; observation wording differences
are recorded and actual replay observations rebuild history. It is a bounded
contract, not an assertion that all hidden Java state has been snapshotted.
Native correctness and two-GPU execution still require returned run evidence.
