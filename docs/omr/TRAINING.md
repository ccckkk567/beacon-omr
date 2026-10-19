# Three-benchmark baseline / Full OMR training

The user requested actual training instead of further standalone checks. The
launcher supports **baseline and Full OMR** on ALFWorld, WebShop and ScienceWorld.
Both arms use the same frozen source. Native GPU training and learning curves
remain unverified until actual job logs are returned.

## Current ScienceWorld restart

The old baseline 166362_2 remains COMPLETING on JDS9B74. The user authorized
an independent-output restart, preserving its files and the existing ALFWorld
and WebShop jobs. Use `experiments/omr/isolate_scienceworld.py --root "$ROOT"
--revision "$REV"` from the clean frozen checkout supplied in the chat. This
controller creates the matching training checkout, backs up and supersedes the
unsubmitted d89c798 recovery request under its original lock, then submits:

| Conditions | Time per job | Dependency |
| --- | --- | --- |
| ScienceWorld baseline/full smoke | 1 hour | None; two may run together |
| ScienceWorld baseline/full formal | 18 hours | None; two may run together |
| One continuation per formal arm | 18 hours | afterany of its own new parent |

Each job retains two H200, 32 CPUs and 256 GiB on testqueue. There is no node
exclusion or node selection constraint; Slurm may also allocate JDS9B74.
Formal paths and SwanLab experiment names have suffix `-restart-166362`; both
arms share `manifests/restart-166362/scienceworld.json`. Model, environment,
seed, batch, reward and method settings are unchanged. Smoke retains its own
array-specific paths and deletes only its owned checkpoint after saving and
recording the result. Formal checkpoints are retained. Continuations resume
only the matching new run and preserve its monitoring lineage; no complete
checkpoint means no restart from update zero.

On a quota rejection, rerun the same command: accepted IDs remain in
`training-submissions/scienceworld-restart-166362/jobs.json`. An uncertain
submission response blocks blind retries. Do not rerun the superseded d89c798
recovery command or submit another generic array. GPU acceptance and the cause
of the earlier worker registration timeout are still pending.

## Original six-condition submission (historical)

Use the assigned work root, accepted preparation report `webshop-prepare-165873/report.json`
and the full published revision supplied in the chat:

```bash
bash "$ROOT/code/beacon-omr/experiments/omr/submit_training.sh" \
  "$ROOT" "$ROOT/evidence/beacon-omr/webshop-prepare-165873/report.json" "$REV" all both
```

For one additional 18-hour continuation of each already submitted array element,
use `RESUME_CHAIN.md`. Keep its training revision identical to the parent; the
newer orchestration checkout does not replace the frozen training code.

The script freezes a clean detached source checkout without touching the four
modified files in the user's existing checkout. It submits array `0-5%6`:
0/1/2 are ALFWorld/WebShop/ScienceWorld baseline; 3/4/5 are the corresponding full
arms. Each element uses two H200, 32 CPUs, 256 GiB RAM and up to 18 hours on
testqueue. All six may run concurrently, using twelve H200 in total when
resources are available; Slurm queues elements as needed.
There is no dependency or prescribed start order between benchmarks. Failure of
one element does not cancel the others. These are training jobs, with no separate
environment, model, index or transport probe beforehand.

To submit/resume one condition, use `alfworld`, `webshop` or `scienceworld` as the
fourth argument and `baseline` or `full` as the fifth, with the **same revision**.
`all baseline` and `all full` submit three independent jobs. The job locks its
logical run, verifies the complete latest checkpoint and continues the exact task
schedule. If source/protocol/manifest changed, it refuses an implicit migration.
An already completed latest run does not perform extra optimizer updates.

**Protocol v2 is a new comparison.** The user already submitted the v1 baseline
array from `7a982c1`; its ID/status have not yet been returned. Preserve those jobs
and results. v2 makes common fixes to native artificial limits, explicit ALF root
seeding and WebShop history rendering, and adds native weight-sync receipts.
Do not compare a v1 baseline with v2 full or migrate its model silently. Run the
v2 baseline/full pair from the same commit; old job cancellation requires its
exact identified array ID, never a blanket cancellation by user or partition.

Slurm requests a stop 20 minutes before the time limit. The trainer saves after a
complete update and can interrupt evaluation between batches. To request an
earlier boundary stop, create `STOP` inside the active attempt directory printed
at startup. Do not cancel unrelated jobs or delete old experiment directories.

## Protocol

- Seed 0, Qwen2.5-1.5B-Instruct, 150 native trainer updates, 16 task groups x 8
  trajectories. H is 30/15/30 for ALF/Web/Sci. Frozen method-specific profiles
  remain in `experiments/omr/configs` and the resolved config is saved per attempt.
- Native instance manifests are frozen under `analysis/beacon-omr/manifests` and
  shared by both arms under a creation lock. The upstream entry points have train/test pools;
  a proportional, deterministic `min(128, floor(train_size/10))` validation holdout
  is removed from train. Final test is a prespecified stratified maximum of 300.
  ALF test remains **valid_seen**, not unseen. Web uses canonical goal identities.
- Training retains ALF's per-group shuffled-cycle distribution and Sci/Web's
  without-replacement per-batch distribution on the adapted train pool. All task
  assignments are frozen in advance. Each group gets the same real root prompt.
- Evaluate all validation instances initially/every 15 updates/finally. Final
  test uses seeds 0, 1, 2 and reports the mean of attempts, not pass@3. Evaluation
  has 16 slots, G=1, explicit tasks, independent request seeds and no archive.
  Tail padding is excluded from result denominators.
- ScienceWorld strict success requires native completion **and** score 100;
  reference success/reward are preserved separately. Native terminated worlds
  are not stepped again while other slots continue.
- vLLM uses explicit per-request seeds. Evaluation does not advance a training
  sampling RNG. The ordinary BEACON optimizer, reward and MiGPO computations run.
- Native artificial step cutoffs are raised for both arms; the collector owns H.
  Real world clocks, resource depletion and native terminal flags remain active.
  ALF explicitly seeds the loaded game. WebShop keeps its configured history=2
  instead of dropping history at 13,000 characters; token overflow still errors.
- Worker receipts verify the actor update version and successful execution of
  the native FSDP-to-vLLM loading path before generation. These receipts are
  retained per step; they are not a substitute for returned GPU execution logs.

## Reuse and records

WebShop uses its accepted v3 environment/index/assets. GPU learning uses the
accepted Torch 2.6/vLLM 0.8.2 stack. For ALFWorld, simulator actors use the existing
ALF package stack through Ray 2.48's per-actor Python option; no incompatible
Torch/vLLM packages are mixed into the learner. Two small source-isolated venv
wrappers reference the existing package files without executing old editable
hooks. Concurrent jobs lock only the short creation of their shared wrapper,
then train independently. No package installation or asset download occurs in
these launchers.

Each logical run is under `analysis/beacon-omr/omr-v2-<benchmark>-<method>-s0`:

- `progress.json`: current status and update; `attempts/<job>/launch.json` and
  `resolved-config.json`: provenance and effective parameters.
- `attempts/<job>/rollouts`: compressed actual prompts, responses, executed
  actions, native receipts, terminal outcomes and episode records. Each action
  round is flushed, including partial collections interrupted before an update.
- `attempts/<job>/optimization`: unique new-step reward/logprob/trajectory,
  segment and joint advantage summaries. Hardware padding is excluded.
- `cost-ledger`: completed collection records, including work later rolled back;
  an in-progress collection updates its own partial record until completion.
  `allocation.json` records attempt wall time and allocated GPU-hours when the
  launcher can close normally. Forced kills can leave incomplete allocation
  receipts and must not be counted as zero cost.
- `attempts/<job>/swanlab`: durable local metrics, online URL/ID and transport
  status. Credentials are read locally from the existing env file, not printed.
  Online SDK initialization is not itself server readback verification.
- `checkpoints/latest.json`: authoritative complete generation. First save is
  update 2, then every 5, final and safe stop. Both ranks' model/optimizer/scheduler/
  RNG, dataloader cursor, task schedule identity, manifests, cost watermark and
  monitoring lineage are saved. A new complete generation commits before the
  previous owned generation is removed. Scientific records are retained.

The baseline has no archive or restore snapshots. Full additionally persists
archive entries, decayed q/D counts, historical discovery, threshold activation,
sampler RNG/reservations at completed boundaries and verified payload references.
Its payload collector protects both the active pool and the current complete
model checkpoint before deleting unreferenced recovery material. Full evaluation
still uses independent root-only workers. See `FULL_OMR.md` for the integration
contract and test boundary; native two-GPU evidence must come from returned logs.
