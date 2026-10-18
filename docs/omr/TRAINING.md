# Three-benchmark training

The user requested actual training instead of further standalone checks. The
first wave trains **baseline** on ALFWorld, WebShop and ScienceWorld. Full OMR is
still being integrated; the trainer refuses `omr.enabled=true` until that path is
implemented. This is not a claim that HPC training has already succeeded.

## Run

Use the assigned work root, accepted preparation report `webshop-prepare-165873/report.json`
and the full published revision supplied in the chat:

```bash
bash "$ROOT/code/beacon-omr/experiments/omr/submit_training.sh" \
  "$ROOT" "$ROOT/evidence/beacon-omr/webshop-prepare-165873/report.json" "$REV"
```

The script freezes a clean detached source checkout without touching the four
modified files in the user's existing checkout. It submits array `0-2%3`:
0 ALFWorld, 1 WebShop, 2 ScienceWorld. Each element uses two H200, 32 CPUs, 256 GiB
RAM and up to 18 hours on testqueue. All three may run concurrently, using six
H200 in total when resources are available; Slurm queues elements as needed.
There is no dependency or prescribed start order between benchmarks. Failure of
one element does not cancel the others. These are training jobs, with no separate
environment, model, index or transport probe beforehand.

To resume one benchmark, repeat the command with `alfworld`, `webshop` or
`scienceworld` as its fourth argument and the **same revision**. The job locks its
logical run, verifies the complete latest checkpoint and continues the exact task
schedule. If source/protocol/manifest changed, it refuses an implicit migration.
An already completed latest run does not perform extra optimizer updates.

Slurm requests a stop 20 minutes before the time limit. The trainer saves after a
complete update and can interrupt evaluation between batches. To request an
earlier boundary stop, create `STOP` inside the active attempt directory printed
at startup. Do not cancel unrelated jobs or delete old experiment directories.

## Protocol

- Seed 0, Qwen2.5-1.5B-Instruct, 150 native trainer updates, 16 task groups x 8
  trajectories. H is 30/15/30 for ALF/Web/Sci. Frozen method-specific profiles
  remain in `experiments/omr/configs` and the resolved config is saved per attempt.
- Native instance manifests are frozen under `analysis/beacon-omr/manifests` and
  shared by the future full arm. The upstream entry points have train/test pools;
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

## Reuse and records

WebShop uses its accepted v3 environment/index/assets. GPU learning uses the
accepted Torch 2.6/vLLM 0.8.2 stack. For ALFWorld, simulator actors use the existing
ALF package stack through Ray 2.48's per-actor Python option; no incompatible
Torch/vLLM packages are mixed into the learner. Two small source-isolated venv
wrappers reference the existing package files without executing old editable
hooks. Concurrent jobs lock only the short creation of their shared wrapper,
then train independently. No package installation or asset download occurs in
these launchers.

Each logical run is under `analysis/beacon-omr/omr-v1-<benchmark>-baseline-s0`:

- `progress.json`: current status and update; `attempts/<job>/launch.json` and
  `resolved-config.json`: provenance and effective parameters.
- `attempts/<job>/rollouts`: compressed actual prompts, responses, executed
  actions, native receipts, terminal outcomes and episode records.
- `attempts/<job>/optimization`: unique new-step reward/logprob/trajectory,
  segment and joint advantage summaries. Hardware padding is excluded.
- `cost-ledger`: immutable collection records, including work later rolled back;
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

The baseline has no archive or restore snapshots. Full will require its own
archive/controller/payload checkpoint state before it can be launched. Local
tests cover configuration, split isolation, terminal handling and checkpoint
failure/recovery; native two-GPU execution must be established by returned logs.
