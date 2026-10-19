# BEACON online milestone reuse

Work only in this independent checkout. Preserve the upstream source identity
`c88af45736b0a178106bc14ea89603ff02d8b222` and the Apache-2.0 notices.

The user approved a public origin repository and local commit/push for GitHub
synchronization. Remote HPC commands are executed by the user through VS Code
Tunnel; the assistant cannot connect directly to Orion.

Current user overrides to the execution specification:
- Use SwanLab, with SWANLAB_API_KEY supplied only at runtime, instead of W&B.
- Use Hugging Face for model/dataset downloads, with HF_TOKEN supplied at runtime.
- Check and configure environments on GPU compute nodes. Pure downloads may use
  CPU jobs. While the CPU queue is unavailable, the user approved lightweight Git
  clone/check on login, followed by direct GPU inventory submission. Keep large
  asset downloads and package installation on compute nodes.
- Prefer existing compatible environments/data/models. Do not mutate environments
  used by old experiments. Use the main assigned storage first; overflow storage
  is for large data/models when needed, never an unconditional duplicate.
- Check only ALFWorld, WebShop and ScienceWorld. Do not inspect or probe AppWorld
  environments, including as a candidate dependency base. The user explicitly
  excluded those installations from further checks.
- Keep local site paths and all credentials outside Git. Do not print credentials
  or a complete environment-variable dump.
- Stop separate readiness/probe rounds. Reuse accepted evidence and advance real
  training; essential protocol checks execute inside the training job. Both arms
  must use the same frozen source and common protocol. Never label baseline as full.
- Submit independent ready benchmark runs concurrently, with two H200 per run
  (twelve H200 for three benchmarks times baseline/full). Let Slurm queue for available resources; do not impose
  a one-at-a-time throttle or benchmark completion dependencies.
- The user paused the earlier v1 baseline jobs; their IDs are still pending.
  Preserve them and their records. Protocol v2 fixes common horizon/history semantics;
  v1 baseline must not be used as the formal comparator for v2 Full OMR. Do not
  silently resume v1 checkpoints under v2 or cancel unidentified jobs.
- The v2 six-condition array is 166362, frozen at c193e617acf03d94e18cad023934e65e64ff6923.
  The user requested one additional 18-hour continuation per element, each waiting
  for its own predecessor. Preserve that training revision during resume; newer
  orchestration scripts may submit/check it without migrating training state.
  Continue the SwanLab run when possible; the user also permits a new linked run.
- The user returned successful continuation submissions: indices 0–5 map to
  166364_0, 166365_1, 166366_2, 166367_3, 166368_4, 166369_5 respectively.
  They use orchestration revision 6993666a2a8305072dbb3024fcec76100fa943ec.
  These six continuations were subsequently cancelled at the user's request.
  No optimizer/resume logs have been returned yet.
- The user subsequently requested short real-training jobs to detect errors while
  the formal array queues. Prepare six independent one-hour/two-update smoke runs
  using the frozen c193e61 training source and isolated outputs/SwanLab project.
  The user explicitly approved real checkpoint saving followed by immediate
  deletion of these owned smoke checkpoints only, preserving save receipts/logs.
  Preserve the formal array. After the short-test submission hit
  QOSMaxSubmitJobPerUserLimit, the user explicitly requested withdrawing the six
  continuation jobs listed above to free slots for all six smoke conditions.
  The user returned successful cancellation/queue-removal and smoke submission
  output: smoke array 166370, controller f130cfe90c1b5e22a6ee1960f73cfa82997813cc.
  Do not duplicate this smoke array. Its GPU execution results are still pending.
  The original continuation submission ledger remains historical; do not treat
  its old child IDs as an active continuation chain after withdrawal.
- The user now requests replacing those six cancelled continuations. Use the
  explicit --replace-cancelled submission path, preserving jobs.json and writing
  jobs-replacement.json. It verifies the old cancellations and retains any new
  accepted jobs across quota-rejection retries.
  The first replacement attempt (controller c7e2f59) stopped before sbatch because
  sacct returned no old-job records. The user had already supplied successful
  cancellation/queue-removal output. Use --confirmed-cancelled-jobs with the six
  exact original IDs and a fresh all-states queue check instead of requiring
  accounting history. Formal 166362 and smoke 166370 were both still pending in
  the latest returned queue snapshot.
  Controller c86ad2bcef86184f3a3def2b1f053866aa22887f successfully submitted
  166437_0 (ALFWorld baseline) and 166438_1 (WebShop baseline). Index 2 hit the
  submit-job quota initially. The later queue snapshot confirms indices 2–5
  were submitted as 166514_2, 166515_3, 166516_4, 166517_5. Do not submit duplicates.
- The first returned ALFWorld baseline execution (attempt 166509-0, source
  c193e61) failed before updates: envs.py imported installed alfworld.info in the
  learner instead of deferring simulator-only imports to worker_python. The fix
  lazily resolves the same vendored factory. ALFWorld baseline/full must both use
  the new fixed source; never edit the old frozen checkout or migrate checkpoints.
  The latest queue showed 166362_1 (WebShop baseline) RUNNING, formal indices
  2–5 pending, all smoke indices pending, and all six continuations pending.
  Replace only the affected pending ALFWorld jobs (166362_3, 166370_0, 166370_3,
  166437_0, 166515_3), then submit ALFWorld-only baseline/full smoke jobs at the
  fixed revision. Their cancellation/new submission output is not yet returned.
  WebShop and ScienceWorld continue at c193e61. The user now explicitly requests
  queueing both repaired ALFWorld formal jobs and their continuations immediately,
  without waiting for smoke results. Freeze both arms and continuations at
  c6856146300b1d6a9f8706cc8a272016b133e400. The newer controller's
  submit_alfworld_training.py submits indices 0,3 concurrently and one afterany
  continuation each, recording accepted IDs across quota retries. No new formal
  submission receipt has been returned yet; do not claim the jobs are submitted.

Frozen experiment: ALFWorld, ScienceWorld, WebShop; baseline/full; seed 0;
Qwen2.5-1.5B-Instruct; one Slurm node and two H200 per training job. Preserve the
upstream per-environment code profiles and baseline/full common fixes. A milestone
does not stop the current rollout. Only later rollouts may restart, each with H
new actions; replay is separately costed. No teacher, SFT, extra reward or ablations.

Real environment restore, two-GPU updates, monitoring and resume need evidence
from actual execution. Unit fixtures are not evidence of completed HPC training.
Checkpoint replacement must commit the new complete generation before removing
the previous one. Preserve full scientific records and active restore payloads.
