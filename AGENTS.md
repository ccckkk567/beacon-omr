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
  continuation each, recording accepted IDs across quota retries. The latest
  returned queue contains new ALFWorld arrays 166613 and 166629 (indices 0,3)
  and continuations 166630_0, 166631_3. Preserve these; do not resubmit ALFWorld.
- WebShop baseline attempt 166511-1 at c193e61 failed before worker/model
  initialization: identical complete goal dictionaries collide in the strict
  task manifest. Canonicalize full goal content, keep all native test identities,
  retain the first index of each identity, and exclude train copies of test goals.
  Do not hide overlap by adding an index to the identity, merge only instruction
  text, or remove the split check. Both baseline and full/restart adapters must
  share the canonical pools; retain counts and duplicate source indices in the
  manifest. Both WebShop arms need the same repaired frozen source.
  The user queue showed old WebShop full 166362_4 RUNNING, smoke 166370_1/4 and
  continuations 166438_1/166516_4 pending. Retirement targeted these exact five
  IDs; subsequent confirmations are recorded below. Submit
  repaired WebShop smoke/formal pairs and per-arm continuations through the
  journaled submit_benchmark_training.py --benchmark webshop --include-smoke.
  Each pair is concurrent; formal jobs do not wait for smoke results. Preserve
  accepted IDs on quota retries and do not duplicate pending jobs.
  Later output confirms 166370_1, 166370_4 and 166362_4 are CANCELLED;
  166438_1 and 166516_4 were absent from the targeted remaining-ID query.
  CANCELLED records can remain visible in squeue --states=all; presence alone
  must not block replacement submission as though these were active jobs.
  The latest queue confirms repaired WebShop formal jobs 166698_1/4 pending.
  New WebShop smoke/resume IDs have not been returned; do not duplicate them.
  ScienceWorld 166362_2 was RUNNING at 46:58 on JDS9B74. The read-only stack
  diagnostic failed to launch step 166572.2 with a Slurm socket timeout, before
  any process/stack output. This establishes a diagnostic launch failure, not
  the training root cause. Full 166362_5 was RUNNING at 1:51 on H64Y574.
  At 2026-10-19 11:51:40 +08, JDS9B74 reports MIXED, CPULoad=621.54,
  CPUEfctv=112, CPUAlloc=88, FreeMem=712726 MiB. Baseline log mtime is
  11:18:11 (over 33 minutes unchanged), progress still starting. Full log mtime
  is 11:46:24 (about 5 minutes unchanged), progress starting, TaskRunner 2268434
  in physical allocation 166362 on H64Y574, attempt 166362-5. Its log ends in
  buffered config output without the baseline resource-check warning; do not
  claim both are stuck in the same call. Node-wide free memory does not rule
  out a per-job memory limit. The H64Y574 diagnostic ran at 11:54:29:
  TaskRunner 2268434 was RNl, elapsed 08:13, CPU 1.5%, RSS 941192 KiB.
  py-spy was denied permission; do not retry with sudo or weaken ptrace policy.
  Its own stdout shows 3322 train and 1661 test variations loaded. This is
  catalog setup, not proof that all 144 asynchronous simulator actors are ready.
  No Python traceback was returned. The next requested evidence is the existing
  full run's own Ray debug_state.txt/raylet logs, not another ptrace attempt.
  New-source startup diagnostics record stages and self-dump stacks every 180
  seconds during a slow initialization stage, stopping before trainer.fit.
  They do not hot-patch current frozen jobs or establish a training-root-cause fix.
- The subsequent read-only Ray-log step 166362.2 could not start before the
  caller's 60-second timeout. Stop repeating srun/ptrace diagnostics. The next
  returned queue confirms formal ScienceWorld 166362_2 RUNNING at 1:01:51 and
  166362_5 RUNNING at 16:44; both shared progress files still say starting at
  c193e61 and both lack complete checkpoints. Old ScienceWorld smoke 166370_2/5
  is RUNNING at 14:43 on GDS9B74; resumes 166514_2/166517_5 are pending.
  Prepare replacement of only these six ScienceWorld jobs with the journaled
  recover_scienceworld.py controller. It checks exact ownership/source bindings,
  no optimizer/checkpoint evidence before and after cancellation, terminal vs
  COMPLETING states and released writer locks before any replacement submission.
  Both arms use training revision fdbf5569b82d9c1afd1ad2578c914adf00d0f417:
  one-hour smoke pair, independent 18-hour formal pair, own-parent 18-hour resumes.
  Retain all earlier attempt/log files. This is an observable real-training
  restart, not a confirmed fix for the unknown stall. New IDs are not returned.
  The same queue confirms WebShop smoke 166697_1/4, formal 166698_1/4, resumes
  166699_1/166700_4; preserve these along with all repaired ALFWorld jobs.
- Old ScienceWorld baseline smoke 166370_2 (physical attempt 166696-2, c193e61)
  now reports RayTaskError(TimeoutError): rank-zero worker registration did not
  publish its center within 300 seconds. Zero optimizer updates/actions and no
  complete checkpoint; deleted smoke checkpoint directory is not save success.
  Worker-count warnings reach 314 but do not establish the timeout cause.
  The user executed recovery d89c79828be5f46f0262f8ba11bdb55cba26ffa4. It requested
  cancellation of 166514_2, 166517_5, 166362_2 and 166362_5, then stopped while
  166362_2/5 were exiting. No replacement jobs were submitted. Continue with the
  same controller/retirement ledger after cleanup; do not create another chain,
  hot-patch frozen sources or claim the startup instrumentation fixes the cause.
- The user initially redirected work to diagnosing why old ScienceWorld
  166362_2 (physical allocation 166572) remains COMPLETING, pausing the proposed
  independent-output restart at that point. At 2026-10-19 12:16:15 +08,
  7m07s after EndTime 12:09:08, JDS9B74 was MIXED+COMPLETING, load 602.63,
  CPUAlloc=56, AllocMem=753664 MiB, allocated GPUs=6. Controller step records
  still list extern/batch/0 as RUNNING. sstat returned "No steps running" and
  exit 0, without statistics. Slurm 23.02.8 sstat source also emits this message
  when allocation lookup fails before node statistics, so it cannot prove
  processes are gone or that the node responded. KillWait=30, MessageTimeout=10,
  UnkillableStepTimeout=300, SlurmdTimeout=300, proctrack/cgroup; node Epilog is
  /cm/local/apps/cmd/scripts/epilog. Do not sum these timers as a hard cleanup
  deadline or assert D-state/epilog hang without node evidence. The user has
  only the login-node terminal, no existing JDS9B74 terminal. Node-side process,
  slurmstepd/slurmd and epilog evidence is still required. Do not retry ptrace,
  failed srun launches, broad kills or assume SSH access. New ScienceWorld
  jobs remain unsubmitted; preserve repaired ALFWorld/WebShop jobs.
- The latest user instruction, "算了，那还是换种方式吧", authorizes the
  independent-output ScienceWorld restart instead of waiting for old cleanup.
  Use isolate_scienceworld.py with one full revision for controller/training.
  Keep old jobs/files intact and use run tag restart-166362 for formal outputs,
  manifests and SwanLab names. The user explicitly clarified that only the old
  job wait should be removed: do not exclude JDS9B74 or constrain node selection;
  let Slurm allocate normally.
  Submit both one-hour smoke arms, both independent 18-hour formal arms and
  each formal arm's own 18-hour afterany continuation. Keep model/env/protocol
  settings unchanged. The continuation reads only its new formal checkpoint.
  Preserve the old retirement receipt in retirement-before-isolation.json;
  mark its original ledger superseded under the same submission lock so even
  frozen d89c798 refuses to create a second chain. New journal lives under
  training-submissions/scienceworld-restart-166362/jobs.json. Quota retries keep
  accepted IDs; uncertain submission outcomes require inspection. No new HPC
  submission output has been returned yet. This avoids the old output/cleanup
  dependency; it is not evidence that the Ray registration failure is fixed.
- The subsequent queue now contains new ScienceWorld arrays 166748/166749
  (indices 2,5) and continuations 166750_2/166751_5, all pending. Preserve them.
  The old ScienceWorld 166362_2 is still COMPLETING. The user explicitly requests
  no node exclusions; do not exclude JDS9B74 in later replacement submissions.
- ALFWorld baseline smoke 166613_0 (physical attempt 166727-0, c685614) reached
  trainer.fit but failed on its first explicit root reset: TextworldBatchGymEnv
  has no .unwrapped in the reused TextWorld 1.7.0 runtime. Zero optimizer updates,
  zero actions, no checkpoint. SwanLab created online run y0whj971 and reported
  uploading records; this is not successful training or checkpoint-save evidence.
  Fix the production AlfworldWorker reset for direct and Gym-wrapped batch APIs,
  preserve explicit seed/game choice and iterator restoration, and verify the
  returned game identity. Full replay shares this same production reset.
  Latest queue has 166613_3/166629_3 RUNNING, 166630_0 pending without dependency,
  166631_3 pending with dependency, and no 166613_0/166629_0. Replace only these
  six identified ALFWorld jobs via recover_alfworld.py. It validates ownership,
  script/source and absence of old training progress, requests cancellation with
  continuations first, then queues independent new outputs tagged restart-166629
  without waiting for cleanup. Preserve old files. Submit baseline/full smoke
  (1h), formal (18h), and own-parent continuations (18h) using one frozen revision,
  normal node selection and the existing quota-safe journal. Preserve all WebShop
  and newly queued ScienceWorld jobs. New ALFWorld replacement IDs are not yet
  returned; package/environment installation is unnecessary for this API fix.

- Latest user override (2026-10-19): withdraw all project `omr-smoke` jobs,
  including pending and running ones, and use formal training for further
  debugging/acceptance. Preserve `omr-training`, `omr-resume`, historical logs
  and experiment files. Cancellation is to be executed by the user through
  Tunnel; no completion receipt has been returned yet. Do not submit more smoke
  jobs or repeat old recovery commands that unconditionally include smoke.
  This overrides all earlier smoke-matrix requests above. Do not resubmit
  already queued formal jobs just to stop smoke jobs.
- ScienceWorld smoke 166748_2 (physical attempt 166830-2, source 80dd118)
  failed on GDS9B74 at the same rank-zero registration timeout (300 seconds),
  with zero optimizer updates/actions and no checkpoint or SwanLab run.
  TaskRunner startup recorded processor.load at 32.57 seconds and
  workers.import_classes at 536.31 seconds. That interval also includes the
  vLLM early-version import, so do not attribute all 503.74 seconds to the
  processor alone. Placement/registration spans 593.57 to 1022.94 seconds.
  Worker-count warnings reach 314; the cause of the rank-zero startup failure
  remains unconfirmed. Formal runs share this startup path; stopping smoke
  does not itself fix the failure.

- The returned queue now has no smoke jobs. WebShop formal 166698_1/4 is
  RUNNING; preserve both and continuations 166699_1/166700_4. ALFWorld formal
  166853_0/3 and children 166854_0/166855_3 are PENDING; ScienceWorld formal
  166749_2/5 and children 166750_2/166751_5 are PENDING. Old 166362_2 remains
  COMPLETING and is not a target. The ALFWorld reset-repair pair was thus queued;
  do not repeat its earlier recovery command.
- Returned TaskRunner self-stacks identify the slow main_ppo.py:89 import of
  verl.utils.vllm_utils -> DeepSeek/MoE -> torch.compile/Inductor. This was only
  an early version check and the configured lora_rank is zero. Use distribution
  metadata only when LoRA needs a version check. The filelock audit hook in one
  sampled stack does not establish a file lock deadlock or a package fault.
  Lazily export environment managers and defer ALFWorld image-only Torch imports
  so simulator class loading does not eagerly import the learner stack.
  These changes remove demonstrated unnecessary imports; they do not prove the
  separate rank-zero registration timeout is fixed.
- refresh_pending_training.py targets only the eight pending jobs above,
  validating identity, source, zero runtime and no previous attempt before
  cancellation. It restricts scancel to PENDING, cancels continuations first,
  checks states again and journals accepted replacements. A running/missing/
  changed target aborts rather than cancelling or duplicating it. New formal
  output tags are startup-166853 (ALFWorld) and startup-166749 (ScienceWorld).
  Submit only the four 18-hour formal tasks and their four own-parent 18-hour
  resumes; no smoke, node exclusions, package changes or file deletion.
  Submission ledger: training-submissions/startup-166749-166853/.
  32 targeted local tests passed; no new submission or GPU acceptance receipt yet.

- The user executed refresh controller 5dd260d6bf8bda538c16b28ccc4428e087d55e6d.
  It stopped on scancel 166853_3 after writing intent for the four continuations
  and both ALFWorld parents; no new jobs were submitted. The follow-up queue
  confirms arrays 166853/166854/166855/166750/166751 CANCELLED at zero runtime,
  with 166853_3 not found. ScienceWorld 166749_2/5 remain PENDING. WebShop
  166698_1/4 remain RUNNING and 166699_1/166700_4 PENDING; preserve all four.
  The repaired controller materializes each remaining array element before
  pending-only cancellation and reports/reconciles nonzero cancellation errors.
  Resume the existing retirement journal with --revision fixed to 5dd260d;
  use --controller-revision for the newer orchestration checkout. Never reset
  the receipt or change the training revision to bypass its binding. Skip the
  six already withdrawn targets, replace only remaining pending targets, then
  submit four formal tasks and four own-parent continuations. No smoke or node
  exclusions. Sixteen local orchestration tests pass; HPC retry receipt pending.

- The user returned successful refresh output at controller 175a616: ALFWorld
  formal 167125_0/3 with resumes 167126_0/167127_3; ScienceWorld formal
  167128_2/5 with resumes 167129_2/167130_5. All use training 5dd260d.
- Both WebShop formal jobs 166698_1/4 at d6dff44 have now failed before their
  first optimizer update. Baseline completed initial evaluation and its first
  15-action rollout, then compute_response_mask rejected object-dtype boolean
  metadata from real collate_fn. Full completed initial evaluation, then the
  shallow copy of its vector in _group_render ran __del__/close and killed the
  shared live actors. This is not an OOM/time-limit diagnosis. Both bugs were
  reproduced locally using production collation and vector destructors.
  Normalize padding to a shape-checked boolean array before Torch conversion;
  preview only manager rendering state, never copy an actor-owning vector.
  ALFWorld preview must not mutate its original admissible-command cache.
- Latest queue: the eight new ALFWorld/ScienceWorld formal/resume elements above
  and old WebShop resumes 166699_1/166700_4 are all PENDING. WebShop parents
  166698_1/4 are absent. Ancient 166362_2 remains COMPLETING; never target it.
  recover_first_update.py validates and withdraws only these ten pending IDs,
  children first. If any target starts or changes identity, stop and inspect.
  Preserve all old artifacts and reject any prior optimizer/checkpoint evidence.
  Use isolated tags update-fix-167125, update-fix-166698, update-fix-167128 for
  ALFWorld, WebShop and ScienceWorld, respectively. Submit six formal tasks
  concurrently plus six own-parent resumes at one frozen repaired revision;
  all remain 18h / 2 H200 / 32 CPU / 256 GiB. No smoke or node exclusions.
  New SwanLab runs use these tags; continuations follow normal checkpoint/run
  lineage. The journal first-update-167125-166698-167128 retains accepted IDs
  across quota retries. 58 targeted local tests pass, with Ray/native engines
  and Slurm transport substituted. No repaired GPU update is claimed yet.

Frozen experiment: ALFWorld, ScienceWorld, WebShop; baseline/full; seed 0;
Qwen2.5-1.5B-Instruct; one Slurm node and two H200 per training job. Preserve the
upstream per-environment code profiles and baseline/full common fixes. A milestone
does not stop the current rollout. Only later rollouts may restart, each with H
new actions; replay is separately costed. No teacher, SFT, extra reward or ablations.

Real environment restore, two-GPU updates, monitoring and resume need evidence
from actual execution. Unit fixtures are not evidence of completed HPC training.
Checkpoint replacement must commit the new complete generation before removing
the previous one. Preserve full scientific records and active restore payloads.
