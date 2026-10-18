# Full OMR integration and evidence boundary

`FullCollectorObserver` is the training observer selected by `omr.enabled=true`.
Evaluation always uses `RootCollectorObserver`. It does not replace the BEACON
optimizer, reward formula, trajectory/segment advantages or task distribution.
Both arms use protocol v2 and the same frozen commit and manifests.

## Update transaction

1. Read the next 16 tasks from the frozen schedule. Freeze archive/statistics and
   historical categories at policy version k; plan one root or restart per group.
2. Restore eight independent worlds for each group. Check concrete task identity,
   native replay receipts and the rendered group input. Any failed member causes
   exactly one whole-group fallback to that same task's root and quarantines the
   requested entry. Failure of the root itself aborts training.
3. Generate at most H **new** actions per trajectory. Executed replay actions never
   become response tokens, rewards, q/D evidence or new exploration. Native
   terminals are not stepped again. Full can capture continuing progress without
   ending or extending the source rollout.
4. Update the ordinary BEACON policy. Only after the optimizer returns, commit
   outcomes to their actual starts and admit pending entries for k+1. No source
   suffix is treated as a fresh H-step trial of a state discovered along it.
5. Save model/optimizer/RNG/dataloader plus the OMR transaction at the same update.
   A complete new checkpoint commits before old checkpoint/payload cleanup.

## Native state and event contracts

- **ALFWorld:** explicitly selected gamefile/root seed, executed command replay,
  game identity, native admissible commands, terminal outcome and world feedback.
  A similar command string alone does not admit a state: require an admissible
  action and affirmative native feedback. Unverified matched events are unknown.
  The reward matching offset is preserved independently, so the original BEACON
  prefix boundary is not rewarded twice. Statistical events use the nonfinal
  milestone slots plus strict Goal, reserving the final slot for Goal to avoid
  duplicate terminal weight. Missing milestone entries produce no invented IDs.
- **ScienceWorld:** the same explicit task/variation and native load/reset/step
  sequence, scores/completion and legal-action receipts. No Java world deepcopy,
  extra alias enumeration or retries with random seeds. Text differences are
  recorded; manager history comes from actual replay observations. This is an
  explicit replay contract, not a claim that an observation hash proves equality
  of every hidden JVM field. Statistics use observed nonterminal native score
  thresholds plus Goal (`getCompleted` and score >=100). New thresholds activate
  next update with zero historical N; no retrospective failures are fabricated.
- **WebShop:** reuse the adapter verifying the goal manifest, action prefix and
  complete session/browser/detector witness. Preserve selected product, options,
  native history and simulator RNG; catalog/index handles are not serialized.
  Events distinguish search, detail, required options and Goal. Optionless goals
  skip the options event; replay clears instantaneous milestone flags.

All three use the ordinary manager history renderer. New prefixes are admitted
only if the next prompt fits the common 7000-token limit. Actual rollout overflow
still errors; it is not silently dropped. New state IDs include concrete restore
and history content. Restore cost estimates use prefix action count as a proxy;
actual replay actions and worker time are measured separately.

## Statistics, coverage and storage

The configured defaults remain alpha=.5, priority/staleness/coverage=.7/.1/.2,
eta=.05, beta=1, half-life=20, Beta(1,1), n_ref=8 and stale cap=50. L averages
`r*4*q*(1-q)` over each start's own uncompleted events including Goal. D uses
decayed discovery successes/trials without a prior or reward modification.

The pool retains the configured 8192 total / 8 per task / 2 per category / 2
untried per task limits. Pending captures reserve the untried capacity within an
update in deterministic slot order; rejected candidates still contribute verified
historical discovery. Newly captured states never enter the current sampler.
Untried entries remain protected, and final untried counts are reported as right
censored rather than claimed as completed first trials.

Each update saves requested and actual starts, mixture probabilities, native
restore receipts/fallbacks, candidates/admission reasons, per-start event counts,
L/D views, source steps and new-only outcomes. Root success and conditional restart
success are separate; main evaluation remains root-only. Partial interaction and
cost records survive interruption. Optimizer-policy and native weight-loading
receipts are checked before accepting generated samples.

Complete model checkpoints contain archive entries, statistics, historical seen
categories, score thresholds and sampler RNG. Their payload references are
content-addressed and checksum-verified before resume. Cleanup protects both the
active pool and the latest committed model's recovery dependencies. Historical
scientific records are retained even when unused recovery payloads are collected.

## What is verified locally

Fixtures exercise the production observer/controller/manager and checkpoint
subclass with deterministic native-interface worlds and a replaced Ray transport:
next-update activation, suffix-only evidence, dynamic thresholds, unknown labels,
group fallback/quarantine, recursive replay, original history limits, independently
restored worlds and full checkpoint rollback. Worker fixtures check rejection of
stale policy versions/missing native weight-load receipts and resume of the version
marker. These are **not** native simulator or two-H200 optimizer acceptance.

Returned evidence so far covers the earlier environment preparation, including
WebShop model loading and offline SwanLab. The user has submitted v1 baseline
jobs but has not yet returned their IDs or optimizer logs. Full native replay,
actual two-H200 updates, remote SwanLab readback, six final learning curves and
real interruption/resume remain evidence to collect from the formal v2 jobs.

The ScienceWorld artificial-limit interpretation follows the upstream Python
wrapper: `envStepLimit` is independent of the native server completion flag, and
reset itself calls `step("look around")`. The common v2 limit fix leaves native
world time untouched. Source:
[ScienceWorld wrapper](https://github.com/allenai/ScienceWorld/blob/main/scienceworld/scienceworld.py).
