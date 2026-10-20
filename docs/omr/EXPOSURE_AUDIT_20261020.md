# OMR opportunity audit: implementation and evidence boundary

User specification: `OMR_Next_Steps_Audit_First_20261020.md`.
Branch: `codex/omr-exposure-audit-20261020`, based on6500b4f, retaining the
WebShop input repair and all existing continuation controllers. No live checkout
or submitted training target is changed. No sampler/reward/identity/L/D formula,
task schedule, B/G/H, archive limit, full-batch generation, per-turn weight sync
or vLLM lifecycle change is included.

## Read-only CPU audit

`experiments/omr/exposure_audit.py` reads the six known logical run paths, not a
generic job name. Actual progress/attempt/launch/resolved config, manifest,
latest pointer and COMPLETE metadata are captured. It verifies manifest content,
frozen source checkout identity and checkpoint signature binding where present.
Model/optimizer shards, `data.pt`, `protocol.pt`, restore payloads and credentials
are never opened. No production save lock, simulator, tokenizer, learner, Ray or
Torch is used. A latest-pointer rotation or incomplete gzip/JSON produces an
explicit gap rather than partial rows. Individual input reads are bounded.

Task schedules are rebuilt by importing the actual frozen source's
`training_schedule` function, with its original local NumPy RandomState. Actual
source file hashes and the resulting schedule hash are retained. TaskSpec uses
the full family/instance/manifest/target/root-seed identity; no field is stripped
to manufacture hits. Shadow sampler modules have an isolated namespace and use
their own random.Random state, reconstructed from the actual seed and a contiguous
retained sequence. Prior update statistics snapshots supply the frozen views;
the final archive is never backfilled into an early update. Saved plan probabilities,
channel/state choices, reservations and L/D are checked against the shadow output.
Gaps disable deterministic claims and unknown denominators remain null.

Resume lineage caps exclude optimized but subsequently rolled-back collections.
Collections outside the retained attempt chain are reported separately. Full
commit files are indexed by completed update; collection `u000016` means policy16
was sampled and update17 committed. An optimizer record is required for U.

The report gives B/J/E/Q/R/U, conditional Q/E, per-group task/eligibility/reason,
selected pre-sampling L/D, active tried post-commit L/D and task/cohort waits.
No numerator or ratio is filled with zero when its evidence is missing. Future15
and through150 scenarios separately freeze the current eligible archive or assume
optimistic successful discovery after an earlier scheduled visit; neither predicts
restoration success or learning. Historical entries absent from every snapshot
remain a stated reconstruction limit; a task revisit is not automatically a
checkpoint-eligible opportunity.

For restored groups, outcomes are joined to actual optimizer trajectory IDs and
response-mask token counts. Known/unknown Goal labels are distinguished. N/S
comparisons aggregate all same-start groups before applying the one-version decay
to the post-commit view. Discovery counts are checked separately. Originally
requested ScienceWorld17/18/20 and ALFWorld10/12 groups additionally read their
new executed steps; prefix replay is not injected into training rows. Initially
complete events, initial milestone offsets, descendant admissions and evidence
gaps remain explicit. Advantage summaries are diagnostic, not gradient proofs.

Existing common root-validation nodes are compared using actual instance IDs,
seeds, known labels, task-macro, native score, terminations and round cost ledgers.
Resolved model initialization, task split/schedule, H/input/history, reward and
optimizer settings are compared separately. No evaluation is rerun. Missing
shared nodes/config evidence cannot support an improvement claim.

## Confirmed timing collision and minimal patch

Production source used `perf/train/*` for both cumulative `split_cost` values and
the last training collection's `performance_metrics`. The train hook writes the
latter last, but validation again returns `split_cost`; the trainer's ordinary
metrics merge can replace the update value with the cumulative value. This is a
key-namespace collision, not evidence of an environment or logging bottleneck.

Changes are restricted to timing:

- `perf/update/train/*`: current train collection.
- `perf/eval/round/validation/*` and `/test/*`: sums over the actual evaluation
  collections in that round; validation and final test do not overwrite each other.
- `perf/cumulative/train/*` and `/eval/*`: accumulated cost-ledger timings,
  including failed attempts as before.
- Cached final metrics omit evaluation-round timing keys when no new round ran.

Old logs are preserved. Their validation-node `perf/train/*` can contain cumulative
values and must not be silently rewritten. New names are intentionally distinct;
historical and fixed series require an explicit analysis-label change.

The patch is **not deployed to current training/continuations**. Their source
remains98d7a69/ddae412/2ebc0a7 as already bound. Although this timing change does not
alter checkpoint schema or scientific formulas, a new training source changes
the protocol signature. Switching a resumable run would require a separate narrow,
tested compatibility decision at a complete checkpoint boundary. There is no
skip-signature option, migration, automatic resubmission or new GPU smoke here.

## Execution and deliverables

One CPU-only ioqueue allocation uses the existing Python environment solely for
NumPy/standard-library file analysis. No environment install/check is requested.
The user runs it through the existing VS Code Tunnel; Codex cannot connect directly.
Outputs live in a fresh `analysis/beacon-omr/audits/exposure-20261020-<job>` directory:

- `restart_opportunity_audit.md`
- `restart_group_trace_audit.md`
- `baseline_full_eval_comparison.md`
- `audit-summary.json` (compact review/flags)
- `audit.json` (full per-group/cohort/lineage evidence and input inventory)

CPU fixtures cover no final-archive backfill, actual sampler replay, unknown Goal,
multiple groups sharing one start, rollback caps, truncated gzip exclusion,
initialization mismatches, report generation without archive/RNG/input mutation,
and the train+validation timing merge. Related learning/root/full-batch regressions
also pass. These fixtures are not actual Orion audit results.

All three real audit conclusions remain pending the CPU job output. In particular,
this implementation does not confirm the plan's example counts, conclude that
low reuse is structural, certify the five native restarts or compare efficacy.
Review real flags first; fix a specific reproduced data-path bug only if supported.
Any outer task-sampling redesign remains a separate proposal for user decision.

## CPU startup failure168120 and isolated retry

Returned accounting is FAILED4:0 after2s. srun reports Illegal instruction on
A386345X1309686C; no AUDIT progress line was printed. No real opportunity/group
audit was completed. The output also confirms WebShop168005/168006 are now
testqueue18h, and the ALF/Science continuations remain h200p72h.

The exact binary/library triggering SIGILL is not identified by that log. The
original launcher invoked the GPU venv interpreter with site startup enabled.
Orion documents generic and AMD-specific Python3.12.5 builds. The retry chooses
the documented build from /proc/cpuinfo vendor_id, invokes it with -I -S -B,
and explicitly appends only the existing site-packages directory to import NumPy
without processing .pth hooks. It verifies Python3.12 and NumPy origin, prints
interpreter/NumPy/RandomState/audit stage markers, and disables core dumps for this
CPU job. It does not install packages or mutate the training venv, GPU checkout,
checkpoint store or submitted jobs. NumPy RandomState and the frozen audit sources
remain unchanged. Unsupported CPU/ABI/origin fails explicitly rather than silently
using another random generator. Real CPU execution remains pending.

Bootstrap regression exercises a separate -I -S process, confirms a poison .pth
is not executed and report generation completes with explicit absent evidence;
it uses a startup-only NumPy fixture and is not an ISA or real schedule test.
References: [Python command-line isolation](https://docs.python.org/3.12/using/cmdline.html),
[NumPy native-extension troubleshooting](https://numpy.org/doc/stable/user/troubleshooting-importerror.html).
