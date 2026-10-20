# WebShop input repair and controlled continuation — 2026-10-20

Scope: the user-approved `WebShop_故障核验与修复续训方案_20261020.md`.
Branch: `codex/webshop-input-resume-fix-20261020`, based on full-batch branch
`07429fb`. The deployment SHA is bound by the submitted requests and receipts.
Public code/operational-metadata synchronization was explicitly authorized.

## Returned evidence and current operational boundary

| Arm | Source commit | Last logged optimizer update | Pinned complete checkpoint |
| --- | --- | --- | --- |
| Baseline | 98d7a695f4a4b333e2984b731330d088612cf81b | 39 | 35 |
| Full | ddae412e6195d2f86dace7f5ce2630d155ac0c01 | 10 | 10 |

Baseline failed with `9910 > 7000` during dynamic prompt preprocessing. Full
failed while previewing a restored product/search page through the root-only
`Instruction:` parser. Full's health-only `step:11` is not an optimizer receipt.
Returned accounting confirms original parents `167237_1` and `167698` FAILED,
exit 1:0. It does not establish absence of every possible cluster issue.

The user executed a freshly guarded hold of only `167238_1`: owner linz3,
account prjsnnn0000269, expected old controller 1d30ff3, zero runtime PENDING;
afterward Priority=0 and Reason=JobHeldUser. `167235_0` remains protected, as
do ALFWorld and ScienceWorld jobs. The new submitter may retire that exact held
old WebShop child and submit only two independent named formal continuations.
No automatic child chain, smoke jobs, node exclusions or package installation.

The reported COMPLETE hashes are pinned in `webshop_migration.SOURCES`. They
have not yet been independently verified against all rank files on HPC. Initial
login-node submission checks inspect metadata and sizes; full SHA256 verification,
protocol/data/OMR loading and actual native/tokenizer acceptance execute inside
the authorized formal allocations. No real-case or resumed-update success is
claimed before those receipts return.

## Input/state implementation

`agent_system/environments/webshop_input.py` is the shared input assembler.
The verified native session supplies its public instruction through the existing
replay adapter. Root text is compared with the root parser; restored text is
checked against the immutable payload's original task. Current page and legal
actions remain those returned by native replay. No reset/navigation is added to
obtain an instruction, and no hidden goal/ASIN/reward metadata is added to prompts.

`native_restart.render_starts()` and the older `restore_manager_slots()` share
per-slot task/page/history assembly. Preview isolates manager memory without
copying the actor-owning vector. Root/restart slots retain their own histories.
Old payload hashes and world/browser/detector witnesses remain checked first.
Same-profile rendered inputs remain byte checked; the sole supported derived
input transition is legacy `webshop-input-v1` to `webshop-input-v2`, with old/new
input hashes recorded. Old payloads are never relabelled or rehashed in place.

The v2 rule is common to root, step, preview, installation, candidate checks and
evaluation, with the actual training tokenizer attached to the manager:

1. Render the ordinary history template. At >13000 characters restore the
   upstream no-history fallback.
2. Measure the entire chat template, including the generation prompt. At >7000
   tokens also try the same no-history template, irrespective of character count.
3. If still long, retain task/instructions, action format, full legal-action menu
   and chat boundaries; trim only the public page body, deterministically by
   complete leading lines, or a leading token boundary when necessary. Add an
   explicit omission marker and re-tokenize the complete input.
4. If non-trimmable material alone exceeds 7000, raise a diagnostic input-budget
   error. It is not an agent failure/reward, and does not quarantine OMR states.

Raw current pages and complete manager histories remain recovery/audit material.
Generation inputs, raw prompt IDs, masks and observer `input` refer to the same
fitted text. Input audits record actual/original lengths, fallback/trim reason,
profile and hashes; changed original input/body are retained. All original slots,
including done slots, still use full-batch generation and the same fitting rule.

No change to Qwen2.5-1.5B identity, 7000/512 caps, B16/G8/H15, reward, advantages,
optimizer/LR, split/schedule, per-turn weight sync/sleep/wake, terminal cache,
OMR priority/sampling/archive formulas or latest-model rotation. Protected engine,
worker, reward/advantage and optimizer code blobs are checked against each source;
Full also protects its archive/sampler/statistics implementations. Baseline's
inherited ray_trainer differences from 98d7a69 are phase/timing logging only.

## Narrow migration and storage

The new named jobs reuse each original logical run and unique model checkpoint
store, under its exclusive `.run.lock`. No historical model copies or symlink
workarounds. Existing execution-profile/lineage files are read against their old
source and remain immutable; the new attempt declares its actual new source and
input profile through an exact migration request.

`webshop_migration.authorize()` requires exactly the known method/source/update,
generation COMPLETE hash, authentic old signature, original launch/assets,
manifest, schedule and data cursor. The frozen source composer is loaded from
the original clean checkout and must reproduce the old signature. Scientific
configs compare equal after only documented source/input-profile/attempt/monitor
differences and explicit missing legacy/sync defaults. Unknown sources, different
models/tasks/arms, altered learning settings or changed protected code reject.
There is no ignore-signature/force-resume option.

Both rank model/optim/extra files, data.pt, protocol.pt, policy version and
scientific ledger watermarks are verified and restored. Full additionally loads
its controller, archive, candidate RNG and immutable payload closure; its committed
update must match the model. Conditional evidence on old archive entries must
actually be zero/untried. Unexpected nonzero evidence stops for an explicit
statistics-version decision rather than clearing the pool. Root evidence is kept.

Baseline resumes 35 -> next update 36; old unsaved updates 36–39 remain historical
branch records and are recomputed on policy. Full resumes 10 -> 11. Resource costs
from old attempts remain accumulated, without counting old branch updates twice
as committed main-line updates. New SwanLab segments record parent run/attempt,
restored update and input-profile change; old points are not overwritten.

The first new complete save follows the normal five-update frequency (typically
Baseline 40 / Full 15), publishes latest only after validation, then removes the
previous owned generation. A failed migration does not rewrite/delete latest.

## Finite real checks and evidence files

Each formal allocation performs full checkpoint verification and the following
checks before advancing beyond its restored update:

- Baseline: read only the last incomplete update-40 collection, deduplicate saved
  active prefixes, replay that finite prefix with the actual native worker, and
  compare each saved input/page/reward/terminal receipt. Require reconstruction
  of the exact 9910-token input with the actual tokenizer, then verify its fitted
  input <=7000. Missing post-done actions are never invented. If the real trigger
  is not reproducible from retained evidence, stop with diagnostics, not a claim
  that a synthetic sample was the real case.
- Full: use the checkpoint ID recorded in the original failed initialization,
  verify immutable native replay, compare preview/installation and task/history,
  then execute one new admissible public action, outside training/statistics.
- Acceptance restores controller Python/NumPy/Torch RNGs and adds zero optimizer
  updates or training samples. Normal collections reset/rebind the worker state.

Per-attempt receipts: `migration-permit.json`, `restored-state.json`,
`resume-lineage.json`, `webshop-real-acceptance.json`, then ordinary rollout,
optimization and checkpoint records. Source/target signatures, checkpoint hashes,
data cursor, RNG checks and archive counts are explicit. Monitoring separates
completed policy, checkpointed and attempted update counters.

Submission directory: `analysis/beacon-omr/repairs/webshop-input-20261020/`.
`inventory.json` preserves initial metadata; `inventory-last.json` holds retries.
`old-child.json`, `requests/{baseline,full}.json` and `jobs.json` bind the exact
scope and accepted IDs. Quota retry retains accepted individual jobs, including
when that arm has already advanced. New IDs are pending the user's Tunnel command.

## Local validation and remaining acceptance

104 combined CPU regression cases passed, plus the direct frozen-source composer
comparison, for 105 distinct cases. Coverage includes actual production replay,
rendering, input-budget, checkpoint loading and migration gates with controlled
native/Ray/tokenizer fixtures; candidate/archive, identity, bounded writer,
full-batch, real synchronization receipt guards and latest checkpoint regressions
remain passing. Slurm script syntax passes. Protected code blobs match both
reported sources (Full additionally checks its unchanged OMR formula modules).

These are local fixtures and source/config checks, not actual HF tokenizer,
Orion native replay, full rank-file hashing, GPU optimizer loading or resumed
training receipts. Await the two formal allocations' acceptance, then at least
Baseline crossing the real input case, Full producing a restart collection and
optimizer update, and subsequent ordinary latest checkpoints. Do not substitute
fixture success or a failed-health step for those results.
