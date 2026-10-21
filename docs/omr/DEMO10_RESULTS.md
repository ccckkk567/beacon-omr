# Demo10 results and acceptance

Status: implementation and CPU contract validation; HPC preparation, six actual
submission IDs, GPU/native acceptance and training outcomes are not yet returned.
No scores, actual subset counts, hashes or native8:8 receipts are invented here.

Local validation on2026-10-21:72 relevant tests passed in one combined run;
after adding the explicit Demo protocol-checkpoint round trip, the affected
26-test subset passed (73 distinct tests covered in total). Both new Slurm
scripts passed Bash syntax checks and `git diff --check` passed. Relative to
aa891ae, native environment, rollout-loop, worker and advantage implementation
trees are unchanged; the trainer has only the Demo update2-save gate described
in the implementation notes.

Actual evidence will be under `$ROOT/analysis/beacon-omr/`:

- `manifests/demo10-active-v1/integration_manifest.json` and benchmark reports.
- `training-submissions/demo10-active-v1/{prepare-jobs,jobs,dry-run}.json`.
- Six `omr-demo10-*-s0-20261021/` run directories, with progress/checkpoints,
  per-attempt launch/resolved config, rollouts, optimization, evaluation and
  SwanLab transport/metric records.

Local tests exercise exact stratified selection, immutable publication and
parent drift, sparse categories, frozen mask/conditional task probabilities,
fixed budgets with empty/one/many archive tasks, forced branch selection,
128-worker production collector integration, group fallback attribution,
next-update visibility, independent group identities and reproducible resume,
evaluation isolation, policy-update monitoring and idempotent partial submission.
The three native simulator interfaces, FSDP/vLLM and remote upload are checked by
the actual formal allocations; no standalone GPU smoke jobs are added.
