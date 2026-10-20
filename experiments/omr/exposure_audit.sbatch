#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set +x
set -euo pipefail
CONTROL="$1"
ROOT="$2"
REV="$3"
test "$(git -C "$CONTROL" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$CONTROL" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
OUT="$ROOT/analysis/beacon-omr/audits/exposure-20261020-${SLURM_JOB_ID}"
date -Is
squeue -r -u "$USER" -o "%.18i %.34j %.12P %.14T %.12M %.12l %.35R"
srun --ntasks=1 "$ROOT/envs/gigpo-scienceworld-py312-v1/bin/python" -I -B \
  "$CONTROL/experiments/omr/exposure_audit.py" --root "$ROOT" --output "$OUT"
printf '\nAudit output: %s\n' "$OUT"
for REPORT in restart_opportunity_audit.md restart_group_trace_audit.md baseline_full_eval_comparison.md; do
  printf '\n===== %s =====\n' "$REPORT"
  cat "$OUT/$REPORT"
done
