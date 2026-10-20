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
ulimit -c 0
OUT="$ROOT/analysis/beacon-omr/audits/exposure-20261020-${SLURM_JOB_ID}"
date -Is
squeue -r -u "$USER" -o "%.18i %.34j %.12P %.14T %.12M %.12l %.35R"
printf '\nSTAGE audit-cpu-runtime\n'
hostname
VENDOR=$(awk -F: '/vendor_id/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/cpuinfo)
printf 'CPU vendor: %s\n' "$VENDOR"
case "$VENDOR" in
  AuthenticAMD) PYTHON_MODULE=python/3.12.5-amd ;;
  GenuineIntel) PYTHON_MODULE=python/3.12.5 ;;
  *) printf 'Unrecognized CPU vendor; preserve training environment and inspect.\n'; exit 2 ;;
esac
module load "$PYTHON_MODULE"
AUDIT_PY=$(command -v python3)
printf 'CPU Python module: %s; executable: %s\n' "$PYTHON_MODULE" "$AUDIT_PY"
srun --ntasks=1 "$AUDIT_PY" -I -S -B \
  "$CONTROL/experiments/omr/audit_cpu_entry.py" --root "$ROOT" --output "$OUT"
printf '\nAudit output: %s\n' "$OUT"
for REPORT in restart_opportunity_audit.md restart_group_trace_audit.md baseline_full_eval_comparison.md; do
  printf '\n===== %s =====\n' "$REPORT"
  cat "$OUT/$REPORT"
done
