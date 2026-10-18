#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set +x
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
REPO="$1"
ROOT="$2"
PREVIOUS="$3"
REV="$4"
test "$(git -C "$REPO" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$REPO" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export TMPDIR="/tmp/omr-check-${SLURM_JOB_ID}-${UID}"
umask 077
mkdir -p "$TMPDIR"
if type module >/dev/null 2>&1; then
  module list 2>&1 || true
fi
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B \
  "$REPO/experiments/omr/validate_continuation.py" --work-root "$ROOT" \
  --prepared-report "$PREVIOUS"
