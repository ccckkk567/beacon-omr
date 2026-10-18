#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set +x
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
REPO="$1"
ROOT="$2"
EXTRA="$3"
PREVIOUS="$4"
REV="$5"
test "$(git -C "$REPO" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$REPO" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONNOUSERSITE=1
# Keep Ray's Unix socket paths short; assets and reports are on persistent storage.
export TMPDIR="/tmp/omr-ws-${SLURM_JOB_ID}-${UID}"
umask 077
mkdir -p "$TMPDIR"
if type module >/dev/null 2>&1; then
  module list 2>&1 || true
fi
command -v python3 >/dev/null 2>&1
command -v java >/dev/null 2>&1
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B \
  "$REPO/experiments/omr/prepare_webshop.py" --work-root "$ROOT" \
  --extra-storage "$EXTRA" --previous-report "$PREVIOUS"
