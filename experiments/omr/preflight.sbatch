#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through sbatch}"
REPO="$1"
ROOT="$2"
EXTRA="$3"
INVENTORY="$4"
SCOPE="${5:-all}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
mkdir -p "$ROOT/tmp/omr-preflight-$SLURM_JOB_ID"
export TMPDIR="$ROOT/tmp/omr-preflight-$SLURM_JOB_ID"
if type module >/dev/null 2>&1; then
    module list 2>&1 || true
    if ! command -v python3 >/dev/null 2>&1; then
        module load miniforge/24.11.3-2
    fi
fi
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -m experiments.omr.preflight \
    --work-root "$ROOT" --extra-storage "$EXTRA" --inventory "$INVENTORY" --scope "$SCOPE"
