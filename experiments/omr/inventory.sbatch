#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through sbatch}"
REPO="$1"
WORK_ROOT="$2"
EXTRA_STORAGE="$3"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if type module >/dev/null 2>&1; then
    module list 2>&1 || true
    if ! command -v python3 >/dev/null 2>&1; then
        module load miniforge/24.11.3-2
    fi
fi
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -m experiments.omr.inventory \
    --work-root "$WORK_ROOT" --extra-storage "$EXTRA_STORAGE" \
    --output "$WORK_ROOT/evidence/beacon-omr/inventory-${SLURM_JOB_ID}.json"
