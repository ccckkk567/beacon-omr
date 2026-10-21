#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set +x
set -euo pipefail
umask 077
REPO="$1"
ROOT="$2"
REV="$3"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
VENDOR=$(awk -F: '/vendor_id/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' /proc/cpuinfo)
case "$VENDOR" in
  AuthenticAMD) module load python/3.12.5-amd ;;
  GenuineIntel) module load python/3.12.5 ;;
  *) echo 'Unknown CPU vendor'; exit 2 ;;
esac
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B "$REPO/experiments/omr/prepare_demo10.py" \
  --root "$ROOT" --revision "$REV" --submit
