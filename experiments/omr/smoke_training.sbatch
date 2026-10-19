#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@300
set +x
set -euo pipefail
CONTROL="$1"
ROOT="$2"
REV="$3"
BENCHES=(alfworld webshop scienceworld)
BENCH="${BENCHES[$((SLURM_ARRAY_TASK_ID % 3))]}"
METHOD=baseline
if (( SLURM_ARRAY_TASK_ID >= 3 )); then METHOD=full; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="/tmp/omr-smoke-${SLURM_JOB_ID}-${UID}"
umask 077
mkdir -p "$TMPDIR"
ATTEMPT="$ROOT/analysis/beacon-omr/smoke-${SLURM_ARRAY_JOB_ID}/$BENCH-$METHOD/attempts/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
trap 'mkdir -p "$ATTEMPT"; touch "$ATTEMPT/STOP"' USR1
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B \
  "$CONTROL/experiments/omr/smoke_training.py" launch --root "$ROOT" --controller-revision "$REV" &
CHILD=$!
set +e
wait "$CHILD"
RESULT=$?
while kill -0 "$CHILD" 2>/dev/null; do
  wait "$CHILD"
  RESULT=$?
done
exit "$RESULT"
