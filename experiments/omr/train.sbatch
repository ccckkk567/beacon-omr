#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@1200
set +x
set -euo pipefail
REPO="$1"
ROOT="$2"
PREVIOUS="$3"
REV="$4"
RUN_TAG="${5:-}"
BENCHES=(alfworld webshop scienceworld)
: "${SLURM_ARRAY_TASK_ID:?}"
BENCH="${BENCHES[$((SLURM_ARRAY_TASK_ID % 3))]}"
METHOD=baseline
if (( SLURM_ARRAY_TASK_ID >= 3 )); then METHOD=full; fi
test "$(git -C "$REPO" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$REPO" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="/tmp/omr-train-${SLURM_JOB_ID}-${UID}"
umask 077
mkdir -p "$TMPDIR"
RUN_ID=$(python3 -I -S -B "$REPO/experiments/omr/run_paths.py" "$BENCH" "$METHOD" "$RUN_TAG")
ATTEMPT="$ROOT/analysis/beacon-omr/$RUN_ID/attempts/${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
trap 'mkdir -p "$ATTEMPT"; touch "$ATTEMPT/STOP"' USR1
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B \
  "$REPO/experiments/omr/launch_training.py" --root "$ROOT" \
  --prepared-report "$PREVIOUS" --benchmark "$BENCH" --method "$METHOD" --run-tag "$RUN_TAG" &
CHILD=$!
# A signal interrupts wait; continue waiting for this owned training process.
set +e
wait "$CHILD"
RESULT=$?
while kill -0 "$CHILD" 2>/dev/null; do
  wait "$CHILD"
  RESULT=$?
done
exit "$RESULT"
