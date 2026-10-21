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
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  # Compatibility for existing frozen orchestration; new submissions are single jobs.
  BENCH="${BENCHES[$((SLURM_ARRAY_TASK_ID % 3))]}"
  METHOD=baseline
  if (( SLURM_ARRAY_TASK_ID >= 3 )); then METHOD=full; fi
else
  BENCH="${6:?explicit benchmark required for a standalone job}"
  METHOD="${7:?explicit method required for a standalone job}"
fi
case "$BENCH" in alfworld|webshop|scienceworld) ;; *) exit 2 ;; esac
case "$METHOD" in baseline|full) ;; *) exit 2 ;; esac
ATTEMPT_SLOT="${SLURM_ARRAY_TASK_ID:-single}"
RESUME_ARGS=()
if [ -n "${8:-}" ]; then
  RESUME_ARGS=(--scienceworld-baseline-resume-request "$8")
fi
test "$(git -C "$REPO" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$REPO" status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="/tmp/omr-train-${SLURM_JOB_ID}-${UID}"
umask 077
mkdir -p "$TMPDIR"
RUN_ID=$(python3 -I -S -B "$REPO/experiments/omr/run_paths.py" "$BENCH" "$METHOD" "$RUN_TAG")
ATTEMPT="$ROOT/analysis/beacon-omr/$RUN_ID/attempts/${SLURM_JOB_ID}-${ATTEMPT_SLOT}"
trap 'mkdir -p "$ATTEMPT"; touch "$ATTEMPT/STOP"' USR1
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B \
  "$REPO/experiments/omr/launch_training.py" --root "$ROOT" \
  --prepared-report "$PREVIOUS" --benchmark "$BENCH" --method "$METHOD" --run-tag "$RUN_TAG" "${RESUME_ARGS[@]}" &
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
