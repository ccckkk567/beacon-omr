#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@1200
set +x
set -euo pipefail
CONTROL="$1"
TRAINING="$2"
ROOT="$3"
REPORT="$4"
TRAIN_REV="$5"
PARENT="$6"
CONTROL_REV="$7"
RUN_TAG="${8:-}"
test "$(git -C "$CONTROL" rev-parse HEAD)" = "$CONTROL_REV"
test -z "$(git -C "$CONTROL" status --porcelain)"
: "${SLURM_ARRAY_TASK_ID:?}"
if python3 -I -S -B "$CONTROL/experiments/omr/resume_chain.py" guard \
  --root "$ROOT" --parent-array "$PARENT" --training-revision "$TRAIN_REV" \
  --index "$SLURM_ARRAY_TASK_ID" --run-tag "$RUN_TAG"; then
  # exec preserves the batch PID: the frozen trainer owns the USR1 stop trap.
  exec bash "$TRAINING/experiments/omr/train.sbatch" "$TRAINING" "$ROOT" "$REPORT" "$TRAIN_REV" "$RUN_TAG"
else
  RESULT=$?
  if [ "$RESULT" -eq 10 ]; then
    printf 'Training and final evaluation already complete; continuation skipped.\n'
    exit 0
  fi
  exit "$RESULT"
fi
