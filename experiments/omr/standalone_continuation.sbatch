#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@1200
set +x
set -euo pipefail
CONTROL="$1"
ROOT="$2"
PARENT="$3"
CONTROL_REV="$4"
TRAIN_REV=ddae412e6195d2f86dace7f5ce2630d155ac0c01
TRAINING="$ROOT/code/beacon-omr-runs/train-$TRAIN_REV"
case "$PARENT" in
  167697) BENCH=alfworld; METHOD=full ;;
  167699) BENCH=scienceworld; METHOD=baseline ;;
  167700) BENCH=scienceworld; METHOD=full ;;
  *) exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
if python3 -I -S -B "$CONTROL/experiments/omr/submit_running_continuations.py" guard \
  --root "$ROOT" --parent "$PARENT" --controller-revision "$CONTROL_REV"; then
  exec bash "$TRAINING/experiments/omr/train.sbatch" "$TRAINING" "$ROOT" \
    "$ROOT/evidence/beacon-omr/webshop-prepare-165873/report.json" "$TRAIN_REV" \
    fullbatch-20261019 "$BENCH" "$METHOD"
else
  RESULT=$?
  if [ "$RESULT" -eq 10 ]; then
    printf 'Training and final evaluation already complete; continuation skipped.\n'
    exit 0
  fi
  exit "$RESULT"
fi
