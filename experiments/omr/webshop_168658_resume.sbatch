#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@1200
set +x
set -euo pipefail
CONTROL="$1"
ROOT="$2"
CONTROL_REV="$3"
export PYTHONDONTWRITEBYTECODE=1
TRAIN_REV=df98029d703d64a1931dcebf0f626fb1e468af9b
TRAINING="$ROOT/code/beacon-omr-runs/train-$TRAIN_REV"
if python3 -I -S -B "$CONTROL/experiments/omr/submit_webshop_168658_resume.py" guard \
  --root "$ROOT" --controller-revision "$CONTROL_REV"; then
  # Resolve baseline/full from the durable parent receipt, not a truncated squeue name.
  METHOD=$(python3 -I -S -B -c 'import json,sys; s=json.load(open(sys.argv[1])); print(next(k.split("/")[1] for k,v in s["jobs"].items() if v.get("job_id")=="168658"))' \
    "$ROOT/analysis/beacon-omr/repairs/webshop-fresh-20261020/jobs.json")
  exec bash "$TRAINING/experiments/omr/train.sbatch" "$TRAINING" "$ROOT" \
    "$ROOT/evidence/beacon-omr/webshop-prepare-165873/report.json" "$TRAIN_REV" \
    input-v2-fresh-20261020 webshop "$METHOD"
else
  RESULT=$?
  if [ "$RESULT" -eq 10 ]; then
    printf 'Training and final evaluation already complete; continuation skipped.\n'
    exit 0
  fi
  exit "$RESULT"
fi
