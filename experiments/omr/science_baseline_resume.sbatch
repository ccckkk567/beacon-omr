#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@1200
set +x
set -euo pipefail
REPO="$1"
ROOT="$2"
REV="$3"
export PYTHONDONTWRITEBYTECODE=1
python3 -I -S -B "$REPO/experiments/omr/science_baseline_resume.py" guard \
  --root "$ROOT" --revision "$REV"
export OMR_MONITORING_RESUME_REQUEST="$ROOT/analysis/beacon-omr/resume-chains/science-baseline-requeue-20261021/monitoring-request.json"
exec bash "$REPO/experiments/omr/train.sbatch" "$REPO" "$ROOT" \
  "$ROOT/evidence/beacon-omr/webshop-prepare-165873/report.json" "$REV" \
  fullbatch-20261019 scienceworld baseline
