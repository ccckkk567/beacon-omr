#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@1200
set +x
set -euo pipefail
REPO="$1"
ROOT="$2"
REV="$3"
BENCH="$4"
METHOD="$5"
MODE="${6:-fresh}"
test "$(git -C "$REPO" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$REPO" status --porcelain)"
case "$BENCH" in alfworld|scienceworld|webshop) ;; *) exit 2 ;; esac
case "$METHOD" in baseline|active_full) ;; *) exit 2 ;; esac
EXTRA=()
case "$MODE" in fresh) ;; resume) EXTRA=(--demo-resume) ;; *) exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR="/tmp/omr-demo10-${SLURM_JOB_ID}-${UID}"
umask 077
mkdir -p "$TMPDIR"
NAME="omr-demo10-${BENCH}-${METHOD//_/-}-s0-20261021"
ATTEMPT="$ROOT/analysis/beacon-omr/$NAME/attempts/${SLURM_JOB_ID}-single"
trap 'mkdir -p "$ATTEMPT"; touch "$ATTEMPT/STOP"' USR1
cd "$REPO"
srun --ntasks=1 --kill-on-bad-exit=1 python3 -I -S -B "$REPO/experiments/omr/launch_training.py" \
  --root "$ROOT" --prepared-report "$ROOT/evidence/beacon-omr/webshop-prepare-165873/report.json" \
  --benchmark "$BENCH" --method "$METHOD" --demo10 "${EXTRA[@]}" &
CHILD=$!
set +e
wait "$CHILD"
RESULT=$?
while kill -0 "$CHILD" 2>/dev/null; do wait "$CHILD"; RESULT=$?; done
exit "$RESULT"
