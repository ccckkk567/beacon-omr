#!/bin/bash
# Submit the follow-up GPU checks; do not activate or mutate existing environments.
set -euo pipefail
ROOT="${1:?Pass the assigned work root}"
EXTRA="${2:?Pass the overflow model/data root}"
INVENTORY="${3:?Pass the completed inventory JSON path}"
REPO="$ROOT/code/beacon-omr"
test -r "$INVENTORY"
test "$(git -C "$REPO" remote get-url origin)" = https://github.com/ccckkk567/beacon-omr.git
test "$(git -C "$REPO" branch --show-current)" = exp/beacon-online-milestone-reuse-h200x2-20261018
test -z "$(git -C "$REPO" status --porcelain)"
mkdir -p "$ROOT/logs/beacon-omr"
JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --gres=gpu:h200:2 --job-name=omr-preflight --cpus-per-task=8 --mem=48G --time=00:40:00 \
  --chdir="$ROOT" --output="$ROOT/logs/beacon-omr/preflight-%j.out" \
  "$REPO/experiments/omr/preflight.sbatch" "$REPO" "$ROOT" "$EXTRA" "$INVENTORY")
JOB="${JOB%%;*}"
printf 'GPU preflight job: %s\nAfter completion return this log:\ncat %q\n' \
  "$JOB" "$ROOT/logs/beacon-omr/preflight-$JOB.out"
squeue -j "$JOB"
