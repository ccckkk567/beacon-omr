#!/bin/bash
# Lightweight Git clone/check on login; environment inspection runs on GPU.
set -euo pipefail
ROOT="${1:?Pass the assigned work root}"
EXTRA="${2:?Pass the overflow model/data root}"
REPO="$ROOT/code/beacon-omr"
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
URL=https://github.com/ccckkk567/beacon-omr.git
test -d "$ROOT"
test -w "$ROOT"
mkdir -p "$ROOT/code" "$ROOT/tmp" "$ROOT/logs/beacon-omr" "$ROOT/evidence/beacon-omr"
if [ -e "$REPO" ]; then
  test -d "$REPO/.git"
  test "$(git -C "$REPO" remote get-url origin)" = "$URL"
  test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
  test -z "$(git -C "$REPO" status --porcelain)"
else
  git clone --branch "$BRANCH" --single-branch "$URL" "$REPO"
fi
test -f "$REPO/experiments/omr/inventory.sbatch"
printf 'Repository commit: '
git -C "$REPO" rev-parse HEAD
PROBE_JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --gres=gpu:2 --job-name=omr-inventory --cpus-per-task=8 --mem=32G --time=00:30:00 \
  --chdir="$ROOT" --output="$ROOT/logs/beacon-omr/inventory-%j.out" \
  "$REPO/experiments/omr/inventory.sbatch" "$REPO" "$ROOT" "$EXTRA")
PROBE_JOB="${PROBE_JOB%%;*}"
printf 'GPU inventory job: %s\n' "$PROBE_JOB"
printf 'After the GPU job finishes, return this log:\ncat %q\n' "$ROOT/logs/beacon-omr/inventory-$PROBE_JOB.out"
squeue -u "$USER"
