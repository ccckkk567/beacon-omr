#!/bin/bash
# Lightweight login-node submission; all simulator/SDK checks run on a GPU node.
set -euo pipefail
ROOT="${1:?Pass the assigned work root}"
PREVIOUS="${2:?Pass the accepted WebShop preparation report.json}"
REV="${3:?Pass a full published commit hash}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
test -r "$PREVIOUS"
test -r "$(dirname "$PREVIOUS")/site.json"
SOURCE="$ROOT/code/beacon-omr"
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
test "$(git -C "$SOURCE" remote get-url origin)" = https://github.com/ccckkk567/beacon-omr.git
git -C "$SOURCE" fetch origin "$BRANCH"
git -C "$SOURCE" merge-base --is-ancestor "$REV" FETCH_HEAD
SNAPSHOT="$ROOT/code/beacon-omr-runs/continuation-$REV"
mkdir -p "$ROOT/code/beacon-omr-runs" "$ROOT/logs/beacon-omr"
if [ -e "$SNAPSHOT" ]; then
  test "$(git -C "$SNAPSHOT" rev-parse --show-toplevel)" = "$SNAPSHOT"
  test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$REV"
else
  git -C "$SOURCE" worktree add --detach "$SNAPSHOT" "$REV"
fi
test -z "$(git -C "$SNAPSHOT" status --porcelain)"
JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --cpus-per-task=8 --mem=48G --time=00:40:00 \
  --job-name=omr-continuation --chdir="$ROOT" \
  --output="$ROOT/logs/beacon-omr/continuation-%j.out" \
  "$SNAPSHOT/experiments/omr/continuation.sbatch" "$SNAPSHOT" "$ROOT" "$PREVIOUS" "$REV")
JOB="${JOB%%;*}"
printf 'Continuation check job: %s\nAfter completion return this log:\ncat %q\n' \
  "$JOB" "$ROOT/logs/beacon-omr/continuation-$JOB.out"
squeue -j "$JOB" || true
