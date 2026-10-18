#!/bin/bash
# Submit the follow-up GPU checks; do not activate or mutate existing environments.
set -euo pipefail
fail() { printf 'Preflight submission stopped: %s\n' "$1" >&2; exit 1; }
ROOT="${1:?Pass the assigned work root}"
EXTRA="${2:?Pass the overflow model/data root}"
INVENTORY="${3:?Pass the completed inventory JSON path}"
SCOPE="${4:-all}"
case "$SCOPE" in all|webshop) ;; *) printf 'Scope must be all or webshop\n' >&2; exit 2 ;; esac
REPO="$ROOT/code/beacon-omr"
printf 'Checking preflight submission prerequisites (scope=%s)...\n' "$SCOPE"
test -r "$INVENTORY" || fail "Inventory JSON is missing or unreadable: $INVENTORY"
test -d "$REPO/.git" || fail "Expected Git checkout is missing: $REPO"
ORIGIN=$(git -C "$REPO" remote get-url origin) || fail 'Cannot read the origin remote'
test "$ORIGIN" = https://github.com/ccckkk567/beacon-omr.git || fail 'Origin does not match the BEACON OMR repository'
BRANCH=$(git -C "$REPO" branch --show-current) || fail 'Cannot read the current branch'
test "$BRANCH" = exp/beacon-online-milestone-reuse-h200x2-20261018 || fail "Unexpected branch: $BRANCH"
DIRTY=$(git -C "$REPO" status --porcelain) || fail 'Cannot inspect checkout status'
if [ -n "$DIRTY" ]; then
  printf 'Checkout changes (paths only):\n%s\n' "$DIRTY" >&2
  fail 'Checkout has modified or untracked files; inspect these paths before submission'
fi
command -v sbatch >/dev/null 2>&1 || fail 'sbatch is not available in this shell'
mkdir -p "$ROOT/logs/beacon-omr"
if ! JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --gres=gpu:h200:2 --job-name=omr-preflight --cpus-per-task=8 --mem=48G --time=01:00:00 \
  --chdir="$ROOT" --output="$ROOT/logs/beacon-omr/preflight-%j.out" \
  "$REPO/experiments/omr/preflight.sbatch" "$REPO" "$ROOT" "$EXTRA" "$INVENTORY" "$SCOPE"); then
  fail 'sbatch failed; see the Slurm error above'
fi
JOB="${JOB%%;*}"
printf 'GPU preflight job: %s\nAfter completion return this log:\ncat %q\n' \
  "$JOB" "$ROOT/logs/beacon-omr/preflight-$JOB.out"
squeue -j "$JOB" || printf 'Job %s was submitted; its queue status could not be read.\n' "$JOB"
