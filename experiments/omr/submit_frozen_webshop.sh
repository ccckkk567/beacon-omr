#!/bin/bash
# Light login-node Git operations, followed by a single-GPU preparation job.
set -euo pipefail
fail() { printf 'WebShop submission stopped: %s\n' "$1" >&2; exit 1; }
ROOT="${1:?Pass the assigned work root}"
EXTRA="${2:?Pass the overflow data root}"
PREVIOUS="${3:?Pass preflight-165837/report.json}"
REV="${4:?Pass a full published commit hash}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || fail 'Revision must be a full commit hash'
test -r "$PREVIOUS" || fail "Preflight report is not readable: $PREVIOUS"
SOURCE="$ROOT/code/beacon-omr"
URL=https://github.com/ccckkk567/beacon-omr.git
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
test "$(git -C "$SOURCE" remote get-url origin)" = "$URL" || fail 'Unexpected origin repository'
git -C "$SOURCE" fetch origin "$BRANCH"
git -C "$SOURCE" cat-file -e "$REV^{commit}" || fail 'Requested commit was not fetched'
git -C "$SOURCE" merge-base --is-ancestor "$REV" FETCH_HEAD || fail 'Requested commit is outside the experiment branch'
SNAPSHOT="$ROOT/code/beacon-omr-runs/webshop-$REV"
mkdir -p "$ROOT/code/beacon-omr-runs"
if [ -e "$SNAPSHOT" ]; then
  test "$(git -C "$SNAPSHOT" rev-parse --show-toplevel)" = "$SNAPSHOT" || fail 'Existing snapshot is not its own checkout'
  test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$REV" || fail 'Existing snapshot has a different commit'
else
  git -C "$SOURCE" worktree add --detach "$SNAPSHOT" "$REV"
fi
test -z "$(git -C "$SNAPSHOT" status --porcelain)" || fail 'Frozen snapshot has local changes'
command -v sbatch >/dev/null 2>&1 || fail 'sbatch is unavailable'
mkdir -p "$ROOT/logs/beacon-omr"
JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:1 --cpus-per-task=8 --mem=48G --time=02:00:00 \
  --job-name=omr-webshop-prepare --chdir="$ROOT" \
  --output="$ROOT/logs/beacon-omr/webshop-prepare-%j.out" \
  "$SNAPSHOT/experiments/omr/prepare_webshop.sbatch" "$SNAPSHOT" "$ROOT" "$EXTRA" "$PREVIOUS" "$REV")
JOB="${JOB%%;*}"
printf 'WebShop preparation job: %s\nAfter completion return this log:\ncat %q\n' \
  "$JOB" "$ROOT/logs/beacon-omr/webshop-prepare-$JOB.out"
squeue -j "$JOB" || true
