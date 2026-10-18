#!/bin/bash
# Login node: Git and submission only. Array elements train independently.
set -euo pipefail
ROOT="${1:?Pass the assigned work root}"
PREVIOUS="${2:?Pass the accepted preparation report}"
REV="${3:?Pass the full published source revision}"
SELECT="${4:-all}"
METHOD="${5:-both}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
case "$SELECT" in
  all) INDEX=all ;;
  alfworld) INDEX=0 ;;
  webshop) INDEX=1 ;;
  scienceworld) INDEX=2 ;;
  *) printf 'Unsupported benchmark: %s\n' "$SELECT" >&2; exit 2 ;;
esac
case "$METHOD" in
  baseline) if [ "$INDEX" = all ]; then ARRAY='0-2%3'; else ARRAY="$INDEX"; fi ;;
  full) if [ "$INDEX" = all ]; then ARRAY='3-5%3'; else ARRAY="$((INDEX + 3))"; fi ;;
  both) if [ "$INDEX" = all ]; then ARRAY='0-5%6'; else ARRAY="$INDEX,$((INDEX + 3))%2"; fi ;;
  *) printf 'Unsupported method: %s\n' "$METHOD" >&2; exit 2 ;;
esac
SOURCE="$ROOT/code/beacon-omr"
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
test -r "$PREVIOUS"
test -r "$(dirname "$PREVIOUS")/site.json"
test "$(git -C "$SOURCE" remote get-url origin)" = https://github.com/ccckkk567/beacon-omr.git
git -C "$SOURCE" fetch origin "$BRANCH"
git -C "$SOURCE" merge-base --is-ancestor "$REV" FETCH_HEAD
SNAPSHOT="$ROOT/code/beacon-omr-runs/train-$REV"
mkdir -p "$ROOT/code/beacon-omr-runs" "$ROOT/logs/beacon-omr"
if [ -e "$SNAPSHOT" ]; then
  test "$(git -C "$SNAPSHOT" rev-parse --show-toplevel)" = "$SNAPSHOT"
  test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$REV"
else
  git -C "$SOURCE" worktree add --detach "$SNAPSHOT" "$REV"
fi
test -z "$(git -C "$SNAPSHOT" status --porcelain)"
JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --nodes=1 --ntasks=1 --gres=gpu:h200:2 --cpus-per-task=32 --mem=256G --time=18:00:00 \
  --array="$ARRAY" --job-name=omr-training --chdir="$ROOT" \
  --output="$ROOT/logs/beacon-omr/train-%A_%a.out" \
  "$SNAPSHOT/experiments/omr/train.sbatch" "$SNAPSHOT" "$ROOT" "$PREVIOUS" "$REV")
JOB="${JOB%%;*}"
printf 'Training array: %s (0/1/2=ALF/Web/Sci baseline; 3/4/5=ALF/Web/Sci full)\n' "$JOB"
printf 'Logs: %s/logs/beacon-omr/train-%s_<index>.out\n' "$ROOT" "$JOB"
squeue -j "$JOB" || true
