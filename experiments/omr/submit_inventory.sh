#!/bin/bash
# Lightweight login-node submission only. Clone on CPU, inspect on GPU.
set -euo pipefail
ROOT="${1:?Pass the assigned work root}"
EXTRA="${2:?Pass the overflow model/data root}"
REPO="$ROOT/code/beacon-omr"
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
URL=https://github.com/ccckkk567/beacon-omr.git
test -d "$ROOT"
test -w "$ROOT"
mkdir -p "$ROOT/code" "$ROOT/tmp" "$ROOT/logs/beacon-omr" "$ROOT/evidence/beacon-omr"
printf -v qrepo '%q' "$REPO"
printf -v qroot '%q' "$ROOT"
printf -v qextra '%q' "$EXTRA"
printf -v qscript '%q' "$REPO/experiments/omr/inventory.sbatch"
clone_command="set -euo pipefail
if [ -e $qrepo ]; then
  test -d $qrepo/.git
  test \"\$(git -C $qrepo remote get-url origin)\" = '$URL'
  test \"\$(git -C $qrepo branch --show-current)\" = '$BRANCH'
  test -z \"\$(git -C $qrepo status --porcelain)\"
  test -f $qscript
  git -C $qrepo rev-parse HEAD
else
  git clone --branch '$BRANCH' --single-branch '$URL' $qrepo
fi"
CLONE_SCRIPT=$(mktemp "$ROOT/tmp/omr-clone-XXXXXXXX.sbatch")
PROBE_SCRIPT=$(mktemp "$ROOT/tmp/omr-inventory-XXXXXXXX.sbatch")
printf '#!/bin/bash\n%s\n' "$clone_command" > "$CLONE_SCRIPT"
printf '#!/bin/bash\nset -euo pipefail\nexec bash %s %s %s %s\n' \
  "$qscript" "$qrepo" "$qroot" "$qextra" > "$PROBE_SCRIPT"
CLONE_JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=ioqueue \
  --job-name=omr-clone --cpus-per-task=2 --mem=4G --time=00:20:00 \
  --chdir="$ROOT" --output="$ROOT/logs/beacon-omr/clone-%j.out" \
  "$CLONE_SCRIPT")
CLONE_JOB="${CLONE_JOB%%;*}"
PROBE_JOB=$(sbatch --parsable --account=prjsnnn0000269 --partition=testqueue \
  --gres=gpu:2 --job-name=omr-inventory --cpus-per-task=8 --mem=32G --time=00:30:00 \
  --dependency="afterok:$CLONE_JOB" --kill-on-invalid-dep=yes \
  --chdir="$ROOT" --output="$ROOT/logs/beacon-omr/inventory-%j.out" \
  "$PROBE_SCRIPT")
PROBE_JOB="${PROBE_JOB%%;*}"
printf 'Clone/check job: %s\nGPU inventory job: %s\n' "$CLONE_JOB" "$PROBE_JOB"
printf 'After the GPU job finishes, return this log:\ncat %q\n' "$ROOT/logs/beacon-omr/inventory-$PROBE_JOB.out"
printf 'If the clone/check job fails, return:\ncat %q\n' "$ROOT/logs/beacon-omr/clone-$CLONE_JOB.out"
squeue -u "$USER"
