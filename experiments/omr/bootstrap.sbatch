#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this pure-download task through sbatch on ioqueue}"
WORK_ROOT="$1"
REPO="$WORK_ROOT/code/beacon-omr"
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
test ! -e "$REPO" || { echo "Refusing to overwrite existing checkout: $REPO" >&2; exit 2; }
mkdir -p "$WORK_ROOT/code"
git clone --branch "$BRANCH" --single-branch https://github.com/ccckkk567/beacon-omr.git "$REPO"
git -C "$REPO" remote add upstream https://github.com/ZJU-REAL/BEACON.git
git -C "$REPO" rev-parse HEAD
echo "REPO=$REPO"
