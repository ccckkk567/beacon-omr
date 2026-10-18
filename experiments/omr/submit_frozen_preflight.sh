#!/bin/bash
# Fetch committed code into a separate worktree; preserve the user's checkout.
set -euo pipefail
fail() { printf 'Frozen preflight stopped: %s\n' "$1" >&2; exit 1; }
ROOT="${1:?Pass the assigned work root}"
EXTRA="${2:?Pass the overflow model/data root}"
INVENTORY="${3:?Pass the inventory JSON}"
REV="${4:?Pass the full published commit hash}"
SCOPE="${5:-all}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]] || fail 'Revision must be a full commit hash'
case "$SCOPE" in all|webshop) ;; *) fail 'Scope must be all or webshop' ;; esac
test -r "$INVENTORY" || fail "Inventory JSON is missing or unreadable: $INVENTORY"
SOURCE="$ROOT/code/beacon-omr"
URL=https://github.com/ccckkk567/beacon-omr.git
BRANCH=exp/beacon-online-milestone-reuse-h200x2-20261018
test "$(git -C "$SOURCE" remote get-url origin)" = "$URL" || fail 'Unexpected origin repository'
# Fetch updates Git objects/remote refs, not the original working files or branch.
git -C "$SOURCE" fetch origin "$BRANCH"
git -C "$SOURCE" cat-file -e "$REV^{commit}" || fail 'Requested commit was not fetched'
git -C "$SOURCE" merge-base --is-ancestor "$REV" FETCH_HEAD || fail 'Requested commit is not on the published experiment branch'
SNAPSHOT="$ROOT/code/beacon-omr-runs/preflight-$REV"
mkdir -p "$ROOT/code/beacon-omr-runs"
if [ -e "$SNAPSHOT" ]; then
  test "$(git -C "$SNAPSHOT" rev-parse --show-toplevel)" = "$SNAPSHOT" || fail 'Existing snapshot path is not its own checkout'
  test "$(git -C "$SNAPSHOT" rev-parse HEAD)" = "$REV" || fail 'Existing snapshot has a different commit'
else
  git -C "$SOURCE" worktree add --detach "$SNAPSHOT" "$REV"
fi
printf 'Preflight source: %s\nFrozen commit: %s\n' "$SNAPSHOT" "$REV"
OMR_SOURCE_REPO="$SNAPSHOT" OMR_SOURCE_COMMIT="$REV" \
  bash "$SNAPSHOT/experiments/omr/submit_preflight.sh" "$ROOT" "$EXTRA" "$INVENTORY" "$SCOPE"
