#!/usr/bin/env bash
# Commit catalog edits made on the server (Add Award, inbox confirms, and the
# checked_on stamps from verify_catalog.py) to the `server` branch and push it,
# so they reach main through a PR and CI validates them. Run as coach, nightly
# by scholarshipcoach-catalog-sync.timer and from deploy/update.sh:
#
#   /srv/scholarshipcoach/deploy/catalog_sync.sh
#
# The server's checkout lives on the `server` branch (bootstrap.sh), and
# update.sh merges origin/main into it. Pushes go over SSH with a repo-scoped
# deploy key that has write access, held only by coach -- never a personal
# token. bootstrap.sh creates the key and prints the public half to add under
# GitHub -> Settings -> Deploy keys, with "Allow write access" ticked.
set -euo pipefail

APP_DIR=/srv/scholarshipcoach
BRANCH=server
RECORDS_DIR=data/catalog/records/
# Snapshot delta reports are tracked in git (.gitignore), and "Rebuild snapshot"
# on the server writes one, so they are committed here rather than blocking.
DELTA_REPORTS='data/processed/changes_*.json'

cd "$APP_DIR"
fail() { echo "catalog_sync.sh: $*" >&2; exit 1; }

current=$(git symbolic-ref --quiet --short HEAD || echo "a detached HEAD")
[[ $current == "$BRANCH" ]] || fail "checkout is on $current, expected $BRANCH"

stray=$(git status --porcelain --untracked-files=all | cut -c4- | grep -v -e "^$RECORDS_DIR" -e '^data/processed/changes_[0-9]*\.json$' || true)
if [[ -n $stray ]]; then
    fail "uncommitted changes outside $RECORDS_DIR; nothing committed:
$stray"
fi

changed=$(git status --porcelain --untracked-files=all -- "$RECORDS_DIR" | cut -c4-)
if [[ -n $changed ]]; then
    echo "==> Validate catalog"
    .venv/bin/python scripts/validate_catalog.py \
        || fail "catalog validation failed; nothing committed. Fix the record in the app."
    ids=$(printf '%s\n' "$changed" | sed -e "s|^$RECORDS_DIR||" -e 's|\.json$||' | sort -u)
    count=$(printf '%s\n' "$ids" | wc -l)
    echo "==> Commit $count record(s) to $BRANCH"
    printf '    %s\n' $ids
    git add --all -- "$RECORDS_DIR"
    git commit -q -m "Catalog: server edits to $count record(s)" -m "$ids" -- "$RECORDS_DIR"
else
    echo "==> No catalog edits to commit"
fi

deltas=$(git status --porcelain --untracked-files=all -- "$DELTA_REPORTS" | cut -c4-)
if [[ -n $deltas ]]; then
    echo "==> Commit snapshot delta report(s) to $BRANCH"
    printf '    %s\n' $deltas
    git add --all -- "$DELTA_REPORTS"
    git commit -q -m "Snapshot delta report(s) from the server" -- "$DELTA_REPORTS"
fi

# Push only records the server changed that main does not already have: after a
# squash merge the server's commits are not ancestors of main, and pushing them
# again would recreate a branch GitHub deleted on merge.
git fetch -q --prune origin
base=$(git merge-base "origin/main" HEAD)
pending=$(git diff --name-only "$base" HEAD -- "$RECORDS_DIR" | while read -r path; do
    git diff --quiet "origin/main" HEAD -- "$path" || echo "$path"
done)
if [[ -z $pending ]]; then
    echo "==> Nothing to push; main has every server edit"
elif [[ $(git rev-parse -q --verify "origin/$BRANCH" || true) == $(git rev-parse HEAD) ]]; then
    echo "==> origin/$BRANCH is already up to date"
else
    echo "==> Push $BRANCH"
    git push -q origin "$BRANCH" \
        || fail "push failed; check the deploy key (ssh -T git@github.com) or whether origin/$BRANCH diverged"
    echo "    Open or refresh the PR: $BRANCH -> main"
fi
