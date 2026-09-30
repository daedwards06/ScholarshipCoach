#!/usr/bin/env bash
# Pull main, reinstall, restart, and prove the app came back. Run as coach:
#
#   /srv/scholarshipcoach/deploy/update.sh
set -euo pipefail

APP_DIR=/srv/scholarshipcoach
SERVICE=scholarshipcoach
BASE_URL=http://127.0.0.1:8501
FONT_PATH=/app/static/fonts/nunito-latin-700-normal.woff2
RECORDS_DIR=data/catalog/records/
DELTA_REPORTS='data/processed/changes_*.json'

cd "$APP_DIR"
fail() { echo "update.sh: $*" >&2; exit 1; }

# Catalog edits made in the app, and the delta reports "Rebuild snapshot"
# writes, are the only local changes allowed; anything else means someone
# hand-edited the server, and a pull would bury it.
stray=$(git status --porcelain --untracked-files=all | cut -c4- | grep -v -e "^$RECORDS_DIR" -e '^data/processed/changes_[0-9]*\.json$' || true)
if [[ -n $stray ]]; then
    fail "uncommitted changes outside $RECORDS_DIR; resolve them first:
$stray"
fi

if [[ -n $(git status --porcelain --untracked-files=all -- "$RECORDS_DIR" "$DELTA_REPORTS") ]]; then
    if [[ -x deploy/catalog_sync.sh ]]; then
        echo "==> Catalog sync"
        deploy/catalog_sync.sh
    else
        fail "catalog edits in $RECORDS_DIR but deploy/catalog_sync.sh is not installed"
    fi
fi

# A pull can add files under src/store/migrations/, and src/store/db.py applies
# them to coach.db on the first connection after the restart -- irreversibly.
echo "==> Backup before pull"
deploy/backup.sh || fail "backup failed; not pulling (coach.db may be migrated by the update)"

before=$(git rev-parse HEAD)
# The checkout is on the `server` branch, which may hold catalog commits main
# has not merged yet, so main is merged in rather than fast-forwarded.
echo "==> git pull (merge origin/main into $(git symbolic-ref --short HEAD))"
git pull -q --no-rebase --no-edit origin main \
    || fail "merging origin/main conflicted; resolve it (or git merge --abort) before updating"

added=$(git diff --name-only --diff-filter=A "$before" HEAD -- src/store/migrations/)
if [[ -n $added ]]; then
    echo "==> New migrations (applied to coach.db on restart):"
    printf '    %s\n' $added
fi

echo "==> pip install"
.venv/bin/pip install -q -e . -c constraints-ci.txt

echo "==> Restart"
sudo /usr/bin/systemctl restart "$SERVICE"

echo "==> Health check"
for _ in $(seq 1 60); do
    if [[ $(curl -fsS "$BASE_URL/_stcore/health" 2>/dev/null) == ok ]]; then
        break
    fi
    sleep 1
done
[[ $(curl -fsS "$BASE_URL/_stcore/health" 2>/dev/null) == ok ]] \
    || fail "no 'ok' from $BASE_URL/_stcore/health after 60s; see: journalctl -u $SERVICE -n 50"

status=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE_URL$FONT_PATH")
[[ $status == 200 ]] \
    || fail "font $FONT_PATH returned HTTP $status; is the service running from $APP_DIR?"

echo "==> Updated to $(git rev-parse --short HEAD); app healthy"
