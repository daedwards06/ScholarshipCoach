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

cd "$APP_DIR"
fail() { echo "update.sh: $*" >&2; exit 1; }

# Catalog edits made in the app are the only local changes allowed; anything
# else means someone hand-edited the server, and a pull would bury it.
stray=$(git status --porcelain --untracked-files=all | cut -c4- | grep -v "^$RECORDS_DIR" || true)
if [[ -n $stray ]]; then
    fail "uncommitted changes outside $RECORDS_DIR; resolve them first:
$stray"
fi

if [[ -n $(git status --porcelain --untracked-files=all -- "$RECORDS_DIR") ]]; then
    if [[ -x deploy/catalog_sync.sh ]]; then
        echo "==> Catalog sync"
        deploy/catalog_sync.sh
    else
        fail "catalog edits in $RECORDS_DIR but deploy/catalog_sync.sh is not installed"
    fi
fi

echo "==> git pull"
git pull --ff-only origin main

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
