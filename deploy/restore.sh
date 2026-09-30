#!/usr/bin/env bash
# Restore data/private/ from a restic snapshot. Run as root:
#
#   sudo /srv/scholarshipcoach/deploy/restore.sh latest
#   sudo /srv/scholarshipcoach/deploy/restore.sh <snapshot-id>
#
# The current data/private/ is moved aside, never deleted. inbox/ and
# secrets.toml are restored only where the server has none (a rebuild); an
# existing copy is left alone.
set -euo pipefail

APP_DIR=/srv/scholarshipcoach
APP_USER=coach
SERVICE=scholarshipcoach
ENV_FILE=/etc/scholarshipcoach/restic.env
STAGE=/home/coach/backup-staging

fail() { echo "restore.sh: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || fail "usage: restore.sh <snapshot-id|latest>"
SNAPSHOT=$1
[[ $EUID -eq 0 ]] || fail "must run as root (it stops and starts $SERVICE)"
[[ -r $ENV_FILE ]] || fail "cannot read $ENV_FILE"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

STAMP=$(date +%Y%m%d_%H%M)
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Fetch before stopping anything: a bad snapshot id or an unreachable
# repository should leave the app running.
echo "==> restic restore $SNAPSHOT"
restic restore "$SNAPSHOT" --tag scholarshipcoach --target "$WORK"
SNAP_DB=$WORK$STAGE/coach.db
[[ -f $SNAP_DB ]] || fail "snapshot has no $STAGE/coach.db"
[[ $(sqlite3 -readonly "$SNAP_DB" "PRAGMA integrity_check") == ok ]] \
    || fail "integrity_check failed on the restored coach.db"

echo "==> Stop $SERVICE"
systemctl stop "$SERVICE"

cd "$APP_DIR"
if [[ -d data/private ]]; then
    ASIDE=data/private_before_restore_$STAMP
    mv data/private "$ASIDE"
    echo "    moved data/private -> $ASIDE"
fi
install -d -m 700 -o "$APP_USER" -g "$APP_USER" data/private
if [[ -d $WORK$APP_DIR/data/private ]]; then
    cp -a "$WORK$APP_DIR/data/private/." data/private/
fi
cp "$SNAP_DB" data/private/coach.db
chown -R "$APP_USER:$APP_USER" data/private
# Copies made as root come out 644; the essays and names stay coach-only.
chmod -R go-rwx data/private

for extra in data/catalog/inbox .streamlit/secrets.toml; do
    if [[ -e $extra ]]; then
        echo "    kept existing $extra"
    elif [[ -e $WORK$APP_DIR/$extra ]]; then
        cp -a "$WORK$APP_DIR/$extra" "$extra"
        chown -R "$APP_USER:$APP_USER" "$extra"
        echo "    restored $extra"
    fi
done

echo "==> Row counts in data/private/coach.db"
tables=$(sqlite3 -readonly data/private/coach.db \
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
[[ -n $tables ]] || fail "restored coach.db has no tables"
for table in $tables; do
    printf '    %-28s %s\n' "$table" "$(sqlite3 -readonly data/private/coach.db "SELECT COUNT(*) FROM \"$table\"")"
done

echo "==> Start $SERVICE"
systemctl start "$SERVICE"
systemctl is-active --quiet "$SERVICE" || fail "$SERVICE did not start; see: journalctl -u $SERVICE -n 50"

echo "==> Restored $SNAPSHOT. Once the app looks right, delete ${ASIDE:-nothing to delete}."
