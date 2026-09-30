#!/usr/bin/env bash
# Encrypted off-server backup of everything git does not carry. Run as coach,
# nightly by scholarshipcoach-backup.timer and before every deploy/update.sh:
#
#   /srv/scholarshipcoach/deploy/backup.sh
#
# Repository, password and B2 keys come from /etc/scholarshipcoach/restic.env
# (owner coach, mode 600, never in git), e.g.:
#   RESTIC_REPOSITORY=b2:<bucket>:scholarshipcoach
#   RESTIC_PASSWORD=...
#   B2_ACCOUNT_ID=...
#   B2_ACCOUNT_KEY=...
set -euo pipefail

APP_DIR=/srv/scholarshipcoach
ENV_FILE=/etc/scholarshipcoach/restic.env
# Fixed path, because restic records absolute paths and restore.sh looks here.
STAGE=/home/coach/backup-staging
DB=$APP_DIR/data/private/coach.db
TAG=scholarshipcoach

fail() { echo "backup.sh: $*" >&2; exit 1; }

[[ -r $ENV_FILE ]] || fail "cannot read $ENV_FILE"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a
[[ -n ${RESTIC_REPOSITORY:-} ]] || fail "RESTIC_REPOSITORY not set in $ENV_FILE"

# sqlite3 would create an empty database at a missing path and back that up.
[[ -f $DB ]] || fail "no database at $DB"

echo "==> Consistent copy of coach.db"
mkdir -p "$STAGE"
rm -f "$STAGE/coach.db"
# .backup uses SQLite's online backup API, so it is safe while the app writes;
# a plain file copy can catch a half-written page.
sqlite3 "$DB" ".backup '$STAGE/coach.db'"
[[ $(sqlite3 "$STAGE/coach.db" "PRAGMA integrity_check") == ok ]] \
    || fail "integrity_check failed on the copy of $DB"

# All of data/private/ (students/, eval/ worksheets, ...) except the live
# database, which only the .backup copy above captures consistently.
paths=("$STAGE/coach.db" "$APP_DIR/data/private")
for extra in \
    "$APP_DIR/data/catalog/inbox" \
    "$APP_DIR/.streamlit/secrets.toml"; do
    if [[ -e $extra ]]; then
        paths+=("$extra")
    else
        echo "    (skipping missing $extra)"
    fi
done

echo "==> restic backup"
restic backup --tag "$TAG" \
    --exclude "$DB" --exclude "$DB-journal" --exclude "$DB-wal" --exclude "$DB-shm" \
    "${paths[@]}"

echo "==> restic forget (14 daily, 8 weekly, 12 monthly)"
# Group by tag, not host+paths: a rebuilt server or a newly present inbox
# would otherwise start a separate retention group.
restic forget --tag "$TAG" --group-by tags \
    --keep-daily 14 --keep-weekly 8 --keep-monthly 12 --prune

echo "==> Backup complete"
