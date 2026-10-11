#!/usr/bin/env bash
# Hold back catalog edits that would publish who the student is or where she is
# applying. catalog_sync.sh runs this on the files it is about to commit to the
# public `server` branch; run it by hand from the checkout the same way:
#
#   deploy/privacy_check.sh data/catalog/records/<id>.json ...
#
# Terms come from two git-ignored lists in data/private/ (coach owned, mode
# 600, never in git), one term per line:
#   privacy_names.txt    case-insensitive fixed strings
#   privacy_abbrevs.txt  the same, whole words only, so a short abbreviation
#                        cannot match inside another word
#
# Only the lines a change adds are checked (git diff against HEAD; a file new
# to HEAD counts whole). Existing public records name colleges and places, and
# the monthly verify run stamps checked_on into every record, so checking whole
# files would hold the sync every month.
#
# Exit 0: no term in any added line. Exit 1: a match; only the matching file
# names are printed, never the term. Exit 2: a terms file is missing, empty,
# or starts with a UTF-8 byte-order mark (which silently matches nothing).
set -euo pipefail
# Case-insensitive matching of non-ASCII names needs a UTF-8 locale; Ubuntu
# always ships C.UTF-8.
export LC_ALL=C.UTF-8

NAMES=data/private/privacy_names.txt
ABBREVS=data/private/privacy_abbrevs.txt

terms_error() { echo "privacy_check.sh: $*; holding the sync" >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

load_terms() {
    local src=$1 dest=$2
    [[ -r $src ]] || terms_error "cannot read $src"
    [[ $(head -c3 "$src" | od -An -tx1 | tr -d ' \n') != efbbbf ]] \
        || terms_error "$src starts with a UTF-8 byte-order mark; save it without one"
    sed -e 's/\r$//' "$src" | grep -v '^[[:space:]]*$' > "$dest" || true
    [[ -s $dest ]] || terms_error "$src has no terms"
}

load_terms "$NAMES" "$work/names"
load_terms "$ABBREVS" "$work/abbrevs"

hits=()
for path in "$@"; do
    [[ -f $path ]] || continue  # a deletion adds nothing
    if git cat-file -e "HEAD:$path" 2>/dev/null; then
        git diff --no-color --unified=0 HEAD -- "$path" \
            | sed -n -e '/^+++ b\//d' -e 's/^+//p' > "$work/added"
    else
        cp "$path" "$work/added"
    fi
    for check in "-qiF -f $work/names" "-qiwF -f $work/abbrevs"; do
        status=0
        # shellcheck disable=SC2086  # $check is the flag set; $work has no spaces
        grep $check "$work/added" || status=$?
        # grep exits 1 for no match; anything higher is an error, which must hold
        # the sync rather than read as clean.
        (( status <= 1 )) || terms_error "grep failed (exit $status) on $path"
        if (( status == 0 )); then
            hits+=("$path")
            break
        fi
    done
done

if (( ${#hits[@]} )); then
    echo "privacy_check.sh: a privacy term appears in what these files add:" >&2
    printf '    %s\n' "${hits[@]}" >&2
    exit 1
fi
