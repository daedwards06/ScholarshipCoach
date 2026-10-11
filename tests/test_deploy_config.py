"""Guards on deploy/: the server app must stay on loopback behind `tailscale serve`."""

from __future__ import annotations

import configparser
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT_DIR / "deploy"
UNIT_PATH = DEPLOY_DIR / "scholarshipcoach.service"
UPDATE_PATH = DEPLOY_DIR / "update.sh"
APP_DIR = "/srv/scholarshipcoach"


def _service_section() -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(UNIT_PATH, encoding="utf-8")
    return parser["Service"]


def test_unit_binds_loopback_only() -> None:
    exec_start = _service_section()["ExecStart"].split()
    assert exec_start[0] == f"{APP_DIR}/.venv/bin/streamlit"
    assert exec_start[1:3] == ["run", "app/main.py"]
    address = exec_start[exec_start.index("--server.address") + 1]
    assert address == "127.0.0.1"


def test_unit_runs_as_coach_from_repo_root() -> None:
    service = _service_section()
    assert service["User"] == "coach"
    assert service["WorkingDirectory"] == APP_DIR
    assert service["Restart"] == "always"


def test_nothing_in_deploy_exposes_the_app_publicly() -> None:
    files = [p for p in DEPLOY_DIR.rglob("*") if p.is_file()]
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8").lower()
        assert "0.0.0.0" not in text, path.name
        assert re.search(r"tailscale\s+funnel", text) is None, path.name


def test_update_health_check_fetches_a_real_font() -> None:
    text = UPDATE_PATH.read_text(encoding="utf-8")
    assert "/_stcore/health" in text
    match = re.search(r"FONT_PATH=(/app/static/fonts/\S+\.woff2)", text)
    assert match is not None
    assert (ROOT_DIR / match.group(1).lstrip("/")).is_file()


RESTIC_ENV = "/etc/scholarshipcoach/restic.env"


def test_restic_env_is_referenced_by_path_and_not_in_repo() -> None:
    for name in ("backup.sh", "restore.sh"):
        text = (DEPLOY_DIR / name).read_text(encoding="utf-8")
        assert f"ENV_FILE={RESTIC_ENV}" in text, name
    assert not list(ROOT_DIR.rglob("restic.env"))


def test_update_backs_up_before_pull() -> None:
    lines = UPDATE_PATH.read_text(encoding="utf-8").splitlines()
    commands = [line.strip() for line in lines if not line.strip().startswith(("#", "echo"))]
    backup = next(i for i, line in enumerate(commands) if line.startswith("deploy/backup.sh"))
    pull = next(i for i, line in enumerate(commands) if line.startswith("git pull"))
    assert backup < pull
    assert "|| fail" in commands[backup]


def test_backup_timer_is_nightly_and_persistent() -> None:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(DEPLOY_DIR / "scholarshipcoach-backup.timer", encoding="utf-8")
    assert parser["Timer"]["OnCalendar"] == "*-*-* 03:30:00"
    assert parser["Timer"]["Persistent"] == "true"


def _timer(name: str) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read(DEPLOY_DIR / name, encoding="utf-8")
    return parser["Timer"]


def test_verify_runs_first_sunday_monthly() -> None:
    timer = _timer("scholarshipcoach-verify.timer")
    assert timer["OnCalendar"] == "Sun *-*-01..07 07:00:00"
    assert timer["Persistent"] == "true"
    unit = (DEPLOY_DIR / "scholarshipcoach-verify.service").read_text(encoding="utf-8")
    assert "scripts/verify_catalog.py" in unit
    assert "User=coach" in unit


def test_catalog_sync_runs_nightly_after_backup() -> None:
    backup = _timer("scholarshipcoach-backup.timer")["OnCalendar"].split()[-1]
    sync = _timer("scholarshipcoach-catalog-sync.timer")
    assert sync["OnCalendar"].startswith("*-*-* ")
    assert sync["OnCalendar"].split()[-1] > backup
    assert sync["Persistent"] == "true"


def test_catalog_sync_validates_then_commits_only_records() -> None:
    text = (DEPLOY_DIR / "catalog_sync.sh").read_text(encoding="utf-8")
    assert "BRANCH=server" in text
    assert "RECORDS_DIR=data/catalog/records/" in text
    validate = text.index("scripts/validate_catalog.py")
    commit = text.index("git commit")
    assert text.index("uncommitted changes outside") < validate < commit
    commit_line = text[commit:].splitlines()[0]
    assert commit_line.rstrip().endswith('-- "$RECORDS_DIR"')


def test_backup_covers_all_private_data_and_restore_keeps_it_private() -> None:
    backup = (DEPLOY_DIR / "backup.sh").read_text(encoding="utf-8")
    assert 'paths=("$STAGE/coach.db" "$APP_DIR/data/private")' in backup
    assert '--exclude "$DB" --exclude "$DB-journal" --exclude "$DB-wal"' in backup
    restore = (DEPLOY_DIR / "restore.sh").read_text(encoding="utf-8")
    assert "install -d -m 700" in restore
    assert restore.index('cp -a "$WORK$APP_DIR/data/private/."') < restore.index('cp "$SNAP_DB"')
    assert restore.index('cp "$SNAP_DB"') < restore.index("chmod -R go-rwx data/private")
    # The moved-aside copy holds the same personal data and must never look like a git change.
    assert "ASIDE=data/private_before_restore_$STAMP" in restore
    gitignore = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/private_before_restore_*/" in gitignore


def test_delta_reports_are_synced_not_treated_as_stray() -> None:
    sync = (DEPLOY_DIR / "catalog_sync.sh").read_text(encoding="utf-8")
    update = UPDATE_PATH.read_text(encoding="utf-8")
    for text in (sync, update):
        assert "DELTA_REPORTS='data/processed/changes_*.json'" in text
        stray_line = next(line for line in text.splitlines() if line.startswith("stray="))
        assert r"^data/processed/changes_[0-9]*\.json$" in stray_line
    assert 'git commit -q -m "Snapshot delta report(s) from the server" -- "$DELTA_REPORTS"' in sync
    assert '-- "$RECORDS_DIR" "$DELTA_REPORTS")' in update


def test_pushes_use_a_deploy_key_not_a_token() -> None:
    bootstrap = (DEPLOY_DIR / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'PUSH_URL="${PUSH_URL:-git@github.com:' in bootstrap
    assert "remote.origin.pushurl" in bootstrap
    for path in (p for p in DEPLOY_DIR.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"ghp_|github_pat_|https://[^/\s]+@github\.com", text) is None, path.name


PRIVACY_PATH = DEPLOY_DIR / "privacy_check.sh"


def test_catalog_sync_runs_privacy_check_before_any_commit() -> None:
    text = (DEPLOY_DIR / "catalog_sync.sh").read_text(encoding="utf-8")
    check = text.index("deploy/privacy_check.sh $changed $deltas")
    assert text.index("uncommitted changes outside") < check < text.index("git commit")
    assert text.index("deltas=$(git status") < check
    assert text[check:].splitlines()[1].strip().startswith("|| fail")


def test_privacy_check_uses_fixed_strings_and_prints_file_names_only() -> None:
    text = PRIVACY_PATH.read_text(encoding="utf-8")
    assert "NAMES=data/private/privacy_names.txt" in text
    assert "ABBREVS=data/private/privacy_abbrevs.txt" in text
    assert '"-qiF -f $work/names" "-qiwF -f $work/abbrevs"' in text
    assert "(( status <= 1 )) || terms_error" in text  # a grep error holds, never passes
    # The only output on a hit is the list of paths; a term never reaches stdout or a log.
    assert r"""printf '    %s\n' "${hits[@]}" >&2""" in text
    gitignore = (ROOT_DIR / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "data/private/" in gitignore
    assert not list((ROOT_DIR / "deploy").rglob("privacy_*.txt"))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.skipif(sys.platform == "win32", reason="runs the bash script; CI is Linux")
def test_privacy_check_checks_added_lines_and_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    records = repo / "data" / "catalog" / "records"
    private = repo / "data" / "private"
    records.mkdir(parents=True)
    private.mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    old = records / "old.json"
    # Pretty-printed like the real records, so checked_on sits on its own line.
    record = '{\n  "note": "On the Example Tech campus",\n  "checked_on": "%s"\n}\n'
    old.write_text(record % "2026-01-01")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")

    def run(*paths: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(PRIVACY_PATH), *paths], cwd=repo, capture_output=True, text=True
        )

    rel_old = "data/catalog/records/old.json"
    assert run(rel_old).returncode == 2  # no lists at all

    (private / "privacy_names.txt").write_text("example tech\r\n\r\n", newline="")
    (private / "privacy_abbrevs.txt").write_bytes(b"\xef\xbb\xbfXT\n")
    assert run(rel_old).returncode == 2  # byte-order mark refused
    (private / "privacy_abbrevs.txt").write_text("XT\n")

    old.write_text(record % "2026-10-01")
    assert run(rel_old).returncode == 0  # only checked_on changed

    new = records / "new.json"
    new.write_text('{"note": "for her: XT merit award"}\n')
    result = run(rel_old, "data/catalog/records/new.json")
    assert result.returncode == 1
    assert "data/catalog/records/new.json" in result.stderr
    assert "old.json" not in result.stderr
    assert "XT" not in result.stdout + result.stderr

    new.write_text('{"note": "NEXT steps for the XTRA fund"}\n')  # abbreviation inside words
    assert run("data/catalog/records/new.json").returncode == 0

    old.write_text(record % "2026-10-01" + '{"extra": "for her: EXAMPLE TECH"}\n')
    assert run(rel_old).returncode == 1  # an added line, case-insensitive
