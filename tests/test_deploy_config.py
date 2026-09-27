"""Guards on deploy/: the server app must stay on loopback behind `tailscale serve`."""

from __future__ import annotations

import configparser
import re
from pathlib import Path

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


def test_pushes_use_a_deploy_key_not_a_token() -> None:
    bootstrap = (DEPLOY_DIR / "bootstrap.sh").read_text(encoding="utf-8")
    assert 'PUSH_URL="${PUSH_URL:-git@github.com:' in bootstrap
    assert "remote.origin.pushurl" in bootstrap
    for path in (p for p in DEPLOY_DIR.rglob("*") if p.is_file()):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"ghp_|github_pat_|https://[^/\s]+@github\.com", text) is None, path.name
