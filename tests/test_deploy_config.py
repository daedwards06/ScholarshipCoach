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
