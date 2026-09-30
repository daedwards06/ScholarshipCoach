"""Capture every Student and Parent section at phone and desktop width.

Starts the app on a free port, drives the system Edge through Playwright (so
nothing is downloaded), and writes ``<width>_<mode>_<section>_<theme>.png``.

By default the sweep runs against a scratch copy of ``data/private/``, so it
never writes the family's database; ``--live-db`` opts out.  ``--seed`` saves
the top three Find results and adds an essay and a recommender through the UI
first, so populated states are captured too.

Needs the ``ui`` extra: ``pip install -e ".[ui]"``.

    python scripts/screenshot_app.py --out $env:TEMP\\coach_shots --scratch-db
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from app.modes import nav_sections, section_url_path
from src.store.db import PRIVATE_DIR

ROOT_DIR = Path(__file__).resolve().parents[1]
# Beside app/main.py so Streamlit serves the same app/static/ (fonts) in a sweep.
ENTRY_SCRIPT = ROOT_DIR / "app" / "screenshot_entry.py"
PRIVATE_DIR_ENV = "COACH_SHOT_PRIVATE_DIR"

VIEWPORTS: tuple[tuple[int, int], ...] = ((400, 860), (1366, 900))
MODE_SECTIONS: dict[str, tuple[str, ...]] = {
    "student": nav_sections("student"),
    "parent": nav_sections("parent"),
}
THEMES = ("light", "dark")

IDLE_SELECTOR = '[data-testid="stApp"][data-test-script-state="notRunning"]'
RUN_BUTTON = "Run Scholarship Coach"
SAVE_BUTTON_SELECTOR = '[class*="st-key-save_"] button:not([disabled])'
SEED_SAVES = 3


@dataclass(frozen=True)
class Capture:
    width: int
    height: int
    mode: str
    section: str
    theme: str

    @property
    def filename(self) -> str:
        return f"{self.width}_{self.mode}_{self.section}_{self.theme}.png"


def selected_themes(choice: str) -> tuple[str, ...]:
    return THEMES if choice == "both" else (choice,)


def plan_captures(themes: tuple[str, ...], sections: set[str] | None = None) -> list[Capture]:
    """Every (viewport, mode, section, theme) the sweep will photograph."""
    return [
        Capture(width, height, mode, section, theme)
        for theme in themes
        for width, height in VIEWPORTS
        for mode, mode_sections in MODE_SECTIONS.items()
        for section in mode_sections
        if sections is None or section in sections
    ]


def section_url(base_url: str, mode: str, section: str) -> str:
    # The first page is the default one, which Streamlit serves at the root.
    path = "" if section == MODE_SECTIONS[mode][0] else section_url_path(section)
    return f"{base_url}/{path}?{urlencode({'shot_mode': mode})}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "coach_shots",
        help="Output directory (default: <temp>/coach_shots, outside the repo)",
    )
    parser.add_argument("--theme", choices=("light", "dark", "both"), default="both")
    db_group = parser.add_mutually_exclusive_group()
    db_group.add_argument(
        "--scratch-db",
        dest="scratch_db",
        action="store_true",
        default=True,
        help="Run against a temp copy of data/private/ (the default)",
    )
    db_group.add_argument(
        "--live-db",
        dest="scratch_db",
        action="store_false",
        help="Run against the real data/private/ (read-only sweep; no --seed)",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Save the top 3 Find results and add an essay and a recommender first",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=MODE_SECTIONS["parent"],
        help="Capture only these sections",
    )
    args = parser.parse_args(argv)
    if args.seed and not args.scratch_db:
        parser.error("--seed writes to the database; it cannot be combined with --live-db")
    return args


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def scratch_private_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="coach_shot_private_") as tmp:
        target = Path(tmp) / "private"
        if PRIVATE_DIR.is_dir():
            shutil.copytree(PRIVATE_DIR, target)
        else:
            target.mkdir()
        yield target


@contextmanager
def running_app(private_dir: Path | None) -> Iterator[str]:
    port = free_port()
    env = dict(os.environ)
    if private_dir is not None:
        env[PRIVATE_DIR_ENV] = str(private_dir)
    else:
        env.pop(PRIVATE_DIR_ENV, None)
    command = [
        sys.executable, "-m", "streamlit", "run", str(ENTRY_SCRIPT),
        "--server.port", str(port),
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    process = subprocess.Popen(
        command, cwd=ROOT_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base_url, process)
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()


def _wait_for_health(base_url: str, process: subprocess.Popen[bytes], timeout: float = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Streamlit exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Streamlit did not become healthy within {timeout:.0f}s")


def wait_idle(page: Any, timeout_ms: float = 120_000) -> None:
    """Wait for the current script run to finish and the page to settle."""
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout_ms)
    # The state flips to "running" a moment after a click, so give it that
    # moment before trusting "notRunning".
    page.wait_for_timeout(400)
    page.wait_for_selector(IDLE_SELECTOR, timeout=timeout_ms)
    page.wait_for_timeout(600)


def open_section(page: Any, base_url: str, mode: str, section: str) -> None:
    page.goto(section_url(base_url, mode, section))
    wait_idle(page)


def run_find(page: Any) -> bool:
    """Press Find's Run button if the page still has one; return whether it ran."""
    button = page.get_by_role("button", name=RUN_BUTTON)
    if button.count() == 0 or not button.first.is_enabled():
        return False
    button.first.click()
    wait_idle(page, timeout_ms=600_000)
    return True


def seed(browser: Any, base_url: str) -> None:
    context = browser.new_context(viewport={"width": 1366, "height": 900})
    page = context.new_page()
    try:
        open_section(page, base_url, "student", "find")
        if not run_find(page):
            raise RuntimeError("Find has no enabled Run button; is there a snapshot?")
        for _ in range(SEED_SAVES):
            save = page.locator(SAVE_BUTTON_SELECTOR)
            if save.count() == 0:
                break
            save.first.click()
            wait_idle(page)

        open_section(page, base_url, "student", "essays")
        form = page.locator('[data-testid="stForm"]').filter(has_text="Add essay")
        form.get_by_label("Title").fill("The summer I rebuilt the robot")
        form.get_by_label("Draft").fill(
            "Our team's robot stopped working two weeks before regionals, and I learned "
            "more fixing it than I did building it."
        )
        form.get_by_role("button", name="Add essay").click()
        wait_idle(page)

        open_section(page, base_url, "student", "recommenders")
        form = page.locator('[data-testid="stForm"]').filter(has_text="Add recommender")
        form.get_by_label("Name").fill("Ms. Rivera")
        form.get_by_label("Role").fill("AP Computer Science teacher")
        form.get_by_role("button", name="Add recommender").click()
        wait_idle(page)
    finally:
        context.close()


def capture(browser: Any, base_url: str, shot: Capture, out_dir: Path) -> Path:
    context = browser.new_context(
        viewport={"width": shot.width, "height": shot.height},
        color_scheme=shot.theme,
    )
    page = context.new_page()
    try:
        open_section(page, base_url, shot.mode, shot.section)
        if shot.section == "find":
            run_find(page)
        # stMain is the scroll container, so a full-page screenshot would stop
        # at the viewport; grow the viewport to the content instead.
        content_height = page.evaluate(
            "() => document.querySelector('[data-testid=\"stMain\"]')?.scrollHeight ?? 0"
        )
        height = max(shot.height, int(content_height))
        if height != shot.height:
            page.set_viewport_size({"width": shot.width, "height": height})
            page.wait_for_timeout(800)
        path = out_dir / shot.filename
        page.screenshot(path=str(path))
        return path
    finally:
        context.close()


def sweep(args: argparse.Namespace, private_dir: Path | None) -> list[Path]:
    from playwright.sync_api import sync_playwright

    shots = plan_captures(selected_themes(args.theme), set(args.sections or []) or None)
    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with running_app(private_dir) as base_url, sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            if args.seed:
                print("Seeding scratch database through the UI...")
                seed(browser, base_url)
            for index, shot in enumerate(shots, start=1):
                path = capture(browser, base_url, shot, args.out)
                print(f"[{index}/{len(shots)}] {path.name}")
                written.append(path)
        finally:
            browser.close()
    return written


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scratch_db:
        with scratch_private_dir() as private_dir:
            written = sweep(args, private_dir)
    else:
        written = sweep(args, None)
    print(f"Wrote {len(written)} screenshots to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
