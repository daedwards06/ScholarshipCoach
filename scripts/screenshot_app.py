"""Capture every Student and Parent section at phone and desktop width.

Starts the app on a free port, drives the system Edge through Playwright (so
nothing is downloaded), and writes ``<width>_<mode>_<section>_<theme>.png``.

By default the sweep runs against a scratch copy of ``data/private/``, so it
never writes the family's database; ``--live-db`` opts out.  ``--seed`` saves
the top three Find results and adds an essay and a recommender through the UI
first, and writes a made-up first target school with requirements straight
into the scratch database, so populated states are captured too.

``--journeys`` walks the plan's journeys (S1-S6, P1-P4) instead, step by step
at 400px, writing one PNG per step and printing taps per journey against its
budget.  It always starts from a fresh scratch database holding the demo
profile set to a 10th grader, and from scratch copies of the catalog records
and the latest snapshot, so P2's catalog form and rebuild never touch
``data/catalog/records/`` or ``data/processed/``.

Needs the ``ui`` extra: ``pip install -e ".[ui]"``.

    python scripts/screenshot_app.py --out $env:TEMP\\coach_shots --scratch-db
    python scripts/screenshot_app.py --journeys --scratch-db
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from app.modes import nav_sections, section_url_path
from src.normalize.catalog_schema import CATALOG_DIR
from src.profile import store as profile_store
from src.profile.grade_levels import infer_graduation_year
from src.profile.store import DEFAULT_STUDENT_ID
from src.store import repo
from src.store.db import PRIVATE_DIR, connect

ROOT_DIR = Path(__file__).resolve().parents[1]
# Beside app/main.py so Streamlit serves the same app/static/ (fonts) in a sweep.
ENTRY_SCRIPT = ROOT_DIR / "app" / "screenshot_entry.py"
PRIVATE_DIR_ENV = "COACH_SHOT_PRIVATE_DIR"
DATA_ROOT_ENV = "COACH_SHOT_DATA_ROOT"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

VIEWPORTS: tuple[tuple[int, int], ...] = ((400, 860), (1366, 900))
MODE_SECTIONS: dict[str, tuple[str, ...]] = {
    "student": nav_sections("student"),
    "parent": nav_sections("parent"),
}
THEMES = ("light", "dark")

IDLE_SELECTOR = '[data-testid="stApp"][data-test-script-state="notRunning"]'
AWARD_CARD_SELECTOR = '[class*="st-key-award_"]'
SAVE_BUTTON_SELECTOR = '[class*="st-key-save_"] button:not([disabled])'
SEED_SAVES = 3

# A made-up school: real names, thresholds and dates never go in a committed file.
SEED_COLLEGE = "Example State University"
SEED_REQUIREMENTS: tuple[tuple[str, str, str, str, str], ...] = (
    # (label, category, target, due_by, status)
    ("Algebra II", "course", "1 unit", "10", "in_progress"),
    ("Unweighted GPA", "gpa", "3.5", "", "not_started"),
    ("Take the PSAT", "test", "", "2026-10-15", "not_started"),
    ("Precalculus", "course", "1 unit", "11", "not_started"),
    ("Summer program application", "program", "", "2027-02-01", "not_started"),
    ("Calculus", "course", "1 unit", "12", "not_started"),
    ("SAT for merit aid", "test", "SAT 1200", "12", "not_started"),
    ("Biology", "course", "1 unit", "9", "met"),
)


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
    parser.add_argument(
        "--journeys",
        action="store_true",
        help="Walk journeys S1-S6 and P1-P4 at 400px instead of the page sweep",
    )
    args = parser.parse_args(argv)
    if args.seed and not args.scratch_db:
        parser.error("--seed writes to the database; it cannot be combined with --live-db")
    if args.journeys and not args.scratch_db:
        parser.error("--journeys writes to the database and catalog; it cannot use --live-db")
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
def running_app(private_dir: Path | None, data_root: Path | None = None) -> Iterator[str]:
    port = free_port()
    env = dict(os.environ)
    if private_dir is not None:
        env[PRIVATE_DIR_ENV] = str(private_dir)
    else:
        env.pop(PRIVATE_DIR_ENV, None)
    if data_root is not None:
        env[DATA_ROOT_ENV] = str(data_root)
    else:
        env.pop(DATA_ROOT_ENV, None)
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


def wait_for_find(page: Any) -> bool:
    """Let Find's first ranking finish (it runs on arrival); return whether cards showed."""
    wait_idle(page, timeout_ms=600_000)
    return page.locator(AWARD_CARD_SELECTOR).count() > 0


def seed(browser: Any, base_url: str) -> None:
    context = browser.new_context(viewport={"width": 1366, "height": 900})
    page = context.new_page()
    try:
        open_section(page, base_url, "parent", "find")
        if not wait_for_find(page):
            raise RuntimeError("Find showed no award cards; is there a snapshot?")
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


def seed_path(private_dir: Path) -> None:
    """Give My Path a made-up first target school (scratch database only)."""
    conn = connect(private_dir / "coach.db")
    try:
        if repo.get_student(conn, DEFAULT_STUDENT_ID) is None:
            repo.upsert_student(conn, DEFAULT_STUDENT_ID)
        college = repo.create_college(conn, DEFAULT_STUDENT_ID, SEED_COLLEGE, in_state=True)
        repo.set_first_target(conn, DEFAULT_STUDENT_ID, college.id)
        for position, (label, category, target, due_by, status) in enumerate(SEED_REQUIREMENTS):
            repo.create_requirement(
                conn, college.id, label, category,
                target=target, due_by=due_by, status=status, position=position,
            )
    finally:
        conn.close()


def capture(browser: Any, base_url: str, shot: Capture, out_dir: Path) -> Path:
    context = browser.new_context(
        viewport={"width": shot.width, "height": shot.height},
        color_scheme=shot.theme,
    )
    page = context.new_page()
    try:
        open_section(page, base_url, shot.mode, shot.section)
        if shot.section == "find":
            wait_for_find(page)
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
    if args.seed and private_dir is not None:
        seed_path(private_dir)
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


# -- journeys ------------------------------------------------------------------

JOURNEY_VIEWPORT = (400, 860)
JOURNEY_GRADE = "10"
# R3: a page change through the folded nav costs three taps (open the menu,
# open the group, pick the page).  The walk opens the URL and charges three.
NAV_TAPS = 3
STEP_TAPS: dict[str, int] = {
    "open": 0, "type": 0, "check": 0, "form": 0, "tap": 1, "link": 1, "nav": NAV_TAPS,
}

SEED_ESSAY = (
    "The summer I rebuilt the robot",
    "Our team's robot stopped working two weeks before regionals, and I learned "
    "more fixing it than I did building it.",
)
# Made up, like the seeded school; P2 adds it through the catalog form.
SEED_PROGRAM = "Example Coding Camp"
SEED_PROGRAM_URL = "https://example.org/coding-camp"
SEED_PROGRAM_GRADE = "High School Sophomore"
SEED_PARENT_ASK = "Pay the PSAT fee"
SEED_LABEL_EDIT = "Precalculus (honors)"
EDITOR_ROW_PX = 35

Act = Callable[["Walker"], None]


@dataclass(frozen=True)
class Budget:
    """A journey's tap budget from the plan: taps, plus a nav or a form it may use."""

    taps: int
    navs: int = 0
    forms: int = 0


@dataclass(frozen=True)
class Step:
    name: str
    kind: str
    act: Act
    # Set for a look at the other person's view; that is a check, never counted.
    mode: str | None = None


@dataclass(frozen=True)
class Journey:
    id: str
    who: str
    title: str
    budget: Budget
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class Cost:
    taps: int
    navs: int
    forms: int


def journey_cost(journey: Journey) -> Cost:
    """What the journey's own person spent: taps (nav excluded), navs and forms."""
    own = [step for step in journey.steps if step.mode in (None, journey.who)]
    return Cost(
        taps=sum(STEP_TAPS[step.kind] for step in own if step.kind != "nav"),
        navs=sum(1 for step in own if step.kind == "nav"),
        forms=sum(1 for step in own if step.kind == "form"),
    )


def charged_taps(cost: Cost, budget: Budget) -> int:
    """Taps counted against the budget: a nav the budget does not allow costs NAV_TAPS."""
    return cost.taps + NAV_TAPS * max(0, cost.navs - budget.navs)


def within_budget(cost: Cost, budget: Budget) -> bool:
    return charged_taps(cost, budget) <= budget.taps and cost.forms <= budget.forms


def _extras(navs: int, forms: int) -> str:
    return (" + nav" if navs else "") + (" + form" if forms else "")


def budget_text(budget: Budget) -> str:
    return f"{budget.taps}{_extras(budget.navs, budget.forms)}"


def cost_text(cost: Cost, budget: Budget) -> str:
    navs = min(cost.navs, budget.navs)
    return f"{charged_taps(cost, budget)}{_extras(navs, cost.forms)}"


def step_filename(journey: Journey, index: int, step: Step) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", step.name.casefold()).strip("-")
    return f"journey_{journey.id}_{index:02d}_{slug}.png"


class JourneyError(RuntimeError):
    pass


class Walker:
    """One 400px browser context per mode, kept for the whole walk."""

    def __init__(self, browser: Any, base_url: str) -> None:
        self._browser = browser
        self._base_url = base_url
        self._contexts: dict[str, Any] = {}
        self._pages: dict[str, Any] = {}

    def page(self, mode: str) -> Any:
        if mode not in self._pages:
            width, height = JOURNEY_VIEWPORT
            context = self._browser.new_context(
                viewport={"width": width, "height": height}, color_scheme="light"
            )
            self._contexts[mode] = context
            self._pages[mode] = context.new_page()
        return self._pages[mode]

    def open(self, mode: str, section: str) -> None:
        open_section(self.page(mode), self._base_url, mode, section)
        if section == "find":
            wait_for_find(self.page(mode))

    def close(self) -> None:
        for context in self._contexts.values():
            context.close()


def _visible_text(page: Any, text: str) -> Any:
    return page.get_by_text(text).filter(visible=True).first


def _expect_text(page: Any, text: str, *, in_view: bool = False) -> None:
    target = _visible_text(page, text)
    try:
        target.wait_for(state="visible", timeout=20_000)
    except Exception as exc:
        raise JourneyError(f"expected to see {text!r}") from exc
    if in_view:
        box = target.bounding_box()
        if box is None or box["y"] + box["height"] > JOURNEY_VIEWPORT[1]:
            raise JourneyError(f"{text!r} is below the fold at 400px")


def _click(page: Any, locator: Any) -> None:
    try:
        locator.click(timeout=20_000)
    except Exception as exc:
        raise JourneyError(f"could not tap {locator}") from exc
    wait_idle(page)


def _open(mode: str, section: str, *expect: str) -> Act:
    def act(walker: Walker) -> None:
        walker.open(mode, section)
        for text in expect:
            _expect_text(walker.page(mode), text)

    return act


def _tap(mode: str, locate: Callable[[Any], Any], *expect: str) -> Act:
    def act(walker: Walker) -> None:
        page = walker.page(mode)
        _click(page, locate(page))
        for text in expect:
            _expect_text(page, text)

    return act


def _expander(label: str) -> Callable[[Any], Any]:
    return lambda page: (
        page.locator('[data-testid="stExpander"] summary').filter(has_text=label).first
    )


def _home_link(label: str) -> Callable[[Any], Any]:
    return lambda page: page.locator(".st-key-home_links").get_by_role("link", name=label)


def _s1_check(walker: Walker) -> None:
    page = walker.page("student")
    _expect_text(page, "Your path to")
    _expect_text(page, "on track", in_view=True)
    _expect_text(page, "Coming up")


def _s2_tick(walker: Walker) -> None:
    page = walker.page("student")
    box = page.locator('[data-testid="stCheckbox"]').filter(has_text="Take the PSAT")
    _click(page, box.first)
    if box.count():
        raise JourneyError("the ticked row is still in Coming up")


def _add_first_opportunity(walker: Walker) -> None:
    page = walker.page("student")
    button = page.locator('[class*="st-key-opp_add_"]').first
    key = re.search(r"st-key-opp_add_(\S+)", button.get_attribute("class") or "")
    if key is None:
        raise JourneyError("no card offers 'Add to my to-dos'")
    _click(page, button.locator("button"))
    card = page.locator(f".st-key-opp_{key.group(1)}")
    if not card.get_by_text("On my to-dos").count():
        raise JourneyError("the card did not turn into 'On my to-dos'")


def _program_card(page: Any) -> Any:
    return page.locator('[class*="st-key-opp_"]').filter(has_text=SEED_PROGRAM).first


def _fill_program_form(walker: Walker) -> None:
    page = walker.page("parent")
    form = page.locator('[data-testid="stForm"]')
    form.get_by_text("Program (a summer program", exact=False).click()
    form.get_by_label("Title", exact=True).fill(SEED_PROGRAM)
    form.get_by_label("Source URL", exact=True).fill(SEED_PROGRAM_URL)
    form.locator('[data-testid="stSelectbox"]').filter(has_text="Trust").click()
    page.get_by_role("option", name="Verified by us").click()
    form.get_by_label("Verified by", exact=True).fill("Parent")
    form.get_by_role("tab", name="Who it's for").click()
    grades = form.locator('[data-testid="stMultiSelect"]').filter(has_text="Grade levels")
    grades.click()
    page.keyboard.type(SEED_PROGRAM_GRADE)
    page.keyboard.press("Enter")
    page.keyboard.press("Escape")


def _submit_program_form(walker: Walker) -> None:
    page = walker.page("parent")
    _click(page, page.get_by_role("button", name="Confirm and add to the catalog"))
    _expect_text(page, "to the catalog (")


def _ask_parents(walker: Walker) -> None:
    page = walker.page("student")
    form = page.locator('[data-testid="stForm"]').filter(has_text="Add to-do")
    field = form.get_by_label("To-do", exact=True)
    field.fill(SEED_PARENT_ASK)
    field.press("Enter")
    wait_idle(page)
    _expect_text(page, SEED_PARENT_ASK)


def _edit_requirement_label(walker: Walker) -> None:
    """Retype the Precalculus row's Requirement cell in the editor.

    The editor draws on a canvas, and at 400px only Order, Category and
    Requirement are in view, so the cell is found by row: the hidden
    accessibility table gives the row index, and Streamlit's rows are 35px.
    """
    page = walker.page("parent")
    editor = page.locator('[data-testid="stDataFrame"]').first
    editor.scroll_into_view_if_needed()
    rows = editor.locator('tbody [role="row"]').all_inner_texts()
    index = next((i for i, text in enumerate(rows) if "Precalculus" in text), None)
    box = editor.locator("canvas").first.bounding_box()
    if index is None or box is None:
        raise JourneyError("could not find the Precalculus row in the editor")
    y = box["y"] + EDITOR_ROW_PX * (index + 1) + EDITOR_ROW_PX / 2
    # The rightmost column in view is Requirement.  Typing into a selected
    # cell opens its editor with the typed text in place of the old value.
    page.mouse.click(box["x"] + box["width"] - 24, y)
    page.keyboard.type(SEED_LABEL_EDIT, delay=40)
    page.keyboard.press("Enter")
    # The grid hands the edit to the form a moment later; Save before that
    # submits the old rows.
    page.wait_for_timeout(1000)


def journeys() -> tuple[Journey, ...]:
    """The plan's "Journeys after Phase A", in walking order (later ones build on earlier)."""
    student, parent = "student", "parent"
    return (
        Journey("S1", student, "Where she stands, and the next 3 things", Budget(0), (
            Step("Open the app", "open", _open(student, "my_path")),
            Step("Progress and Coming up on screen", "check", _s1_check),
        )),
        Journey("S2", student, "Tick a to-do done", Budget(1), (
            Step("Open the app", "open", _open(student, "my_path")),
            Step("Tick Take the PSAT in Coming up", "tap", _s2_tick),
        )),
        Journey("P1", parent, "Find, save an award, assign a step to her", Budget(3, navs=1), (
            Step("Open the app", "open", _open(parent, "my_path")),
            Step("Find Scholarships through the nav", "nav", _open(parent, "find")),
            Step("Save the first award", "tap", _tap(
                parent, lambda page: page.locator(SAVE_BUTTON_SELECTOR).first, "Saved",
            )),
            Step("Suggest it to her", "tap", _tap(
                parent,
                lambda page: page.locator('[class*="st-key-suggest_"] button').first,
                "Suggested",
            )),
            Step("Her To-dos show it", "check", _open(student, "todos", "from your parents"),
                 mode=student),
        )),
        Journey("P2", parent, "Add a program; it shows in her Opportunities", Budget(1, forms=1), (
            Step("Open Catalog & Inbox", "open", _open(parent, "catalog_inbox")),
            Step("Fill in a made-up program", "type", _fill_program_form),
            Step("Confirm the form", "form", _submit_program_form),
            Step("Rebuild the snapshot", "tap", _tap(
                parent,
                lambda page: page.get_by_role("button", name="Rebuild snapshot"),
                "Snapshot rebuilt",
            )),
            Step("It is on her Opportunities", "check",
                 _open(student, "opportunities", SEED_PROGRAM), mode=student),
            Step("Opportunities through the nav", "nav", _open(parent, "opportunities")),
            Step("Suggest it to her", "tap", _tap(
                parent,
                lambda page: _program_card(page).get_by_role("button", name="Suggest to her"),
                "On her to-dos",
            )),
            Step("Her card says From your family", "check",
                 _open(student, "opportunities", "From your family"), mode=student),
        )),
        Journey("S3", student, "Read a suggested program, add one to her to-dos",
                Budget(2), (
            Step("Open the app", "open", _open(student, "my_path")),
            Step("Opportunities link", "link", _tap(
                student, _home_link("Opportunities"), "From your family",
            )),
            Step("Add the first open card to my to-dos", "tap", _add_first_opportunity),
        )),
        Journey("S4", student, "Ask her parents for something", Budget(2), (
            Step("Open To-dos", "open", _open(student, "todos")),
            Step("Open Add a to-do", "tap", _tap(student, _expander("Add a to-do"))),
            Step("For Parents", "tap", _tap(
                student,
                lambda page: page.locator('[data-testid="stForm"]').get_by_text(
                    "Parents", exact=True
                ),
            )),
            Step("Type it and press Enter", "type", _ask_parents),
            Step("Parent To-dos show it", "check",
                 _open(parent, "todos", SEED_PARENT_ASK, "from her"), mode=parent),
        )),
        Journey("S5", student, "What If from her GPA row", Budget(1), (
            Step("Open the app", "open", _open(student, "my_path")),
            Step("Try What If on the GPA row", "link", _tap(
                student,
                lambda page: page.get_by_role("link", name="Try What If").first,
                "Your goals",
            )),
        )),
        Journey("S6", student, "Share an essay with her parents", Budget(2), (
            Step("Open Essays", "open", _open(student, "essays")),
            Step("Open the essay", "tap", _tap(student, _expander(SEED_ESSAY[0]))),
            Step("Share with parents", "tap", _tap(
                student,
                lambda page: page.get_by_text("Share with parents", exact=True),
                "Shared",
            )),
            Step("Parent Essays list it", "check",
                 _open(parent, "essays", SEED_ESSAY[0]), mode=parent),
        )),
        Journey("P3", parent, "See what she did this week", Budget(1), (
            Step("Open the app", "open", _open(parent, "my_path", "on track")),
            Step("All to-dos link", "link", _tap(
                parent, lambda page: page.get_by_role("link", name="All to-dos"),
            )),
            Step("Open Done recently", "tap", _tap(
                parent, _expander("Done recently"), "Take the PSAT",
            )),
        )),
        Journey("P4", parent, "Edit the first target school's requirements", Budget(0, forms=1), (
            Step("Open Colleges & Money", "open", _open(parent, "colleges_money")),
            Step("Open the school", "tap", _tap(parent, _expander(SEED_COLLEGE))),
            Step("Rename a requirement", "type", _edit_requirement_label),
            Step("Save requirements", "form", _tap(
                parent, lambda page: page.get_by_role("button", name="Save requirements"),
            )),
            Step("Her My Path shows it", "check",
                 _open(student, "my_path", SEED_LABEL_EDIT), mode=student),
        )),
    )


def demo_profile_in_grade(grade: str) -> dict[str, Any]:
    """The committed demo profile, moved to ``grade`` for this school year."""
    profile = profile_store.load_demo_profile()
    return {
        **profile,
        "student_id": DEFAULT_STUDENT_ID,
        "grade_level": grade,
        "graduation_year": infer_graduation_year(grade),
    }


def prepare_journey_data(root: Path) -> Path:
    """Fill ``root`` with a fresh private dir, a catalog copy and the latest snapshot.

    Returns the private dir.  Nothing is copied from ``data/private/``: the walk
    sees only the demo profile and made-up content.
    """
    private_dir = root / "private"
    profile_store.save_profile(
        demo_profile_in_grade(JOURNEY_GRADE), students_dir=private_dir / "students"
    )
    seed_path(private_dir)
    conn = connect(private_dir / "coach.db")
    try:
        repo.create_essay(conn, DEFAULT_STUDENT_ID, *SEED_ESSAY)
    finally:
        conn.close()

    shutil.copytree(
        CATALOG_DIR, root / "data" / "catalog", ignore=shutil.ignore_patterns("inbox")
    )
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    snapshots = sorted(PROCESSED_DIR.glob("scholarships_snapshot_*.parquet"))
    if not snapshots:
        raise RuntimeError("No snapshot in data/processed/; the journeys need one.")
    shutil.copy2(snapshots[-1], processed / snapshots[-1].name)
    return private_dir


def walk(journey: Journey, walker: Walker, out_dir: Path) -> Cost:
    print(f"{journey.id} ({journey.who}): {journey.title}")
    for index, step in enumerate(journey.steps, start=1):
        mode = step.mode or journey.who
        path = out_dir / step_filename(journey, index, step)
        try:
            step.act(walker)
        except Exception:
            walker.page(mode).screenshot(path=str(path.with_stem(f"{path.stem}_FAILED")))
            raise
        walker.page(mode).screenshot(path=str(path))
        cost = STEP_TAPS[step.kind] if step.mode is None else 0
        print(f"  {index}. {step.name} [{step.kind}, {cost}] -> {path.name}")
    return journey_cost(journey)


def walk_journeys(args: argparse.Namespace) -> int:
    from playwright.sync_api import sync_playwright

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[tuple[Journey, Cost | None, str]] = []
    with tempfile.TemporaryDirectory(prefix="coach_journeys_") as tmp:
        root = Path(tmp)
        private_dir = prepare_journey_data(root)
        with running_app(private_dir, data_root=root) as base_url, sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            walker = Walker(browser, base_url)
            try:
                for journey in journeys():
                    try:
                        results.append((journey, walk(journey, walker, args.out), ""))
                    except Exception as exc:
                        # One broken journey should not hide how the rest went.
                        print(f"  FAILED: {exc}")
                        results.append((journey, None, str(exc)))
            finally:
                walker.close()
                browser.close()

    print("\nJourney  Taps        Budget      Result")
    failed = 0
    for journey, cost, error in sorted(results, key=lambda item: item[0].id):
        budget = budget_text(journey.budget)
        if cost is None:
            failed += 1
            print(f"{journey.id:<8} {'-':<11} {budget:<11} failed: {error}")
            continue
        verdict = "within" if within_budget(cost, journey.budget) else "OVER"
        print(f"{journey.id:<8} {cost_text(cost, journey.budget):<11} {budget:<11} {verdict}")
    print(f"Screenshots in {args.out}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.journeys:
        return walk_journeys(args)
    if args.scratch_db:
        with scratch_private_dir() as private_dir:
            written = sweep(args, private_dir)
    else:
        written = sweep(args, None)
    print(f"Wrote {len(written)} screenshots to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
