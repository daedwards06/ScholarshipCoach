"""The essay bank: a few themed drafts, reused across many prompts.

Reuse is the whole point.  A student who writes one strong "challenge" essay
answers a dozen prompts with it, so the bank is organised by theme rather than
by award, and every essay carries the count of applications it is already
serving.

The other half is the pairing.  ``checklist_items`` already holds one row per
essay prompt an award asks for (:mod:`src.store.tracker` builds them), and
``essay_links`` pairs an essay with an application's prompt.  A prompt with no
link is work the student still has to do, so this module reads the two tables
together into :class:`PromptSlot` and calls that state *open*.  Linking an
essay to a slot ticks its checklist item and unlinking unticks it --- an
answered prompt is a finished task, and letting the two disagree is how a
student ends up rewriting an essay they already have.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.store import repo
from src.store.repo import Essay

# The labels tracker.checklist_labels writes: "Essay: <prompt>" for a single
# prompt, "Essay 2: <prompt>" when an award asks for several.
_PROMPT_LABEL_RE = re.compile(r"^Essay(?:\s+\d+)?:\s*(?P<prompt>.+)$")

# The award states an essay prompt but the catalog has no prompt text.
_UNTITLED_PROMPT_LABEL = "Write the essay"


def prompt_from_label(label: str) -> str | None:
    """Return the prompt a checklist label states, or ``None`` if it is not one."""
    text = str(label or "").strip()
    if text == _UNTITLED_PROMPT_LABEL:
        return text
    match = _PROMPT_LABEL_RE.match(text)
    return match.group("prompt").strip() if match else None


@dataclass(frozen=True, slots=True)
class PromptSlot:
    """One essay prompt on one application, with the essay answering it."""

    item_id: int
    application_id: int
    label: str
    prompt: str
    done: bool = False
    essay_id: int | None = None
    essay_title: str = ""

    @property
    def is_open(self) -> bool:
        return self.essay_id is None


@dataclass(frozen=True, slots=True)
class BankEntry:
    """An essay plus what it is doing: its theme, length and reuse."""

    essay: Essay
    reuse_count: int = 0
    used_by: tuple[str, ...] = ()

    @property
    def theme_label(self) -> str:
        return repo.ESSAY_THEME_LABELS[repo.normalize_theme(self.essay.theme)]


def prompt_slots(conn: Any, application_id: int) -> list[PromptSlot]:
    """The award's essay prompts, each with its linked essay if it has one."""
    links = repo.list_essay_links(conn, application_id=application_id)
    by_prompt = {link.prompt: link for link in links if link.prompt}
    titles = {
        essay.id: essay.title
        for essay in (repo.get_essay(conn, link.essay_id) for link in links)
        if essay is not None
    }

    slots: list[PromptSlot] = []
    for item in repo.list_checklist_items(conn, application_id):
        prompt = prompt_from_label(item.label)
        if prompt is None:
            continue
        link = by_prompt.get(prompt)
        slots.append(
            PromptSlot(
                item_id=item.id,
                application_id=application_id,
                label=item.label,
                prompt=prompt,
                done=item.done,
                essay_id=None if link is None else link.essay_id,
                essay_title="" if link is None else titles.get(link.essay_id, ""),
            )
        )
    return slots


def open_prompt_slots(conn: Any, application_id: int) -> list[PromptSlot]:
    """The prompts on an application that no essay answers yet."""
    return [slot for slot in prompt_slots(conn, application_id) if slot.is_open]


def use_essay_for_prompt(conn: Any, essay_id: int, slot: PromptSlot) -> None:
    """Answer a prompt with an essay, ticking the checklist item it belongs to.

    ``essay_links`` is unique on ``(essay_id, application_id)``, so one essay
    answers one prompt per award; pointing it at a second prompt on the same
    award moves the link rather than duplicating it.
    """
    repo.link_essay(conn, essay_id, slot.application_id, slot.prompt)
    if not slot.done:
        repo.update_checklist_item(conn, slot.item_id, done=True)


def clear_prompt(conn: Any, slot: PromptSlot) -> None:
    """Drop the essay answering a prompt, putting the task back on the list."""
    if slot.essay_id is not None:
        repo.unlink_essay(conn, slot.essay_id, slot.application_id)
    if slot.done:
        repo.update_checklist_item(conn, slot.item_id, done=False)


def essay_bank(conn: Any, student_id: str) -> list[BankEntry]:
    """Every essay the student has, with its reuse count and the awards using it."""
    counts = repo.essay_reuse_counts(conn, student_id)
    entries: list[BankEntry] = []
    for essay in repo.list_essays(conn, student_id):
        used_by: list[str] = []
        for link in repo.list_essay_links(conn, essay_id=essay.id):
            application = repo.get_application(conn, link.application_id)
            if application is not None:
                used_by.append(application.title or application.catalog_id)
        entries.append(
            BankEntry(essay=essay, reuse_count=counts.get(essay.id, 0), used_by=tuple(used_by))
        )
    return entries


def theme_counts(entries: list[BankEntry]) -> dict[str, int]:
    """How many essays sit under each theme, in the canonical theme order."""
    counts = dict.fromkeys(repo.ESSAY_THEMES, 0)
    for entry in entries:
        counts[repo.normalize_theme(entry.essay.theme)] += 1
    return counts
