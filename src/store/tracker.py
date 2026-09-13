"""The application tracker: save an award, plan it, work it, submit it, score it.

This is the coaching loop.  A ranked card is a suggestion until the student
saves it; from then on the award is a row in ``applications`` with a checklist
built from what the sponsor actually asks for, a status that moves along a
fixed graph, and an outcome when the answer comes back.

Everything here is a function over a connection, so the whole loop is testable
without Streamlit.  The rules it encodes:

* A save is idempotent.  Pressing Save twice on the same card returns the
  application that already exists rather than a duplicate, and never rebuilds
  a checklist the student has been ticking off.
* The checklist comes from the catalog's ``requirements`` object, one item per
  thing a person has to do -- one per essay prompt, one per letter -- because
  "2 letters" is two separate asks of two separate teachers.
* Status moves along :data:`ALLOWED_TRANSITIONS`.  ``won`` and ``lost`` are
  ends, but each can step back to ``submitted``, because a misclick on a
  teenager's laptop should not be permanent.
* ``won`` and ``lost`` are also outcomes, so the status and the ``outcomes``
  row are written together and cannot disagree.
* A recommendation request has its own small lifecycle -- planned, asked,
  received, declined -- because a letter is a second person's work on a second
  person's schedule.  Its due date lands in This Week alongside the student's
  own tasks, since the thing a student has to do about an outstanding letter is
  ask again.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from src.store import repo
from src.store.repo import Application, ChecklistItem

APPLICATION_STATUSES: tuple[str, ...] = (
    "saved",
    "planning",
    "in_progress",
    "submitted",
    "won",
    "lost",
    "skipped",
)

STATUS_LABELS: dict[str, str] = {
    "saved": "Saved",
    "planning": "Planning",
    "in_progress": "In progress",
    "submitted": "Submitted",
    "won": "Won",
    "lost": "Lost",
    "skipped": "Skipped",
}

DEFAULT_STATUS = "saved"

# Statuses that still need work from the student. This Week draws only from
# these; a submitted or decided award has nothing left to do.
OPEN_STATUSES: tuple[str, ...] = ("saved", "planning", "in_progress")

DECIDED_STATUSES: tuple[str, ...] = ("won", "lost")

ALLOWED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "saved": ("planning", "in_progress", "submitted", "skipped"),
    "planning": ("saved", "in_progress", "submitted", "skipped"),
    "in_progress": ("planning", "submitted", "skipped"),
    "submitted": ("in_progress", "won", "lost"),
    "won": ("submitted",),
    "lost": ("submitted",),
    "skipped": ("saved",),
}

# A letter has its own small lifecycle, and its own dates: when the student
# asked, when the teacher owes it, when it landed.  "declined" is a real answer
# and has to be visible, or the student waits on a letter that is not coming.
REQUEST_STATUSES: tuple[str, ...] = ("planned", "asked", "received", "declined")

REQUEST_STATUS_LABELS: dict[str, str] = {
    "planned": "Planned",
    "asked": "Asked",
    "received": "Received",
    "declined": "Declined",
}

DEFAULT_REQUEST_STATUS = "planned"

# Statuses where the letter is still outstanding; This Week draws from these.
OPEN_REQUEST_STATUSES: tuple[str, ...] = ("planned", "asked")

ALLOWED_REQUEST_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "planned": ("asked", "declined"),
    "asked": ("received", "declined", "planned"),
    "received": ("asked",),
    "declined": ("planned",),
}

THIS_WEEK_DAYS = 14

# Truncating in the label keeps a 300-word prompt out of the checklist row; the
# full prompt stays in the catalog and on the sponsor's page.
_PROMPT_LABEL_CHARS = 80


class TransitionError(ValueError):
    """A status change the tracker does not allow."""


@dataclass(frozen=True, slots=True)
class DueItem:
    """One line in This Week: a deadline, a dated checklist item, or a letter."""

    kind: str
    due_on: str
    days_until: int
    application_id: int
    catalog_id: str
    award_title: str
    label: str
    status: str = DEFAULT_STATUS
    done: bool = False

    @property
    def is_overdue(self) -> bool:
        return self.days_until < 0


def normalize_status(value: object) -> str:
    text = str(value or "").strip().casefold().replace(" ", "_")
    return text if text in APPLICATION_STATUSES else DEFAULT_STATUS


def can_transition(current: object, target: object) -> bool:
    """True when ``current`` may become ``target`` (staying put always may)."""
    start = normalize_status(current)
    end = normalize_status(target)
    return end == start or end in ALLOWED_TRANSITIONS[start]


def next_statuses(current: object) -> tuple[str, ...]:
    """The statuses a picker should offer, current first."""
    start = normalize_status(current)
    return (start, *ALLOWED_TRANSITIONS[start])


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _truncate(text: str, limit: int = _PROMPT_LABEL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _letter_count(value: Any) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return max(count, 0)


def checklist_labels(requirements: Mapping[str, Any] | None) -> list[str]:
    """Build the checklist a sponsor's ``requirements`` object implies.

    Only stated requirements produce items.  ``None`` means the catalog does
    not know, and inventing a task out of a null is how a student ends up
    writing an essay nobody asked for.
    """
    if not isinstance(requirements, Mapping):
        return []

    labels: list[str] = []

    prompts = requirements.get("essay_prompts")
    prompt_texts = (
        [_clean(prompt) for prompt in prompts if _clean(prompt)]
        if isinstance(prompts, (list, tuple))
        else []
    )
    if prompt_texts:
        for index, prompt in enumerate(prompt_texts, start=1):
            prefix = f"Essay {index}" if len(prompt_texts) > 1 else "Essay"
            labels.append(f"{prefix}: {_truncate(prompt)}")
    elif requirements.get("essay") is True:
        labels.append("Write the essay")

    letters = _letter_count(requirements.get("recommendation_letters"))
    for index in range(1, letters + 1):
        suffix = f" {index} of {letters}" if letters > 1 else ""
        labels.append(f"Recommendation letter{suffix}")

    if requirements.get("transcript") is True:
        labels.append("Request transcript")
    if requirements.get("fafsa") is True:
        labels.append("Complete the FAFSA")
    if requirements.get("video_or_portfolio") is True:
        labels.append("Record video or assemble portfolio")
    if requirements.get("interview") is True:
        labels.append("Prepare for the interview")

    return labels


def save_award(
    conn: Any,
    student_id: str,
    catalog_id: str,
    *,
    title: str = "",
    source_url: str = "",
    deadline: str | None = None,
    requirements: Mapping[str, Any] | None = None,
) -> tuple[Application, bool]:
    """Save an award for a student, returning ``(application, created)``.

    An award already saved comes back untouched, checklist included.
    """
    existing = repo.find_application(conn, student_id, catalog_id)
    if existing is not None:
        return existing, False

    application = repo.create_application(
        conn,
        student_id,
        catalog_id,
        status=DEFAULT_STATUS,
        title=_clean(title),
        source_url=_clean(source_url),
        deadline=_clean(deadline) or None,
    )
    for position, label in enumerate(checklist_labels(requirements)):
        repo.create_checklist_item(conn, application.id, label, position=position)
    return application, True


def add_checklist_item(
    conn: Any, application_id: int, label: str, due_on: str | None = None
) -> ChecklistItem:
    """Append a free-form item after whatever the template generated."""
    existing = repo.list_checklist_items(conn, application_id)
    position = max((item.position for item in existing), default=-1) + 1
    return repo.create_checklist_item(
        conn, application_id, _clean(label), due_on=_clean(due_on) or None, position=position
    )


def checklist_progress(items: list[ChecklistItem]) -> tuple[int, int]:
    """Return ``(done, total)`` for a checklist."""
    return sum(1 for item in items if item.done), len(items)


def set_status(
    conn: Any, application_id: int, status: str, *, today: date | None = None
) -> Application:
    """Move an application's status, keeping dates and outcomes consistent."""
    application = repo.get_application(conn, application_id)
    if application is None:
        raise TransitionError(f"No application {application_id}.")

    current = normalize_status(application.status)
    target = normalize_status(status)
    if not can_transition(current, target):
        raise TransitionError(
            f"{STATUS_LABELS[current]} cannot become {STATUS_LABELS[target]}."
        )
    if target == current:
        return application

    stamp = (today or date.today()).isoformat()
    changes: dict[str, Any] = {"status": target}
    if target == "submitted" and not application.submitted_on:
        changes["submitted_on"] = stamp
    repo.update_application(conn, application_id, **changes)

    if target in DECIDED_STATUSES:
        prior = repo.get_outcome(conn, application_id)
        repo.set_outcome(
            conn,
            application_id,
            result=target,
            amount_awarded=None if prior is None else prior.amount_awarded,
            paid_to="" if prior is None else prior.paid_to,
            renewal_terms="" if prior is None else prior.renewal_terms,
            decided_on=stamp if prior is None or not prior.decided_on else prior.decided_on,
            notes="" if prior is None else prior.notes,
        )

    updated = repo.get_application(conn, application_id)
    if updated is None:  # pragma: no cover - the update above guarantees a row
        raise TransitionError(f"Application {application_id} vanished mid-update.")
    return updated


def record_outcome(
    conn: Any,
    application_id: int,
    result: str,
    *,
    amount_awarded: float | None = None,
    paid_to: str = "",
    renewal_terms: str = "",
    decided_on: str | None = None,
    notes: str = "",
    today: date | None = None,
) -> repo.Outcome:
    """Write the result of an application and move its status to match."""
    decision = normalize_status(result)
    if decision not in DECIDED_STATUSES:
        raise TransitionError(f"{result!r} is not an outcome; use won or lost.")

    application = repo.get_application(conn, application_id)
    if application is None:
        raise TransitionError(f"No application {application_id}.")

    # A student who forgot to mark the award submitted still gets to record the
    # result; walk them through submitted rather than refusing the news.
    if not can_transition(application.status, decision):
        set_status(conn, application_id, "submitted", today=today)
    set_status(conn, application_id, decision, today=today)

    stamp = (today or date.today()).isoformat()
    return repo.set_outcome(
        conn,
        application_id,
        result=decision,
        amount_awarded=amount_awarded,
        paid_to=paid_to,
        renewal_terms=renewal_terms,
        decided_on=_clean(decided_on) or stamp,
        notes=notes,
    )


def normalize_request_status(value: object) -> str:
    text = str(value or "").strip().casefold().replace(" ", "_")
    return text if text in REQUEST_STATUSES else DEFAULT_REQUEST_STATUS


def can_transition_request(current: object, target: object) -> bool:
    start = normalize_request_status(current)
    end = normalize_request_status(target)
    return end == start or end in ALLOWED_REQUEST_TRANSITIONS[start]


def next_request_statuses(current: object) -> tuple[str, ...]:
    """The request statuses a picker should offer, current first."""
    start = normalize_request_status(current)
    return (start, *ALLOWED_REQUEST_TRANSITIONS[start])


def set_request_status(
    conn: Any, request_id: int, status: str, *, today: date | None = None
) -> repo.RecommendationRequest:
    """Move a letter request along, stamping the date the move implies.

    Each date is stamped once: a request that goes back to ``asked`` to correct
    a misclick keeps the day the student actually asked.
    """
    request = repo.get_recommendation_request(conn, request_id)
    if request is None:
        raise TransitionError(f"No recommendation request {request_id}.")

    current = normalize_request_status(request.status)
    target = normalize_request_status(status)
    if not can_transition_request(current, target):
        raise TransitionError(
            f"{REQUEST_STATUS_LABELS[current]} cannot become "
            f"{REQUEST_STATUS_LABELS[target]}."
        )
    if target == current:
        return request

    stamp = (today or date.today()).isoformat()
    changes: dict[str, Any] = {"status": target}
    if target == "asked" and not request.asked_on:
        changes["asked_on"] = stamp
    if target == "received" and not request.received_on:
        changes["received_on"] = stamp
    repo.update_recommendation_request(conn, request_id, **changes)

    updated = repo.get_recommendation_request(conn, request_id)
    if updated is None:  # pragma: no cover - the update above guarantees a row
        raise TransitionError(f"Recommendation request {request_id} vanished mid-update.")
    return updated


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def this_week(
    conn: Any,
    student_id: str,
    *,
    today: date | None = None,
    within_days: int = THIS_WEEK_DAYS,
) -> list[DueItem]:
    """Everything due in the next ``within_days`` days, overdue work included.

    Overdue items stay on the list.  A tracker that quietly drops a task the
    day it goes red is worse than no tracker.
    """
    reference = today or date.today()
    horizon = reference + timedelta(days=within_days)
    items: list[DueItem] = []

    for application in repo.list_applications(conn, student_id):
        status = normalize_status(application.status)
        if status not in OPEN_STATUSES:
            continue
        award_title = application.title or application.catalog_id

        deadline = _parse_date(application.deadline or "")
        if deadline is not None and deadline <= horizon:
            items.append(
                DueItem(
                    kind="deadline",
                    due_on=deadline.isoformat(),
                    days_until=(deadline - reference).days,
                    application_id=application.id,
                    catalog_id=application.catalog_id,
                    award_title=award_title,
                    label="Application deadline",
                    status=status,
                )
            )

        for request in repo.list_recommendation_requests(conn, application_id=application.id):
            if normalize_request_status(request.status) not in OPEN_REQUEST_STATUSES:
                continue
            due = _parse_date(request.due_on or "")
            if due is None or due > horizon:
                continue
            recommender = repo.get_recommender(conn, request.recommender_id)
            who = recommender.name if recommender is not None else "a recommender"
            state = REQUEST_STATUS_LABELS[normalize_request_status(request.status)]
            items.append(
                DueItem(
                    kind="letter",
                    due_on=due.isoformat(),
                    days_until=(due - reference).days,
                    application_id=application.id,
                    catalog_id=application.catalog_id,
                    award_title=award_title,
                    label=f"Letter from {who} ({state.casefold()})",
                    status=status,
                )
            )

        for item in repo.list_checklist_items(conn, application.id):
            if item.done:
                continue
            due = _parse_date(item.due_on or "")
            if due is None or due > horizon:
                continue
            items.append(
                DueItem(
                    kind="checklist",
                    due_on=due.isoformat(),
                    days_until=(due - reference).days,
                    application_id=application.id,
                    catalog_id=application.catalog_id,
                    award_title=award_title,
                    label=item.label,
                    status=status,
                )
            )

    items.sort(key=lambda due: (due.due_on, due.award_title, due.kind, due.label))
    return items
