"""One dated feed over everything the family has to plan around, and its ICS export.

This Week answers "what is due in the next fortnight".  The timeline answers
the other question -- what the next three years look like -- and it needs every
dated thing in one shape: application deadlines, the checklist items a student
put a date on, the letters a teacher owes, and the general milestones from
:mod:`src.store.milestones`.  :class:`CalendarEvent` is that shape.

Buckets here mean school years, not open-or-closed listings.  ``now`` is the
school year in progress (a date already past stays in it, because an overdue
thing is still this year's problem), ``next_cycle`` is the one after, and
``senior_year`` is everything further out.  Catalog awards arrive with a bucket
already assigned by :mod:`src.rank.timeline` and keep it, so the award a
sophomore cannot apply for until senior year stays where ranking put it.

The ICS writer is deliberately stdlib-only and emits all-day events.  A
deadline is a day, not a time, and inventing 9am for it would put the wrong
thing on a phone.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.profile.grade_levels import school_year_end
from src.store import repo, tracker
from src.store.milestones import (
    Milestone,
    load_milestones,
    occurrences_for_school_year,
)

# now, next_cycle and senior_year are the three the timeline plans in; the
# other two timeline buckets describe an award, not a date on a calendar.
CALENDAR_BUCKETS: tuple[str, ...] = ("now", "next_cycle", "senior_year")

EVENT_KINDS: tuple[str, ...] = ("deadline", "checklist", "letter", "milestone", "award")

EVENT_KIND_LABELS: dict[str, str] = {
    "deadline": "Application deadline",
    "checklist": "Task",
    "letter": "Letter",
    "milestone": "Milestone",
    "award": "Award deadline",
}

# How far ahead the feed plans when the student's grade does not end it sooner.
DEFAULT_HORIZON_YEARS = 3

PRODUCT_ID = "-//ScholarshipCoach//Family Calendar//EN"
CALENDAR_NAME = "Scholarship Coach"
UID_DOMAIN = "scholarshipcoach.local"

_ICS_LINE_OCTETS = 75


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """One dated thing on the family's calendar."""

    uid: str
    kind: str
    title: str
    starts_on: date
    ends_on: date | None = None
    detail: str = ""
    note: str = ""
    bucket: str = "now"

    @property
    def kind_label(self) -> str:
        return EVENT_KIND_LABELS.get(self.kind, self.kind)

    @property
    def last_day(self) -> date:
        return self.ends_on or self.starts_on

    def days_until(self, today: date) -> int:
        return (self.starts_on - today).days


def bucket_for_date(when: date, today: date) -> str:
    """Place a date in a school-year bucket, folding the past into ``now``."""
    distance = school_year_end(when) - school_year_end(today)
    if distance <= 0:
        return "now"
    if distance == 1:
        return "next_cycle"
    return "senior_year"


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def milestone_events(
    milestones: Iterable[Milestone],
    grade_level: str | None,
    *,
    today: date,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
) -> list[CalendarEvent]:
    """Milestone occurrences for this school year and the next ``horizon_years``."""
    rows = list(milestones)
    current = school_year_end(today)
    events: list[CalendarEvent] = []
    for offset in range(horizon_years + 1):
        year_end = current + offset
        dated = occurrences_for_school_year(rows, grade_level, year_end, today=today)
        if not dated:
            # Either the student has graduated out of this year or nothing
            # applies to the grade they will be in; either way, no events.
            continue
        for item in dated:
            events.append(
                CalendarEvent(
                    uid=f"milestone-{item.milestone.id}-{year_end}",
                    kind="milestone",
                    title=item.milestone.title,
                    starts_on=item.starts_on,
                    ends_on=item.ends_on if item.is_window else None,
                    # The dates already say it is a window, and repeating the
                    # title as the detail reads as a stutter in both surfaces.
                    detail="Window" if item.is_window else "",
                    note=item.milestone.note,
                    bucket=bucket_for_date(item.starts_on, today),
                )
            )
    return events


def application_events(conn: Any, student_id: str, *, today: date) -> list[CalendarEvent]:
    """Deadlines, dated tasks and outstanding letters on the family's own saved awards.

    Only applications still open produce events.  A submitted or decided award
    has nothing left to plan, and its deadline on the calendar is clutter.
    """
    events: list[CalendarEvent] = []
    for application in repo.list_applications(conn, student_id):
        status = tracker.normalize_status(application.status)
        if status not in tracker.OPEN_STATUSES:
            continue
        award_title = application.title or application.catalog_id

        deadline = _parse_date(application.deadline or "")
        if deadline is not None:
            events.append(
                CalendarEvent(
                    uid=f"application-{application.id}",
                    kind="deadline",
                    title=award_title,
                    starts_on=deadline,
                    detail="Application deadline",
                    bucket=bucket_for_date(deadline, today),
                )
            )

        for item in repo.list_checklist_items(conn, application.id):
            due = _parse_date(item.due_on or "")
            if item.done or due is None:
                continue
            events.append(
                CalendarEvent(
                    uid=f"checklist-{item.id}",
                    kind="checklist",
                    title=item.label,
                    starts_on=due,
                    detail=award_title,
                    bucket=bucket_for_date(due, today),
                )
            )

        for request in repo.list_recommendation_requests(conn, application_id=application.id):
            state = tracker.normalize_request_status(request.status)
            due = _parse_date(request.due_on or "")
            if state not in tracker.OPEN_REQUEST_STATUSES or due is None:
                continue
            recommender = repo.get_recommender(conn, request.recommender_id)
            who = recommender.name if recommender is not None else "a recommender"
            label = tracker.REQUEST_STATUS_LABELS[state].casefold()
            events.append(
                CalendarEvent(
                    uid=f"letter-{request.id}",
                    kind="letter",
                    title=f"Letter from {who} ({label})",
                    starts_on=due,
                    detail=award_title,
                    bucket=bucket_for_date(due, today),
                )
            )

    return events


def family_events(
    conn: Any,
    student_id: str,
    *,
    grade_level: str | None = None,
    today: date | None = None,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    milestones: Iterable[Milestone] | None = None,
) -> list[CalendarEvent]:
    """Everything the family owns a date for: their applications plus milestones."""
    reference = today or date.today()
    rows = list(milestones) if milestones is not None else load_milestones(conn)
    events = application_events(conn, student_id, today=reference)
    events.extend(
        milestone_events(rows, grade_level, today=reference, horizon_years=horizon_years)
    )
    return sort_events(events)


def sort_events(events: Iterable[CalendarEvent]) -> list[CalendarEvent]:
    return sorted(events, key=lambda event: (event.starts_on, event.kind, event.title))


def events_in_bucket(events: Iterable[CalendarEvent], bucket: str) -> list[CalendarEvent]:
    return [event for event in events if event.bucket == bucket]


def group_by_month(
    events: Iterable[CalendarEvent],
) -> dict[tuple[int, int], list[CalendarEvent]]:
    """Group events into ``(year, month)`` buckets, months in date order."""
    grouped: dict[tuple[int, int], list[CalendarEvent]] = {}
    for event in sort_events(events):
        grouped.setdefault((event.starts_on.year, event.starts_on.month), []).append(event)
    return grouped


def group_by_school_year(
    events: Iterable[CalendarEvent],
) -> dict[int, list[CalendarEvent]]:
    """Group events by the school year they fall in, years in order."""
    grouped: dict[int, list[CalendarEvent]] = {}
    for event in sort_events(events):
        grouped.setdefault(school_year_end(event.starts_on), []).append(event)
    return dict(sorted(grouped.items()))


def _escape(text: str) -> str:
    """Escape an ICS TEXT value (RFC 5545 section 3.3.11)."""
    return (
        str(text or "")
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _fold(line: str) -> list[str]:
    """Fold a content line to 75 octets, continuations starting with a space."""
    encoded = line.encode("utf-8")
    if len(encoded) <= _ICS_LINE_OCTETS:
        return [line]

    chunks: list[str] = []
    buffer = b""
    limit = _ICS_LINE_OCTETS
    for character in line:
        octets = character.encode("utf-8")
        if len(buffer) + len(octets) > limit:
            chunks.append(buffer.decode("utf-8"))
            buffer = b""
            # A continuation line spends one octet on its leading space.
            limit = _ICS_LINE_OCTETS - 1
        buffer += octets
    if buffer:
        chunks.append(buffer.decode("utf-8"))
    return [chunks[0], *(f" {chunk}" for chunk in chunks[1:])]


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def to_ics(
    events: Iterable[CalendarEvent],
    *,
    calendar_name: str = CALENDAR_NAME,
    now: datetime | None = None,
) -> str:
    """Render events as an RFC 5545 calendar of all-day events.

    ``DTEND`` on an all-day event is exclusive, so a one-day deadline ends the
    following day -- get that wrong and every deadline shows up a day short.
    """
    moment = now or datetime.now(timezone.utc)
    stamp = _stamp(moment)

    lines: list[str] = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODUCT_ID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        f"X-WR-CALNAME:{_escape(calendar_name)}",
    ]

    for event in sort_events(events):
        summary = f"{event.kind_label}: {event.title}"
        description = " ".join(part for part in (event.detail, event.note) if part).strip()
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{event.uid}@{UID_DOMAIN}",
                f"DTSTAMP:{stamp}",
                f"DTSTART;VALUE=DATE:{event.starts_on.strftime('%Y%m%d')}",
                f"DTEND;VALUE=DATE:{(event.last_day + timedelta(days=1)).strftime('%Y%m%d')}",
                f"SUMMARY:{_escape(summary)}",
            ]
        )
        if description:
            lines.append(f"DESCRIPTION:{_escape(description)}")
        lines.extend(["TRANSP:TRANSPARENT", "END:VEVENT"])

    lines.append("END:VCALENDAR")

    folded: list[str] = []
    for line in lines:
        folded.extend(_fold(line))
    return "\r\n".join(folded) + "\r\n"
