from __future__ import annotations

import calendar
from datetime import date

import pandas as pd
import streamlit as st

from app import modes, state
from app.helpers import (
    award_count_text,
    format_amount_range,
    needs_date_awards,
    row_catalog_id,
    timeline_deadline,
    urgency_indicator,
)
from src.rank.timeline import TIMELINE_BUCKET_LABELS
from src.store import calendar_feed, milestones
from src.store.db import open_db
from src.text_utils import coerce_text

# The timeline shows catalog awards alongside the family's own dates. A whole
# eligible catalog would bury them, so only the nearest deadlines in each
# bucket are drawn.
TIMELINE_AWARDS_PER_BUCKET = 40
CALENDAR_FILE_NAME = "scholarship_coach.ics"


def _award_calendar_events(today_value: date) -> list[calendar_feed.CalendarEvent]:
    """Catalog awards as calendar events, keeping the bucket ranking gave them."""
    eligible_df = st.session_state.get("eligible_df")
    if not isinstance(eligible_df, pd.DataFrame) or eligible_df.empty:
        return []

    per_bucket: dict[str, list[calendar_feed.CalendarEvent]] = {}
    for _, row in eligible_df.iterrows():
        bucket = coerce_text(row.get("timeline_bucket")) or "now"
        if bucket not in calendar_feed.CALENDAR_BUCKETS:
            continue
        deadline_text, is_projected = timeline_deadline(row, today_value)
        try:
            starts_on = date.fromisoformat(str(deadline_text)[:10])
        except ValueError:
            continue
        catalog_id = row_catalog_id(row) or coerce_text(row.get("title"))
        per_bucket.setdefault(bucket, []).append(
            calendar_feed.CalendarEvent(
                uid=f"award-{catalog_id}",
                kind="award",
                title=coerce_text(row.get("title")) or catalog_id,
                starts_on=starts_on,
                detail=coerce_text(row.get("sponsor")),
                note="Projected from this award's usual cycle" if is_projected else "",
                bucket=bucket,
            )
        )

    events: list[calendar_feed.CalendarEvent] = []
    for bucket_events in per_bucket.values():
        events.extend(
            calendar_feed.sort_events(bucket_events)[:TIMELINE_AWARDS_PER_BUCKET]
        )
    return events


def _render_calendar_event(event: calendar_feed.CalendarEvent, today_value: date) -> None:
    when = event.starts_on.strftime("%a %d")
    if event.ends_on is not None and event.ends_on != event.starts_on:
        when = f"{when} – {event.ends_on.strftime('%a %d')}"
    urgency_text, _ = urgency_indicator(event.days_until(today_value))

    col_when, col_what = st.columns([0.18, 0.82])
    with col_when:
        st.markdown(f"**{when}**")
    with col_what:
        st.markdown(f"{event.kind_label}: {event.title}")
        caption = " · ".join(part for part in (event.detail, event.note) if part)
        if caption:
            st.caption(caption)
        if event.bucket == "now":
            st.caption(urgency_text)


def _render_timeline_bucket(
    events: list[calendar_feed.CalendarEvent], today_value: date
) -> None:
    if not events:
        st.info("Nothing scheduled in this stretch yet.")
        return

    for year_end, year_events in calendar_feed.group_by_school_year(events).items():
        st.markdown(f"**{milestones.school_year_label(year_end)} school year**")
        for (year, month), month_events in calendar_feed.group_by_month(year_events).items():
            with st.container(border=True):
                st.markdown(f"##### {calendar.month_name[month]} {year}")
                for event in month_events:
                    _render_calendar_event(event, today_value)


def _render_needs_date_bucket() -> None:
    """Awards the catalog has no date for, biggest first, each with its source."""
    waiting = needs_date_awards(st.session_state.get("eligible_df"))
    if waiting.empty:
        st.info("Every award you qualify for has a date on record.")
        return

    st.caption(
        f"{award_count_text(len(waiting))} with no deadline and no cycle month on "
        "record. Check the source, then add the date under My Applications."
    )
    for _, row in waiting.iterrows():
        catalog_id = row_catalog_id(row)
        with st.container(border=True):
            col_award, col_link = st.columns([0.75, 0.25])
            with col_award:
                st.markdown(f"**{coerce_text(row.get('title')) or catalog_id}**")
                sponsor = coerce_text(row.get("sponsor"))
                amount = format_amount_range(row.get("amount_min"), row.get("amount_max"))
                st.caption(" · ".join(part for part in (sponsor, amount) if part))
            with col_link:
                source_url = coerce_text(row.get("source_url"))
                if source_url:
                    st.link_button("Look it up", source_url, use_container_width=True)
                else:
                    st.caption("No source link")


def render() -> None:
    st.subheader(modes.SECTION_LABELS["timeline"])
    today_value = state.effective_today(st.session_state.profile)
    grade_level = str(st.session_state.profile.get("grade_level") or "")

    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            events = calendar_feed.family_events(
                conn, student_id, grade_level=grade_level or None, today=today_value
            )
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    family_only = list(events)
    events = calendar_feed.sort_events([*events, *_award_calendar_events(today_value)])

    st.caption(
        "Deadlines, tasks, letters and milestones by month. Milestone dates are typical, "
        "not guaranteed — confirm each one with the college or agency."
    )
    if st.session_state.get("eligible_df") is None:
        st.caption("Run the pipeline under Find Scholarships to add catalog award cycles here.")

    st.download_button(
        "Download calendar (.ics)",
        data=calendar_feed.to_ics(family_only),
        file_name=CALENDAR_FILE_NAME,
        mime="text/calendar",
        help="Saved applications, their tasks and letters, and the milestones that apply.",
    )

    tabs = st.tabs(
        [
            TIMELINE_BUCKET_LABELS.get(bucket, bucket)
            for bucket in (*calendar_feed.CALENDAR_BUCKETS, "needs_date")
        ]
    )
    for tab, bucket in zip(tabs, calendar_feed.CALENDAR_BUCKETS):
        with tab:
            _render_timeline_bucket(
                calendar_feed.events_in_bucket(events, bucket), today_value
            )
    with tabs[-1]:
        _render_needs_date_bucket()
