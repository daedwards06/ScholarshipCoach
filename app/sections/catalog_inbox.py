from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app import modes, state
from app.helpers import csv_list, csv_text
from scripts.run_ingest import run_ingest
from src.catalog import inbox
from src.catalog import entry as catalog_entry
from src.ingest.prefill import prefill_from_url
from src.ingest.sources.curated_catalog import CuratedCatalogSource
from src.normalize.catalog_schema import RECORDS_DIR
from src.profile.grade_levels import GRADE_LABEL_TO_LEVELS

CURATED_SOURCE_NAME = CuratedCatalogSource.name
CATALOG_PAGE_KEY = "catalog_page"
CATALOG_PAGE_REQUEST_KEY = "catalog_page_request"
CATALOG_FORM_KEY = "catalog_form"
CATALOG_EDIT_KEY = "catalog_editing_proposal"
CATALOG_NOTICE_KEY = "catalog_notice"
CATALOG_PAGES: tuple[str, ...] = ("add", "inbox")
CATALOG_PAGE_LABELS = {"add": "Add an award", "inbox": "Inbox"}

# A catalog field is unknown until the page says otherwise, which is a different
# thing from the profile's "prefer not to say".
CATALOG_TRISTATE_OPTIONS = ("Not stated", "Yes", "No")
GRADE_LEVEL_LABELS = {
    level: label
    for label, (_, level) in GRADE_LABEL_TO_LEVELS.items()
    if level is not None
}

REQUIREMENT_LABELS = {
    "essay": "Essay",
    "transcript": "Transcript",
    "fafsa": "FAFSA",
    "video_or_portfolio": "Video or portfolio",
    "interview": "Interview",
}

TRUST_LABELS = {
    "verified_local": "Verified by us",
    "structured_feed": "Structured feed",
    "aggregator": "Aggregator",
    "unverified": "Unverified",
}

PROPOSAL_KIND_LABELS = {
    "prefill": "From a pasted URL",
    "feed": "From a feed",
    "reverify": "Re-verification found a change",
    "manual": "Entered by hand",
}


def _catalog_tristate_label(value: bool | None) -> str:
    if value is None:
        return CATALOG_TRISTATE_OPTIONS[0]
    return "Yes" if value else "No"


def _catalog_tristate_value(label: str) -> bool | None:
    if label == "Yes":
        return True
    if label == "No":
        return False
    return None


def _tristate_select(label: str, value: bool | None, *, help: str | None = None) -> bool | None:
    choice = st.selectbox(
        label,
        options=CATALOG_TRISTATE_OPTIONS,
        index=CATALOG_TRISTATE_OPTIONS.index(_catalog_tristate_label(value)),
        help=help,
    )
    return _catalog_tristate_value(str(choice))


def _month_select(label: str, value: int | None) -> int | None:
    options = (0, *catalog_entry.MONTH_OPTIONS)
    choice = st.selectbox(
        label,
        options=options,
        index=options.index(value) if value in options else 0,
        format_func=lambda number: "—" if not number else calendar.month_name[int(number)],
    )
    return int(choice) or None


def _with_existing(options: tuple[str, ...], selected: list[str]) -> list[str]:
    """Offer the vocabulary plus whatever this record already carries.

    Editing a record must never silently drop a value just because it predates
    the dropdown's list.
    """
    return list(dict.fromkeys([*options, *selected]))


def _catalog_form_values() -> dict[str, Any]:
    values = st.session_state.get(CATALOG_FORM_KEY)
    if not isinstance(values, dict):
        values = catalog_entry.blank_form()
        st.session_state[CATALOG_FORM_KEY] = values
    return values


def _load_catalog_form(values: dict[str, Any], *, proposal_id: str | None = None) -> None:
    st.session_state[CATALOG_FORM_KEY] = values
    st.session_state[CATALOG_EDIT_KEY] = proposal_id
    # The page selector owns its key, so a switch asked for from the inbox has
    # to wait until just before that widget is drawn on the next run.
    st.session_state[CATALOG_PAGE_REQUEST_KEY] = "add"


def _set_catalog_notice(level: str, message: str) -> None:
    st.session_state[CATALOG_NOTICE_KEY] = (level, message)


def _render_catalog_notice() -> None:
    notice = st.session_state.pop(CATALOG_NOTICE_KEY, None)
    if not notice:
        return
    level, message = notice
    {"success": st.success, "warning": st.warning, "error": st.error}.get(level, st.info)(message)


def _render_prefill_input() -> None:
    col_url, col_fetch = st.columns([0.75, 0.25], vertical_alignment="bottom")
    with col_url:
        url = st.text_input(
            "Scholarship URL",
            key="catalog_prefill_url",
            placeholder="https://example.org/scholarship",
            help="Paste a listing to prefill the form. Leave it empty to type an award in by hand.",
        )
    with col_fetch:
        fetch = st.button("Read the page", use_container_width=True)

    if not fetch:
        return
    if not url.strip():
        st.warning("Paste a URL first, or fill the form in by hand below.")
        return

    with st.spinner("Reading the page…"):
        result = prefill_from_url(url.strip())
    _load_catalog_form(catalog_entry.form_from_prefill(result.to_form_dict()))
    if result.error:
        _set_catalog_notice(
            "warning",
            f"Could not read that page ({result.error}). The URL is filled in; the rest is by hand.",
        )
    else:
        _set_catalog_notice("success", "Read the page. Check every field before confirming.")
    st.rerun()


def _render_prefill_candidates(values: dict[str, Any]) -> None:
    deadlines = list(values.get("deadline_candidates") or [])
    amounts = list(values.get("amount_candidates") or [])
    if not deadlines and not amounts:
        return

    with st.container(border=True):
        st.caption("Found on the page. Pick one, or ignore these and type your own.")
        if deadlines:
            col_pick, col_use = st.columns([0.7, 0.3], vertical_alignment="bottom")
            choice = col_pick.selectbox("Dates on the page", options=deadlines)
            if col_use.button("Use as deadline", use_container_width=True):
                values["deadline"] = str(choice)
                st.rerun()
        if amounts:
            col_pick, col_min, col_max = st.columns([0.5, 0.25, 0.25], vertical_alignment="bottom")
            amount = col_pick.selectbox(
                "Amounts on the page",
                options=amounts,
                format_func=lambda value: f"${float(value):,.0f}",
            )
            if col_min.button("Use as minimum", use_container_width=True):
                values["amount_min"] = float(amount)
                st.rerun()
            if col_max.button("Use as maximum", use_container_width=True):
                values["amount_max"] = float(amount)
                st.rerun()

        confidence = values.get("confidence") or {}
        if confidence:
            st.caption(
                "Extractor confidence — "
                + ", ".join(
                    f"{name} {float(score):.2f}" for name, score in sorted(confidence.items())
                )
            )


def _award_form_values(values: dict[str, Any]) -> dict[str, Any]:
    """Draw the entry form and return the values it was submitted with.

    Returns an empty dict while the form has not been submitted.  The widgets
    carry no keys on purpose: the defaults come from ``values``, so loading a
    prefill or a proposal into the form redraws it with the new content.
    """
    form = dict(values)
    with st.form("catalog_entry_form"):
        st.markdown("**The award**")
        col_title, col_id = st.columns(2)
        form["title"] = col_title.text_input("Title", value=values["title"])
        form["catalog_id"] = col_id.text_input(
            "Catalog ID",
            value=values["catalog_id"],
            help="Stable lowercase slug; it is this award's identity across cycles. Leave it to derive one from the title.",
        )
        col_sponsor, col_url = st.columns(2)
        form["sponsor"] = col_sponsor.text_input("Sponsor", value=values["sponsor"])
        form["source_url"] = col_url.text_input("Source URL", value=values["source_url"])
        form["description"] = st.text_area("Description", value=values["description"], height=80)
        form["eligibility_text"] = st.text_area(
            "Eligibility text",
            value=values["eligibility_text"],
            height=80,
            help="The page's own words about who may apply. Stage 2 scores against this.",
        )

        st.markdown("**Money and dates**")
        col_min, col_max, col_status = st.columns(3)
        form["amount_min"] = col_min.number_input(
            "Amount minimum", min_value=0.0, step=500.0, value=values["amount_min"]
        )
        form["amount_max"] = col_max.number_input(
            "Amount maximum", min_value=0.0, step=500.0, value=values["amount_max"]
        )
        status_options = catalog_entry.status_options()
        form["status"] = col_status.selectbox(
            "Status",
            options=status_options,
            index=status_options.index(values["status"]) if values["status"] in status_options else 0,
        )
        col_deadline, col_opens, col_month = st.columns(3)
        with col_deadline:
            deadline = st.date_input(
                "Deadline",
                value=date.fromisoformat(values["deadline"]) if values["deadline"] else None,
                format="YYYY-MM-DD",
            )
            form["deadline"] = deadline
        with col_opens:
            form["cycle_opens_month"] = _month_select("Opens in", values["cycle_opens_month"])
        with col_month:
            form["cycle_deadline_month"] = _month_select(
                "Deadline month", values["cycle_deadline_month"]
            )
        form["cycle_recurring"] = _tristate_select(
            "Runs every year", values["cycle_recurring"], help="Drives the projected next deadline."
        )

        st.markdown("**Who it is for**")
        col_level, col_grades = st.columns(2)
        level_options = ("", *catalog_entry.EDUCATION_LEVEL_OPTIONS)
        form["education_level"] = col_level.selectbox(
            "Education level",
            options=level_options,
            index=level_options.index(values["education_level"])
            if values["education_level"] in level_options
            else 0,
            format_func=lambda name: str(name) or "Any",
        )
        form["grade_levels"] = col_grades.multiselect(
            "Grade levels",
            options=catalog_entry.GRADE_LEVEL_OPTIONS,
            default=values["grade_levels"],
            format_func=lambda level: GRADE_LEVEL_LABELS.get(str(level), str(level)),
        )
        form["majors_allowed"] = st.multiselect(
            "Majors allowed (empty means any)",
            options=_with_existing(catalog_entry.MAJOR_OPTIONS, values["majors_allowed"]),
            default=values["majors_allowed"],
            format_func=lambda major: str(major).title(),
        )
        col_states, col_counties = st.columns(2)
        form["states_allowed"] = col_states.multiselect(
            "States allowed (empty means national)",
            options=catalog_entry.STATE_OPTIONS,
            default=values["states_allowed"],
        )
        form["counties_allowed"] = col_counties.multiselect(
            "NC counties allowed",
            options=_with_existing(catalog_entry.COUNTY_OPTIONS, values["counties_allowed"]),
            default=values["counties_allowed"],
        )
        col_gpa, col_sat, col_act = st.columns(3)
        form["min_gpa"] = col_gpa.number_input(
            "Minimum GPA", min_value=0.0, max_value=5.0, step=0.1, value=values["min_gpa"]
        )
        form["sat"] = col_sat.number_input(
            "Minimum SAT", min_value=400, max_value=1600, step=10, value=values["sat"]
        )
        form["act"] = col_act.number_input(
            "Minimum ACT", min_value=1, max_value=36, step=1, value=values["act"]
        )
        col_citizen, col_gender, col_religion = st.columns(3)
        form["citizenship"] = col_citizen.text_input("Citizenship", value=values["citizenship"])
        gender_options = ("", *catalog_entry.gender_options())
        form["gender"] = col_gender.selectbox(
            "Gender",
            options=gender_options,
            index=gender_options.index(values["gender"]) if values["gender"] in gender_options else 0,
            format_func=lambda name: str(name).title() or "Not stated",
        )
        form["religion"] = col_religion.text_input("Religion", value=values["religion"])
        col_need, col_first_gen, col_military, col_disability = st.columns(4)
        with col_need:
            form["need_based"] = _tristate_select("Need based", values["need_based"])
        with col_first_gen:
            form["first_gen_only"] = _tristate_select("First generation only", values["first_gen_only"])
        with col_military:
            form["military_family"] = _tristate_select("Military family", values["military_family"])
        with col_disability:
            form["disability"] = _tristate_select("Disability", values["disability"])
        col_heritage, col_employer, col_membership = st.columns(3)
        form["heritage"] = col_heritage.text_input(
            "Heritage (comma separated)", value=csv_text(values["heritage"])
        )
        form["employer_restricted"] = col_employer.text_input(
            "Employers (comma separated)", value=csv_text(values["employer_restricted"])
        )
        form["membership_required"] = col_membership.text_input(
            "Memberships (comma separated)", value=csv_text(values["membership_required"])
        )

        st.markdown("**What it asks for**")
        requirement_columns = st.columns(len(catalog_entry.REQUIREMENT_FLAGS))
        evidence = values.get("requirement_evidence") or {}
        for column, flag in zip(requirement_columns, catalog_entry.REQUIREMENT_FLAGS, strict=True):
            with column:
                form[f"req_{flag}"] = _tristate_select(
                    REQUIREMENT_LABELS[flag],
                    values[f"req_{flag}"],
                    help=str(evidence.get(flag) or "") or None,
                )
        col_letters, col_renewal = st.columns(2)
        form["recommendation_letters"] = col_letters.number_input(
            "Recommendation letters",
            min_value=0,
            max_value=10,
            step=1,
            value=values["recommendation_letters"],
        )
        form["renewal_terms"] = col_renewal.text_input(
            "Renewal terms", value=values["renewal_terms"]
        )
        form["essay_prompts"] = st.text_area(
            "Essay prompts (one per line)",
            value="\n".join(values["essay_prompts"]),
            height=68,
        )
        form["keywords"] = st.text_input(
            "Keywords (comma separated)", value=csv_text(values["keywords"])
        )

        st.markdown("**Where it came from**")
        col_trust, col_kind = st.columns(2)
        trust_options = catalog_entry.trust_options()
        form["trust"] = col_trust.selectbox(
            "Trust",
            options=trust_options,
            index=trust_options.index(values["trust"]) if values["trust"] in trust_options else 0,
            format_func=lambda name: TRUST_LABELS.get(str(name), str(name)),
            help="verified_local means a person opened the URL and confirmed this record today.",
        )
        source_kinds = catalog_entry.source_kind_options()
        form["source_kind"] = col_kind.selectbox(
            "Source kind",
            options=source_kinds,
            index=source_kinds.index(values["source_kind"]) if values["source_kind"] in source_kinds else 0,
            format_func=lambda name: str(name).replace("_", " ").capitalize(),
        )
        col_verified, col_by = st.columns(2)
        with col_verified:
            verified_on = st.date_input(
                "Verified on",
                value=date.fromisoformat(values["verified_on"]) if values["verified_on"] else None,
                format="YYYY-MM-DD",
            )
            form["verified_on"] = verified_on
        form["verified_by"] = col_by.text_input("Verified by", value=values["verified_by"])
        form["notes"] = st.text_area("Notes", value=values["notes"], height=68)

        submitted = st.form_submit_button("Confirm and add to the catalog", type="primary")

    if not submitted:
        return {}
    # Back to the shapes blank_form() uses, so a rejected submission redraws
    # with what was typed rather than with a date object or a split string.
    for field_name in ("heritage", "employer_restricted", "membership_required", "keywords"):
        form[field_name] = csv_list(form[field_name])
    form["essay_prompts"] = [
        line.strip() for line in str(form["essay_prompts"]).splitlines() if line.strip()
    ]
    for field_name in ("deadline", "verified_on"):
        value = form[field_name]
        form[field_name] = value.isoformat() if isinstance(value, date) else ""
    return form


def _confirm_award_record(record: dict[str, Any], proposal_id: str | None) -> Path:
    """Write a confirmed record, always through the inbox's ``confirm``.

    Hand entry proposes and immediately confirms so that ``records/`` keeps a
    single writer; an edited proposal is confirmed in place so it leaves the
    queue instead of lingering after its award is in the catalog.
    """
    if proposal_id:
        return inbox.confirm(proposal_id, edits=record)
    proposal = inbox.propose(record, kind="manual", notes="Entered by hand on the catalog page.")
    return inbox.confirm(proposal.proposal_id)


def _render_add_award_page() -> None:
    editing = st.session_state.get(CATALOG_EDIT_KEY)
    if editing:
        st.info(f"Editing proposal `{editing}`. Confirming clears it from the inbox.")
    _render_prefill_input()
    values = _catalog_form_values()
    _render_prefill_candidates(values)

    submitted = _award_form_values(values)
    if not submitted:
        return

    # Keep what was typed on screen when validation fails.
    st.session_state[CATALOG_FORM_KEY] = submitted
    record, errors = catalog_entry.validate_form(submitted)
    if errors:
        st.error("This is not a valid catalog record yet:")
        for message in errors:
            st.markdown(f"- {message}")
        return

    try:
        path = _confirm_award_record(record, editing)
    except inbox.ProposalError as exc:
        st.error(f"Could not add the award: {exc}")
        return

    st.session_state[CATALOG_FORM_KEY] = catalog_entry.blank_form()
    st.session_state[CATALOG_EDIT_KEY] = None
    _set_catalog_notice(
        "success",
        f"Added **{record['title']}** to the catalog ({Path(path).name}). "
        "Rebuild the snapshot to rank it.",
    )
    st.rerun()


def _proposal_edit_record(proposal: inbox.Proposal) -> dict[str, Any]:
    """The record to edit: a reverify proposal carries only the fields it changes."""
    existing: dict[str, Any] = {}
    catalog_id = proposal.catalog_id
    if catalog_id:
        path = RECORDS_DIR / f"{catalog_id}.json"
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                existing = loaded
    return {**existing, **proposal.record}


def _render_proposal(proposal: inbox.Proposal) -> None:
    label = proposal.title or proposal.catalog_id or proposal.proposal_id
    with st.expander(f"{label} · {proposal.created_on}", expanded=False):
        if proposal.notes:
            st.caption(proposal.notes)
        rows = catalog_entry.diff_rows(proposal.diff)
        if rows:
            st.dataframe(
                pd.DataFrame(
                    [{"Field": row.field, "In the catalog": row.old, "Proposed": row.new} for row in rows]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("Nothing in the catalog to compare against — this would be a new award.")
            st.json(proposal.record, expanded=False)

        proposal_id = proposal.proposal_id
        col_confirm, col_edit = st.columns(2)
        if col_confirm.button("Confirm", key=f"inbox_confirm_{proposal_id}", use_container_width=True):
            try:
                path = inbox.confirm(proposal_id)
            except inbox.ProposalError as exc:
                st.error(f"{exc}")
                st.caption("Use “Edit and confirm” to fill in what the record is missing.")
            else:
                _set_catalog_notice("success", f"Confirmed {Path(path).name}. Rebuild the snapshot to rank it.")
                st.rerun()
        if col_edit.button("Edit and confirm", key=f"inbox_edit_{proposal_id}", use_container_width=True):
            _load_catalog_form(
                catalog_entry.form_from_record(_proposal_edit_record(proposal)),
                proposal_id=proposal_id,
            )
            st.rerun()

        reason = st.text_input("Reason to reject", key=f"inbox_reason_{proposal_id}")
        if st.button("Reject", key=f"inbox_reject_{proposal_id}"):
            if not reason.strip():
                st.warning("A rejection needs a reason — a later pass reads it.")
            else:
                inbox.reject(proposal_id, reason)
                _set_catalog_notice("success", f"Rejected {proposal_id}.")
                st.rerun()


def _render_inbox_page() -> None:
    try:
        proposals = inbox.list_proposals()
    except Exception as exc:
        st.error(f"Could not read the inbox: {exc}")
        return

    if not proposals:
        st.info("Nothing waiting. Feeds and re-verification runs leave their proposals here.")
        return

    st.caption(
        f"{len(proposals)} waiting. Nothing reaches the catalog until it is confirmed here."
    )
    for kind in inbox.PROPOSAL_KINDS:
        matching = [proposal for proposal in proposals if proposal.kind == kind]
        if not matching:
            continue
        st.markdown(f"**{PROPOSAL_KIND_LABELS.get(kind, kind)}** ({len(matching)})")
        for proposal in matching:
            _render_proposal(proposal)


def _render_rebuild_snapshot() -> None:
    with st.container(border=True):
        st.markdown("**Rebuild snapshot**")
        st.caption(
            "Runs the curated catalog only — no scrapers, no network — so an award "
            "confirmed a moment ago is rankable now."
        )
        if not st.button("Rebuild snapshot"):
            return
        try:
            with st.spinner("Rebuilding from the curated catalog…"):
                report = run_ingest(date=None, only_sources=[CURATED_SOURCE_NAME])
        except Exception as exc:
            st.error(f"Rebuild failed: {exc}")
            return

        snapshot = report["artifact_paths"]["snapshot"]
        if snapshot is None:
            st.error(
                report["artifact_notes"]["snapshot_skip_reason"] or "No snapshot was written."
            )
            if report["artifact_notes"]["snapshot_blocked"]:
                st.caption(
                    "The prior snapshot is untouched. Nothing was lost — but the curated "
                    "records just confirmed are not rankable until this is resolved."
                )
            return
        st.session_state.ingest_report = report
        st.session_state.latest_snapshot_path = snapshot
        st.session_state.latest_delta_summary = report["delta_counts"]
        carried = report["records"]["carried_forward"]
        curated_total = report["records"]["snapshot_total"] - sum(carried.values())
        st.success(
            f"Snapshot rebuilt with {curated_total} curated awards "
            f"and {report['records']['snapshot_total']} records in total."
        )
        if carried:
            st.caption(
                "Carried forward from the prior snapshot: "
                + ", ".join(f"{source} ({count})" for source, count in carried.items())
            )


def render() -> None:
    st.subheader(modes.SECTION_LABELS["catalog_inbox"])
    if state.current_mode() == "student":
        st.info("Adding and reviewing awards lives in Parent view.")
        return

    _render_catalog_notice()
    st.session_state.setdefault(CATALOG_PAGE_KEY, CATALOG_PAGES[0])
    requested = st.session_state.pop(CATALOG_PAGE_REQUEST_KEY, None)
    if requested in CATALOG_PAGES:
        st.session_state[CATALOG_PAGE_KEY] = requested
    page = str(
        st.radio(
            "Catalog page",
            options=CATALOG_PAGES,
            format_func=lambda name: CATALOG_PAGE_LABELS[str(name)],
            horizontal=True,
            label_visibility="collapsed",
            key=CATALOG_PAGE_KEY,
        )
    )
    if page == "inbox":
        _render_inbox_page()
    else:
        _render_add_award_page()
    st.divider()
    _render_rebuild_snapshot()
