from __future__ import annotations

from typing import Any

import streamlit as st

from app import modes, state
from src.store import essays, repo, tracker
from src.store.db import open_db


def _essay_choice_labels(entries: list[essays.BankEntry]) -> dict[int, str]:
    return {
        entry.essay.id: (
            f"{entry.essay.title} · {entry.theme_label} · {entry.essay.word_count} words"
        )
        for entry in entries
    }


def render_prompt_slot(
    conn: Any, slot: essays.PromptSlot, entries: list[essays.BankEntry], can_edit: bool
) -> None:
    """The essay picker under an award's essay prompt."""
    if not entries:
        st.caption("No essays in the bank yet — write one under Essays.")
        return
    if not can_edit:
        st.caption(slot.essay_title or "No essay linked yet.")
        return

    labels = _essay_choice_labels(entries)
    options = [0, *labels]
    chosen = st.selectbox(
        "Essay for this prompt",
        options=options,
        index=options.index(slot.essay_id) if slot.essay_id in labels else 0,
        format_func=lambda value: (
            "— no essay linked —" if value == 0 else labels[int(value)]
        ),
        key=f"slot_essay_{slot.item_id}",
        label_visibility="collapsed",
    )
    if int(chosen) == (slot.essay_id or 0):
        return
    if int(chosen) == 0:
        essays.clear_prompt(conn, slot)
    else:
        essays.use_essay_for_prompt(conn, int(chosen), slot)
    st.rerun()


def _render_new_essay_form(conn: Any, student_id: str) -> None:
    with st.form("new_essay", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. The summer I rebuilt the robot")
        theme = st.selectbox(
            "Theme",
            options=repo.ESSAY_THEMES,
            format_func=lambda name: repo.ESSAY_THEME_LABELS[str(name)],
        )
        body = st.text_area("Draft", height=160)
        if st.form_submit_button("Add essay") and title.strip():
            repo.create_essay(conn, student_id, title.strip(), body, str(theme))
            st.rerun()


def _reuse_caption(count: int) -> str:
    if not count:
        return "not used yet"
    return f"used in {count} application{'s' if count != 1 else ''}"


def _render_essay_detail(conn: Any, entry: essays.BankEntry, can_edit: bool) -> None:
    essay = entry.essay
    header = (
        f"{essay.title} · {entry.theme_label} · {essay.word_count} words "
        f"· {_reuse_caption(entry.reuse_count)}"
    )
    with st.expander(header, expanded=False):
        if entry.used_by:
            st.caption("Used in: " + ", ".join(entry.used_by))

        if can_edit:
            title = st.text_input("Title", value=essay.title, key=f"essay_title_{essay.id}")
            theme = st.selectbox(
                "Theme",
                options=repo.ESSAY_THEMES,
                index=repo.ESSAY_THEMES.index(repo.normalize_theme(essay.theme)),
                format_func=lambda name: repo.ESSAY_THEME_LABELS[str(name)],
                key=f"essay_theme_{essay.id}",
            )
            body = st.text_area(
                "Draft", value=essay.body, height=240, key=f"essay_body_{essay.id}"
            )
            st.caption(f"{repo.word_count(body)} words")
            col_save, col_delete = st.columns([0.7, 0.3])
            with col_save:
                if st.button("Save draft", key=f"essay_save_{essay.id}"):
                    repo.update_essay(
                        conn,
                        essay.id,
                        title=title.strip() or essay.title,
                        body=body,
                        theme=str(theme),
                    )
                    st.rerun()
            with col_delete:
                if st.button("Delete essay", key=f"essay_delete_{essay.id}"):
                    repo.delete_essay(conn, essay.id)
                    st.rerun()
        else:
            st.markdown(essay.body or "_Nothing written yet._")

        earlier = repo.list_essay_versions(conn, essay.id)[1:]
        if earlier and st.checkbox(
            f"Earlier drafts ({len(earlier)})", key=f"essay_versions_{essay.id}"
        ):
            for version in earlier:
                st.caption(f"Saved {version.created_at} · {version.word_count} words")
                st.text_area(
                    "Earlier draft",
                    value=version.body,
                    height=140,
                    disabled=True,
                    key=f"essay_version_{version.id}",
                    label_visibility="collapsed",
                )


def _render_open_prompts(
    conn: Any, student_id: str, entries: list[essays.BankEntry], can_edit: bool
) -> None:
    open_slots: list[tuple[str, essays.PromptSlot]] = []
    for application in repo.list_applications(conn, student_id):
        if tracker.normalize_status(application.status) not in tracker.OPEN_STATUSES:
            continue
        title = application.title or application.catalog_id
        open_slots.extend(
            (title, slot) for slot in essays.open_prompt_slots(conn, application.id)
        )

    st.markdown("**Prompts waiting for an essay**")
    if not open_slots:
        st.caption("Every prompt on your open applications has an essay.")
        return
    for award_title, slot in open_slots:
        with st.container(border=True):
            st.markdown(f"**{slot.prompt}**")
            st.caption(award_title)
            render_prompt_slot(conn, slot, entries, can_edit)


def render() -> None:
    st.subheader(modes.SECTION_LABELS["essays"])
    can_edit = modes.can_edit_essays(state.current_mode())
    if not can_edit:
        st.caption("Parent view reads the essay bank. Switch to Student to edit.")
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            entries = essays.essay_bank(conn, student_id)
            if entries:
                counts = essays.theme_counts(entries)
                st.caption(
                    " · ".join(
                        f"{repo.ESSAY_THEME_LABELS[theme]}: {count}"
                        for theme, count in counts.items()
                        if count
                    )
                )
            else:
                st.info(
                    "No essays yet. A few themed drafts answer most prompts — start "
                    "with a challenge story and one on why your major."
                )

            if can_edit:
                with st.expander("Add an essay", expanded=not entries):
                    _render_new_essay_form(conn, student_id)

            for entry in entries:
                _render_essay_detail(conn, entry, can_edit)

            _render_open_prompts(conn, student_id, entries, can_edit)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
