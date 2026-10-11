from __future__ import annotations

from typing import Any

import streamlit as st

from app import identity, modes, state
from app.sections import todos as todos_page
from src.store import essays, repo, todos, tracker
from src.store.db import open_db

# At least twelve lines of text on a phone (measured 12.8 at 400px; 320 gave 11.9,
# since padding and border come out of the height).
EDITOR_HEIGHT = 340

SHARE_CAPTION = "Parents can read shared essays. They can't edit them."


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
    """The essay picker under an award's essay prompt.

    Read-only callers pass only the essays they may read, so a linked draft
    she has not shared is named as such, never by title.
    """
    if not can_edit:
        if slot.essay_id is None:
            st.caption("No essay linked yet.")
        elif slot.essay_id in {entry.essay.id for entry in entries}:
            st.caption(slot.essay_title)
        else:
            st.caption("An essay is linked (not shared).")
        return
    if not entries:
        st.caption("No essays in the bank yet — write one under Essays.")
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
        body = st.text_area("Draft", height=EDITOR_HEIGHT)
        if st.form_submit_button("Add essay") and title.strip():
            repo.create_essay(conn, student_id, title.strip(), body, str(theme))
            st.rerun()


def _reuse_caption(count: int) -> str:
    if not count:
        return "not used yet"
    return f"used in {count} application{'s' if count != 1 else ''}"


def _render_share_toggle(conn: Any, essay: repo.Essay) -> None:
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        if essay.shared_with_parents:
            st.badge("Shared", icon=":material/visibility:", color="green")
        else:
            st.badge("Private", icon=":material/lock:", color="gray")
        shared = st.toggle(
            "Share with parents",
            value=essay.shared_with_parents,
            key=f"essay_shared_{essay.id}_{int(essay.shared_with_parents)}",
        )
    st.caption(SHARE_CAPTION)
    if shared != essay.shared_with_parents:
        repo.set_essay_shared(conn, essay.id, shared)
        st.rerun()


def _render_suggest_form(conn: Any, essay: repo.Essay) -> None:
    with st.form(f"essay_suggest_{essay.id}", clear_on_submit=True):
        suggestion = st.text_area(
            "Suggest a change", placeholder="e.g. The ending could say what you'd build next."
        )
        st.caption("She gets it as a to-do that links this essay.")
        submitted = st.form_submit_button("Suggest a change")
    if not submitted or not suggestion.strip():
        return
    # She may have stopped sharing since this page drew.
    if essays.readable_essay(conn, essay.id, shared_only=True) is None:
        st.error("This essay is no longer shared.")
        return
    created_by, created_role = todos_page.author_for(identity.current_identity(), "parent")
    todos.suggest_essay_change(
        conn, essay, suggestion, created_by=created_by, created_role=created_role
    )
    st.toast("Added to her to-dos.")


def _render_essay_detail(conn: Any, entry: essays.BankEntry, can_edit: bool) -> None:
    essay = entry.essay
    header = (
        f"{essay.title} · {entry.theme_label} · {essay.word_count} words "
        f"· {_reuse_caption(entry.reuse_count)}"
    )
    if can_edit and essay.shared_with_parents:
        header += " · Shared"
    with st.expander(header, expanded=False):
        if can_edit:
            _render_share_toggle(conn, essay)
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
                "Draft", value=essay.body, height=EDITOR_HEIGHT, key=f"essay_body_{essay.id}"
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
                    essays.delete_essay(conn, essay.id)
                    st.rerun()
        else:
            st.markdown(essay.body or "_Nothing written yet._")

        earlier = repo.list_essay_versions(conn, essay.id)[1:]
        if earlier and st.checkbox(
            f"Earlier drafts ({len(earlier)})", key=f"essay_versions_{essay.id}"
        ):
            for version in earlier:
                st.caption(f"Saved {version.created_at} · {version.word_count} words")
                if not can_edit:
                    st.markdown(version.body or "_Nothing written yet._")
                    continue
                st.text_area(
                    "Earlier draft",
                    value=version.body,
                    height=140,
                    disabled=True,
                    key=f"essay_version_{version.id}",
                    label_visibility="collapsed",
                )

        if not can_edit:
            _render_suggest_form(conn, essay)


def _render_open_prompts(
    conn: Any, student_id: str, entries: list[essays.BankEntry], can_edit: bool
) -> None:
    slots: list[tuple[str, essays.PromptSlot]] = []
    for application in repo.list_applications(conn, student_id):
        if tracker.normalize_status(application.status) not in tracker.OPEN_STATUSES:
            continue
        title = application.title or application.catalog_id
        slots.extend((title, slot) for slot in essays.prompt_slots(conn, application.id))
    if not slots:
        return

    open_slots = [(title, slot) for title, slot in slots if slot.is_open]
    st.markdown("**Prompts waiting for an essay**")
    if not open_slots:
        st.caption("Every prompt on your open applications has an essay.")
        return
    for award_title, slot in open_slots:
        with st.container(border=True):
            st.markdown(f"**{slot.prompt}**")
            st.caption(award_title)
            render_prompt_slot(conn, slot, entries, can_edit)


def _render_parent_summary(entries: list[essays.BankEntry], unshared: int) -> None:
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        st.badge("Read-only", icon=":material/visibility:", color="gray")
        if unshared:
            st.caption(f"{unshared} draft{'s' if unshared != 1 else ''} not shared")
    if not entries:
        st.info("No shared essays yet. Essays she shares show up here to read.")


def render() -> None:
    modes.render_page_header("essays")
    can_edit = modes.can_edit_essays(state.current_mode())
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            entries = essays.essay_bank(conn, student_id, shared_only=not can_edit)
            if not can_edit:
                _render_parent_summary(entries, essays.unshared_count(conn, student_id))
            elif not entries:
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
