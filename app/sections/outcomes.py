from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from app import modes, state
from app.helpers import award_count_text, money_text
from scripts.export_outcomes import build_outcomes_frame
from src.store import money, repo, tracker
from src.store.db import open_db


def _outcomes_table(rows: list[money.OutcomeRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Award": row.award_title,
                "Cycle": "—" if row.cycle_year is None else str(row.cycle_year),
                "Result": tracker.STATUS_LABELS.get(row.result, row.result.title()),
                "Amount": money_text(row.amount),
                "Renewal terms": row.renewal_terms,
                "Paid to": row.paid_to,
                "Decided": row.decided_on,
            }
            for row in rows
        ]
    )


def _outcomes_csv_callable(
    student_id: str, profile: dict[str, Any], snapshot_path: str | None, today_value: date
) -> Any:
    # The download callable runs on its own thread after the click, so it
    # opens its own connection and reads nothing from session state.
    def build() -> str:
        snapshot_df = pd.read_parquet(snapshot_path) if snapshot_path else pd.DataFrame()
        with open_db() as conn:
            frame = build_outcomes_frame(
                conn, student_id, profile, snapshot_df, today=today_value
            )
        return str(frame.to_csv(index=False))

    return build


def render() -> None:
    st.subheader(modes.SECTION_LABELS["outcomes"])
    profile = st.session_state.profile
    today_value = state.effective_today(profile)
    try:
        with open_db() as conn:
            student_id = state.ensure_student(conn)
            applications = repo.list_applications(conn, student_id)
            outcomes = repo.list_outcomes(conn, student_id)
    except Exception as exc:
        st.error(f"Could not open the family database: {exc}")
        return

    if not outcomes:
        st.info(
            "No results recorded yet. When an award decides, open it in My Applications, "
            "set its status to Submitted, and record the result there."
        )
        return

    col_won, col_decided = st.columns(2)
    col_won.metric("Total won", money_text(money.total_won(outcomes)))
    col_decided.metric("Results recorded", str(len(outcomes)))

    by_year = money.won_by_year(outcomes)
    if by_year:
        st.markdown("**Won by school year**")
        for year in by_year:
            st.markdown(f"{year.label}: {money_text(year.total)} · {award_count_text(year.count)}")

    st.dataframe(
        _outcomes_table(money.outcome_rows(applications, outcomes)),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download outcomes.csv",
        data=_outcomes_csv_callable(
            student_id, dict(profile), state.active_snapshot_path(), today_value
        ),
        file_name="outcomes.csv",
        mime="text/csv",
        on_click="ignore",
        help="Submitted applications with their results and ranking features, for the win model.",
    )
