from __future__ import annotations

import streamlit as st

from app import modes, state
from src.profile.grade_levels import GRADE_LABELS
from src.profile.store import load_profile, save_profile


def render() -> None:
    modes.render_page_header("profile")

    if st.session_state.get("profile_is_demo", False):
        st.info(
            "Demo profile — fictional data. Edit the fields and press Save Profile to "
            "store your own under data/private/."
        )

    st.subheader("Academic", divider=False)
    st.text_input("Full Name", key="profile_name", placeholder="Your name (optional)")
    st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.01, key="profile_gpa",
                   help="Your current GPA (0.0-4.0)")
    st.text_input("State", key="profile_state", placeholder="e.g., NC")
    st.text_input("County", key="profile_county",
                 placeholder="e.g., Guilford",
                 help="Many local awards are county-restricted")
    st.text_input("High School", key="profile_high_school",
                 placeholder="e.g., Northside High School")
    st.text_input("Intended Major", key="profile_major", placeholder="e.g., Computer Science")
    st.selectbox("Grade Level",
                options=list(GRADE_LABELS),
                key="profile_grade_label",
                help="Your current or upcoming grade level")
    st.number_input("Graduation Year", min_value=0, max_value=2100, step=1,
                   key="profile_graduation_year",
                   help="Leave at 0 to estimate it from your grade level")
    st.number_input("SAT", min_value=0, max_value=1600, step=10, key="profile_sat",
                   help="Leave at 0 if you have not taken it")
    st.number_input("ACT", min_value=0, max_value=36, step=1, key="profile_act",
                   help="Leave at 0 if you have not taken it")

    st.subheader("About you", divider=False)
    st.caption("Every field here is optional and only used to match restricted awards.")
    st.text_input("Citizenship", key="profile_citizenship", placeholder="e.g., U.S. Citizen")
    st.selectbox("Financial need", options=list(state.TRISTATE_OPTIONS),
                key="profile_financial_need",
                help="Whether you qualify for need-based aid")
    st.selectbox("First-generation college student", options=list(state.TRISTATE_OPTIONS),
                key="profile_first_gen")
    st.selectbox("Gender", options=list(state.GENDER_OPTIONS), key="profile_gender")
    st.text_input("Heritage / background", key="profile_heritage_csv",
                 placeholder="e.g., Hispanic, Cherokee (comma-separated)")
    st.selectbox("Military family", options=list(state.TRISTATE_OPTIONS),
                key="profile_military_family")
    st.selectbox("Disability", options=list(state.TRISTATE_OPTIONS), key="profile_disability")
    st.text_input("Religion", key="profile_religion", placeholder="Optional")
    st.text_input("Parent employers", key="profile_parent_employers_csv",
                 placeholder="e.g., Duke Energy (comma-separated)",
                 help="Some awards are restricted to employees' children")

    st.subheader("Activities", divider=False)
    st.text_input(
        "Interests / Keywords",
        key="profile_keywords_csv",
        placeholder="e.g., robotics, leadership, community service (comma-separated)",
        help="Topics you're passionate about—we'll match scholarships to these"
    )
    st.text_input("Extracurriculars", key="profile_extracurriculars_csv",
                 placeholder="e.g., robotics team, marching band (comma-separated)")
    st.text_input("Memberships", key="profile_memberships_csv",
                 placeholder="e.g., 4-H, National Honor Society (comma-separated)")
    st.number_input("Community service hours", min_value=0, max_value=10000, step=5,
                   key="profile_service_hours")
    st.checkbox("I have an essay draft ready", key="profile_essay_ready")
    st.text_area("Your Goals", key="profile_goals", height=120,
                placeholder="Tell us about your goals, dreams, or what matters to you",
                help="We use this to find scholarships that align with your aspirations")

    st.subheader("Colleges", divider=False)
    st.text_input("Intended colleges", key="profile_intended_colleges_csv",
                 placeholder="e.g., NC State, UNC Charlotte (comma-separated)")

    st.subheader("Advanced", divider=False)
    st.checkbox("Use custom date", key="profile_use_today_override", help="Override today's date for testing")
    st.date_input("Custom date", key="profile_today_override")

    save_col, load_col = st.columns(2)
    if save_col.button("Save Profile", use_container_width=True):
        profile = state.profile_from_widgets()
        saved_path = save_profile(profile)
        st.session_state.profile = profile
        st.session_state.profile_is_demo = False
        st.success(f"Saved to {saved_path}")
    if load_col.button("Load Profile", use_container_width=True):
        loaded = load_profile()
        if loaded is None:
            st.warning(f"No profile file found at {state.PROFILE_PATH}")
        else:
            st.session_state.profile = loaded
            st.session_state.profile_is_demo = False
            state.apply_profile_to_widgets(loaded)
            st.success("Profile loaded.")
            st.rerun()
