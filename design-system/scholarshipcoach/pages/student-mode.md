# Student Mode Page Overrides

> **PROJECT:** ScholarshipCoach (Study Hall)
> **Applies to:** every page reached in Student mode: This Week, Find Scholarships, My
> Applications, Essays, Recommenders, What If, My Profile.
> **Primary device:** phone, ~400px wide, one thumb, often at night.

> ⚠️ **IMPORTANT:** Rules in this file **override** the Master file
> (`design-system/scholarshipcoach/MASTER.md`). Only deviations from the Master are documented
> here. For all other rules, refer to the Master.

The theme itself is global in Streamlit, so these overrides are layout, density, copy and
component choices, not colors.

---

## Layout overrides

- `st.set_page_config(layout="centered")` in Student mode (Master/Parent use `"wide"`). This
  gives a readable ~730px column on desktop and changes nothing on a phone.
- Page opens with: h1 (page name) → one-line caption → the single most important thing. On This
  Week, the first card must be visible without scrolling at 400×860.
- Top navigation (`st.navigation(position="top")`) is limited to **five** student destinations:
  This Week, Find, Applications, Essays, and **More** (a `st.navigation` section holding
  Recommenders, What If, My Profile). Verify at 400px; see Master "Navigation".
  *Not built yet:* Task 1.3 ships one ungrouped list per mode; grouping waits on the Roles &
  Product Plan (`docs/decisions.md`, 2026-09-30).

## Spacing overrides (comfortable)

- Between cards: `gap="medium"`. Inside a card: `gap="small"`.
- Page padding at ≤640px stays as `phone_width_css()` sets it (1.5rem 1rem 4rem).
- Buttons in a card sit in **one horizontal row** (`st.container(horizontal=True)`), not stacked
  full-width. The existing "all buttons 100% wide on phone" rule is scoped to buttons outside
  `.st-key-award_*` / `.st-key-due_*` containers.

## Typography overrides

- None to sizes. Copy rules are stricter than Master:
  - No operator vocabulary (Master anti-pattern list), and also no "catalog ID", "bucket",
    "cycle" or "reason code".
  - Relative time first: "Due Fri · 4 days", then the absolute date.

## Component overrides

### This Week
- Each item is a compact bordered row: title, what's due (step / letter / essay), deadline badge.
  One action per row (open the application, or mark the step done).
- Empty state: "Nothing due this week." plus **Next up** (the nearest future deadline, as an
  award card) plus `st.page_link` to Find. Use a neutral container, not `st.success`.

### Find Scholarships
- **Results render on arrival.** Run the ranking with the saved profile on first load (cached).
  The filter controls go into a collapsed `st.expander("Filters", icon=":material/tune:")` or
  `st.popover`: search, "When can I apply" (the Timeline select, renamed), minimum amount,
  no-essay only.
- Remove from Student mode: the "Active snapshot" info box, "Pipeline Execution" heading, "Active
  ranking weights", the Top-N slider (fixed at 25, with a "Show more" button), the "Pipeline
  complete: eligible= ineligible=" box, and the "Signal details" expander. They stay in Operator
  mode.
- Card: the Master award card at ≤260px tall. Header reads "21 matches for you".

### My Applications
- Expander header = title only; status and progress go *inside* the expander's first line as a
  status badge plus a "3 steps left" badge. (Expander labels can't hold badges.) Alternatively use
  a bordered container per application with a `st.progress` bar and a "Open" toggle.
- Empty state links to Find.

### Essays
- Replace "Challenge: 1" with theme chips (`st.pills`, read-only look) or drop it.
- "Prompts waiting for an essay" becomes a card list headed "Needs an essay", each card linking
  the prompt to an essay via one select plus a button.
- Editor text area: ≥ 12 rows on a phone, word count under it as a caption.

### Recommenders
- One row per person: name, role, and a badge "2 letters in / 1 waiting". The add form stays an
  expander, open only when the list is empty (current behaviour).

### What If
- Result first: when something changes, the "N more awards open up" line and the list render above
  the controls on a phone, or directly under the control that changed.

## Colors

No overrides. Student mode is where the calm deadline mapping matters most: the orange badge is the
loudest thing a student normally sees; red appears only for overdue.
