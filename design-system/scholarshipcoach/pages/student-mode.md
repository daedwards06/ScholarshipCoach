# Student Mode Page Overrides

> **PROJECT:** ScholarshipCoach (Study Hall)
> **Applies to:** every page reached in Student mode: My Path, To-dos, Opportunities, Essays,
> What If, Recommenders, My Profile (page list from `docs/plans/SCHOLARSHIPCOACH_ROLES_PRODUCT_PLAN.md`,
> 2026-10-03). Student mode is her path to college, not a scholarship tracker; Find Scholarships
> and Applications are Parent pages until the senior-year switch.
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
- Page opens with: h1 (page name) → one-line caption → the single most important thing. On My
  Path, the progress line and the first "This semester" item must be visible without scrolling at
  400×860.
- Top navigation (`st.navigation(position="top")`) holds **four** destinations: My Path, To-dos,
  Opportunities, Essays. **More** (a `st.navigation` section) holds What If, Recommenders, My
  Profile. At 400px Streamlit folds the top bar behind `»` (`docs/decisions.md`, 2026-09-30), so
  My Path also links every top page with `st.page_link`, which makes each one tap from home.

## Spacing overrides (comfortable)

- Between cards: `gap="medium"`. Inside a card: `gap="small"`.
- Page padding at ≤640px stays as `phone_width_css()` sets it (1.5rem 1rem 4rem).
- Buttons in a card sit in **one horizontal row** (`st.container(horizontal=True)`), not stacked
  full-width. The existing "all buttons 100% wide on phone" rule is scoped to buttons outside
  `.st-key-award_*` / `.st-key-opp_*` / `.st-key-todo_*` containers.

## Typography overrides

- None to sizes. Copy rules are stricter than Master:
  - No operator vocabulary (Master anti-pattern list), and also no "catalog ID", "bucket",
    "cycle", "reason code" or "kind".
  - Relative time first: "Due Fri · 4 days", then the absolute date.
  - Speak to her, not about her: "your first-choice school", "you're 0.25 away", never "the
    student".
  - Never name a parent's worry for her ("you could lose $…"). Money appears as a fact on a card,
    not as pressure.

## Component overrides

### Opportunity card (Opportunities, My Path "Coming up", anywhere a program or award is listed)

The student card headline is **the next step and its date**, not the money.

```
┌ st.container(border=True, key=f"opp_{id}") ─────────────────┐
│ **Title** (bold body)                       [badge: kind]   │
│ Sponsor (caption)                                           │
│ ### Apply by Mar 15   (h3: the next step and its date)      │
│ [badge: Open to 10th grade] [badge: Free / $2,000] [local]  │
│ [badge: From your family, violet] (only when suggested)     │
│ Why it fits: one line, max three reasons joined by " · "    │
│ [ Add to my to-dos ] [ Apply ↗ ]   one row, Add = primary   │
│ ▸ More about this (expander)                                │
└─────────────────────────────────────────────────────────────┘
```

- Headline patterns: "Apply by Mar 15", "Opens Jan 10 · apply in 11th grade", "Date not posted
  yet". An amount or cost is a badge ("Free", "$2,000", "Amount not listed"), never the h3.
- Something her grade cannot apply to yet never says "Apply now"; its headline says when it opens.
- Once added, the card shows a green "On my to-dos" badge and Apply becomes primary.
- Target ≤ 260px tall at 400px, as Master.

### My Path (home)
- Header "Your path to <school>" (the name comes from the family database, never from code).
  Under it a single progress line ("5 of 12 on track") and `st.progress`.
- Requirement rows grouped **This semester**, **Next year**, **Senior year**, **Done**. A row is a
  compact bordered container: label, target ("4 units", "GPA 3.0+"), a status badge (gray Not
  started · blue In progress · green Met), and one control to change status where she owns it.
- GPA and test rows show the gap in words ("You're at 3.25 · 0.25 to go") and link to What If.
- "Coming up": the next three dated items (to-dos, requirement dates, programs) as Opportunity
  or to-do rows, then `st.page_link`s to To-dos and Opportunities.
- Empty state (no school yet): "Your family hasn't added a school yet." with a link to
  Opportunities.

### To-dos
- Groups **Overdue**, **This week**, **Coming up**, **Someday**. Each row is a compact bordered
  container keyed `todo_<id>`: checkbox + title, who it's for and who added it ("from Mom") as a
  caption, deadline badge on the right. One action per row: the checkbox.
- Rows that come from an application, a letter or a requirement link back to it.
- Add form in an expander at the top, open only when the list is empty: title, date (optional),
  "For: me / parents / family".
- Empty state: "Nothing to do this week." plus **Next up** (the nearest dated item) plus
  `st.page_link` to Opportunities. Neutral container, not `st.success`.

### Opportunities
- Groups **Open now**, **Opens later** (with when), **Passed this year** (collapsed). Opportunity
  cards as above. Filters (kind, free only, local only) in a collapsed
  `st.expander("Filters", icon=":material/tune:")`.
- Only records with an explicit grade match appear under Open now; records with no grade
  information are not shown to her.

### Essays
- Each essay shows a "Share with parents" toggle with the caption "Parents can read shared
  essays. They can't edit them." and a gray "Private" or green "Shared" badge.
- Replace "Challenge: 1" with theme chips (`st.pills`, read-only look) or drop it.
- "Needs an essay" cards appear only when a saved application has a prompt; otherwise nothing.
- Editor text area: ≥ 12 rows on a phone, word count under it as a caption.

### Recommenders
- One row per person: name, role, and a badge "2 letters in / 1 waiting". The add form stays an
  expander, open only when the list is empty (current behaviour).

### What If
- Result first: "Your goals" (each GPA/test requirement of her first target school, reached or
  how far) renders above the controls on a phone, or directly under the control that changed;
  awards that open up follow, with "Amount not listed" instead of "$0".

## Colors

No overrides. Student mode is where the calm deadline mapping matters most: the orange badge is the
loudest thing a student normally sees; red appears only for overdue.
