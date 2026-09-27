# Parent Mode Page Overrides

> **PROJECT:** ScholarshipCoach (Study Hall)
> **Applies to:** the Parent-only pages (Catalog & Inbox, Timeline, Colleges & Money, Outcomes,
> Settings) and the Parent read-only view of the student pages.
> **Primary device:** laptop, 1280–1440px; must still work (not shine) at 400px.

> ⚠️ **IMPORTANT:** Rules in this file **override** the Master file
> (`design-system/scholarshipcoach/MASTER.md`). Only deviations from the Master are documented
> here. For all other rules, refer to the Master.

---

## Layout overrides

- `st.set_page_config(layout="wide")`.
- Navigation groups the pages into two `st.navigation` sections: **Student views** (This Week,
  Find, Applications, Essays (read-only), Recommenders, What If) and **Family tools** (Catalog &
  Inbox, Timeline, Colleges & Money, Outcomes, Settings).
- Multi-column forms are fine here; `st.columns` still stacks below 640px.

## Spacing overrides (dense)

- Between cards/sections: `gap="small"`; section breaks use `gap="medium"`, not `st.divider()`.
- Prefer `st.dataframe` with `column_config` (currency, date, link, progress columns) over stacks of
  bordered cards whenever a parent is comparing more than ~8 rows (Timeline list view, Outcomes,
  Colleges).

## Typography overrides

- Tabular figures for money columns (see Master; only if the font subset supports it).
- Operator-adjacent terms are allowed when they are the parent's own vocabulary: "catalog ID",
  "source URL", "verified". Pipeline internals still stay in Operator mode.

## Component overrides

### Catalog & Inbox (Add an award)
- Split the ~40-field form into `st.tabs`: **Basics** (URL, title, sponsor, description) ·
  **Money & dates** · **Who it's for** · **Requirements**, with the Save button outside the tabs
  so it is always visible. Today the page is about 3,000px (desktop) / 5,000px (phone).
- "Read the page" (URL fetch) sits at the top as the primary action; the form is secondary.

### Timeline
- Keep the month-grouped bordered cards; restyle entries with the Master deadline mapping:
  milestones are **blue "Opens"** badges, never "URGENT"; passed items are gray.
- Add a compact list view toggle (`st.segmented_control`: Cards / List) backed by `st.dataframe`
  for scanning a whole year.

### Colleges & Money
- Metrics with `border=True` in one horizontal row. When nothing is recorded, replace the zero row
  with an empty state that points at "Add a college" and My Applications → Record result.

### Outcomes
- Empty state as Master. When populated: `st.dataframe` (award, result, amount, paid to,
  renewable) with a currency column.

### Settings
- Order: Family milestones first (the frequent task), **Advanced** expander last holding "Show
  operator tools" and the mode/PIN notes. Destructive or expert switches never lead the page.

### Parent view of Essays
- Show a gray "Read-only" badge at the top of the page, and render drafts as text, not disabled
  text areas.

## Colors

No overrides.
