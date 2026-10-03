# Parent Mode Page Overrides

> **PROJECT:** ScholarshipCoach (Study Hall)
> **Applies to:** every page reached in Parent mode: the money and strategy view (Find
> Scholarships, Applications, Colleges & Money, Timeline, Outcomes), family tools (Catalog & Inbox,
> Settings, Profile), and the Parent view of her path (My Path, To-dos, Opportunities, shared
> Essays, What If, Recommenders). Page list from
> `docs/plans/SCHOLARSHIPCOACH_ROLES_PRODUCT_PLAN.md` (2026-10-03).
> **Primary device:** laptop, 1280–1440px; must still work (not shine) at 400px.

> ⚠️ **IMPORTANT:** Rules in this file **override** the Master file
> (`design-system/scholarshipcoach/MASTER.md`). Only deviations from the Master are documented
> here. For all other rules, refer to the Master.

---

## Layout overrides

- `st.set_page_config(layout="wide")`.
- Navigation groups the pages into three `st.navigation` sections: **Her path** (My Path,
  To-dos, Opportunities, Essays, What If, Recommenders), **Money** (Find Scholarships,
  Applications, Colleges & Money, Timeline, Outcomes) and **Family tools** (Catalog & Inbox,
  Settings, Profile). Decided 2026-10-03; built in Roles & Product Plan Task A.3.
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

### Award card: money is the headline
- Find Scholarships and anywhere a parent lists awards use the Master award card with the amount
  as the h3 (this was a Master rule until 2026-10-03; it now applies to Parent mode only).
- A third card action, "Suggest to her", creates a to-do for her; the card then shows a violet
  "Suggested" badge.

### Find Scholarships
- **Results render on arrival.** Rank with the saved profile on first load (cached). Filters
  (search, "When can she apply", minimum amount, no-essay only) live in a collapsed
  `st.expander("Filters", icon=":material/tune:")`.
- Pipeline internals (snapshot box, weights, Top-N, `eligible=`/`ineligible=`, signal details)
  stay in Operator mode. 25 results and a "Show more" button.
- An award whose grade scope is unknown and her grade is below 12 shows the yellow "Confirm you
  meet: grade" badge and is never labelled "Apply now".

### Applications
- Expander label = title only; first line inside: status badge, deadline badge, "3 steps left",
  `st.progress`.

### Her path (read-only views)
- My Path, Opportunities and What If render as in `student-mode.md` with controls she owns
  disabled or hidden, and a gray "Her view" badge under the h1.
- To-dos are shared: parents add, tick and see who added each one.

### Catalog & Inbox (Add an award)
- Split the ~40-field form into `st.tabs`: **Basics** (URL, title, sponsor, description) ·
  **Money & dates** · **Who it's for** · **Requirements**, with the Save button outside the tabs
  so it is always visible. Today the page is about 3,000px (desktop) / 5,000px (phone).
- "Read the page" (URL fetch) sits at the top as the primary action; the form is secondary.

### Timeline
- A Parent page (Money group) showing every date: award deadlines, family milestones, her dated
  to-dos and requirement dates. She sees her own dates on My Path and To-dos, not here.
- Keep the month-grouped bordered cards; restyle entries with the Master deadline mapping:
  milestones are **blue "Opens"** badges, never "URGENT"; passed items are gray.
- Add a compact list view toggle (`st.segmented_control`: Cards / List) backed by `st.dataframe`
  for scanning a whole year.

### Colleges & Money
- Each college has a **Requirements** section (category, label, target, due by, source, verified
  on); one college can be marked her first target. Requirement rows are what My Path shows her.
- Metrics with `border=True` in one horizontal row. When nothing is recorded, replace the zero row
  with an empty state that points at "Add a college" and My Applications → Record result.

### Outcomes
- Empty state as Master. When populated: `st.dataframe` (award, result, amount, paid to,
  renewable) with a currency column.

### Settings
- Order: Family milestones first (the frequent task), **Advanced** expander last holding "Show
  operator tools" and the mode/PIN notes. Destructive or expert switches never lead the page.

### Parent view of Essays
- Only essays she has shared are listed (decided 2026-10-03). Show a gray "Read-only" badge at the
  top of the page and render shared drafts as text, not disabled text areas. Unshared drafts appear
  only as a count ("2 drafts not shared"), never by title.
- "Suggest a change" creates a to-do for her that links the essay.

## Colors

No overrides.
