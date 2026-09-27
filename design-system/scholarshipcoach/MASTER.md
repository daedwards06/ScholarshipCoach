# Design System Master File — ScholarshipCoach ("Study Hall")

> **LOGIC:** When building a specific page, first check `design-system/scholarshipcoach/pages/[page-name].md`.
> If that file exists, its rules **override** this Master file.
> If not, strictly follow the rules below.

---

**Project:** ScholarshipCoach
**Direction:** A — Study Hall (chosen 2026-09-26 from three proposals; see
`docs/plans/SCHOLARSHIPCOACH_DESIGN_REFRESH_PLAN.md`)
**Stack:** Streamlit 1.54. Every value below is expressed as the `.streamlit/config.toml` key or
native element that carries it. CSS is the last resort (see "Where styling lives").
**Provenance:** Scaffolded with `ui-ux-pro-max --design-system --persist`. The generator's raw
palette (indigo/orange), fonts (Baloo 2 + Comic Neue), landing-page pattern and GSAP motion did not
fit this product or stack and were replaced with the values chosen and contrast-checked below.
**Page overrides:** `pages/student-mode.md`, `pages/parent-mode.md`.

---

## Product and tone

A private scholarship finder and application tracker for one family. The primary user is a
high-school student on her phone; parents use a denser Parent mode on a laptop.

- **Calm, encouraging, never alarmed.** Deadlines are facts with a date and a day count, not
  sirens. Reserve red for *overdue* only.
- **Say what to do next.** Every empty state and every card ends in one obvious action.
- **Plain words for the student.** No pipeline, model, snapshot, weight or Top-N vocabulary on a
  student surface. That language belongs in Operator mode.
- **Money is the headline.** On an award card, the amount is the most prominent fact after the
  title.

---

## Color tokens

Light and dark are defined together and follow the device's system setting. Every text pair below
was measured: body text ≥ 4.5:1, UI boundaries and large text ≥ 3:1.

### Core

| Role | Light | Dark | config.toml key | Measured |
|------|-------|------|-----------------|----------|
| Background | `#F6FAF9` | `#0F1917` | `backgroundColor` | — |
| Surface (inputs, code, secondary) | `#E7F0EE` | `#182624` | `secondaryBackgroundColor` | — |
| Sidebar background | `#EEF5F3` | `#131F1D` | `sidebar.backgroundColor` | — |
| Text | `#12302C` | `#E4F1EE` | `textColor` | 13.4:1 / 15.5:1 on bg |
| Muted text (captions) | `#4B625E` | `#9DB5B0` | `grayTextColor` (+ caption opacity, below) | 6.2:1 / 8.3:1 on bg |
| Primary (buttons, active radio, slider) | `#0F766E` | `#11807A` | `primaryColor` | white label 5.5:1 / 4.8:1; vs bg 5.2:1 / 3.8:1 |
| Link | `#0F766E` | `#5FE0CD` | `linkColor` | 5.2:1 / 11.1:1 on bg |
| Border (widgets **and** containers) | `#739690` | `#4F6964` | `borderColor` + `showWidgetBorder = true` | 3.1:1 / 3.0:1 on bg |

`st.caption` does not use `grayTextColor`: Streamlit 1.54 paints it as body text at 60% opacity,
which is only 3.9:1 on the light background. `phone_width_css()` raises caption opacity to 0.75
(`CAPTION_OPACITY`), which gives 6.2:1 light and 9.1:1 dark and is tested per theme and surface.

The dark primary is deliberately a mid-tone. Streamlit paints primary-button labels white, and the
current bright `#4ECDC4` (rendered `#2D9F98`) gives white text only 3.2:1. A bright teal is used
for dark-mode *links* instead, via `linkColor`.

### Status (drives `st.badge`, `st.success/info/warning/error`)

| Meaning | Streamlit color | Light text / bg | Dark text / bg | Measured (light / dark) |
|---------|-----------------|-----------------|----------------|-------------------------|
| Overdue, error | `red` | `#B42318` / `#FEE4E2` | `#F97066` / `#3B1716` | 5.5 / 5.7 |
| Due within 7 days | `orange` | `#B54708` / `#FEF0C7` | `#F5A524` / `#3A2A0B` | 4.8 / 6.8 |
| Heads-up, confirm eligibility | `yellow` | `#93370D` / `#FEF7C3` | `#FDE272` / `#36300B` | 6.9 / 10.3 |
| Won, done, saved | `green` | `#067647` / `#DCFAE6` | `#47CD89` / `#0E3222` | 5.1 / 6.9 |
| Opens, milestone, info | `blue` | `#1D4ED8` / `#DBEAFE` | `#7CB4FF` / `#132A4A` | 5.5 / 6.7 |
| Local award, special | `violet` | `#5B21B6` / `#EDE9FE` | `#BDB4FE` / `#2A2150` | 7.6 / 7.8 |
| Later, passed, neutral | `gray` | `#475467` / `#EAECF0` | `#B0BEBA` / `#24302E` | 6.5 / 7.1 |

Set each as `<name>Color`, `<name>BackgroundColor` and `<name>TextColor` under `[theme.light]` and
`[theme.dark]`. Gray is the exception: Streamlit 1.54 has no gray badge text key, so gray badges
use `grayTextColor`, the muted-text token (`#4B625E` / `#9DB5B0`, 5.5 / 6.3 on the gray bg). `primary` badges use the primary color, so `#0F766E` on `#CCFBEF` (4.9:1) light and
`#5FE0CD` on `#113A36` (7.8:1) dark.

Color never carries meaning alone. Every badge has a text label and a Material icon.

### Deadline states (one mapping, used everywhere)

| State | Badge | Icon | Label pattern |
|-------|-------|------|---------------|
| Overdue | red | `:material/error:` | "Overdue · was Oct 1" |
| ≤ 7 days | orange | `:material/schedule:` | "Due Fri, Oct 10 · 4 days" |
| 8–30 days | blue | `:material/event:` | "Due Oct 30 · 34 days" |
| > 30 days | gray | `:material/event:` | "Due Nov 5" |
| Projected | gray | `:material/update:` | "Usually ~Apr 1" |
| Needs a date | yellow | `:material/help:` | "Date not posted yet" |
| Milestone opens | blue | `:material/flag:` | "FAFSA opens Oct 1" (never "URGENT") |
| Passed | gray | `:material/history:` | "Passed" |

This replaces `urgency_indicator()`'s emoji and hex colors in `app/helpers.py`.

---

## Typography

| Role | Family | Weights | config.toml |
|------|--------|---------|-------------|
| Headings | **Nunito** | 700, 800 | `headingFont = "Nunito, sans-serif"` |
| Body / UI | **Nunito Sans** | 400, 600, 700 | `font = "Nunito Sans, sans-serif"` |
| Code (Operator only) | Streamlit default mono | — | unchanged |

- **Self-host, don't hot-link.** Put the `.woff2` files under `app/static/fonts/` and declare them
  with `[[theme.fontFaces]]` plus `server.enableStaticServing = true`. The Google Fonts URL form
  would send a request from every family phone to Google, which contradicts
  `gatherUsageStats = false`. Both families are SIL OFL; commit the `OFL.txt`.
- `baseFontSize = 16`. Never style body text below 14px, and inputs never below 16px (keeps
  the existing iOS no-zoom rule).
- `headingFontSizes = ["1.75rem", "1.375rem", "1.125rem", "1rem"]` (h1–h4). The default 2.75rem
  h1 wraps and takes about 200px on a phone.
- `headingFontWeights = [800, 700, 700, 700]`
- Line height is Streamlit's default (~1.6). Keep prose to about 65 characters per line on desktop.
  Student mode's centered layout does this for free.
- Money: prefer tabular figures (`font-variant-numeric: tabular-nums`) in Parent tables if Nunito
  Sans exposes `tnum` in the shipped subset; skip it if not.

---

## Shape, spacing, elevation

- `baseRadius = "0.75rem"`, `buttonRadius = "0.75rem"`.
- **Elevation:** none. Separation comes from `st.container(border=True)` and surface color. No
  shadows, glows or gradients.
- **Spacing rhythm:** 4/8. Use `st.container(gap=...)` and `st.space` rather than blank
  `st.markdown("")` or `st.divider()` spacers.
  - `gap="small"` (≈8px): inside a card
  - `gap="medium"` (≈16px): between cards
  - `gap="large"` (≈24px): between page sections
- **Touch:** every button, link button, expander header and nav item is ≥ 44px tall on a phone
  (already enforced by `phone_width_css()`; keep it).

---

## Iconography

- **Material Symbols Rounded**, which Streamlit ships, used via `:material/<name>:` in `icon=`
  parameters and labels. Rounded matches Nunito.
- **No emoji as icons or status.** This removes 🔴🟡🟢⏰⚠️📊 from helpers and sections. Emoji in
  user-authored text (essay titles, notes) is the user's business.
- One icon per nav item, per badge and per primary button at most. Decorative icons sit beside
  visible text; never use an icon-only control.

Suggested nav icons: This Week `today`, Find `search`, Applications `assignment`, Essays
`edit_note`, Recommenders `group`, What If `tune`, Profile `person`, Catalog & Inbox `inbox`,
Timeline `calendar_month`, Colleges & Money `payments`, Outcomes `emoji_events`, Settings
`settings`.

---

## Navigation and page chrome

- `st.navigation(pages, position="top")` with `st.Page(..., title=, icon=, url_path=)` replaces the
  two sidebar radios. That gives each section its own URL, so a phone bookmark opens Essays directly,
  and removes the hamburger step. Verify the top bar's overflow behaviour at 400px before
  committing; fall back to `position="sidebar"` with grouped sections if it wraps badly.
- The mode switch (Student / Parent, PIN) stays in the sidebar or on Settings. It is a rare action.
- **Profile becomes its own page** ("My Profile"). The sidebar stops being a 30-field form under the
  navigation.
- **No global title block.** Drop the per-page `st.title("Scholarship Coach")` and the
  "Find scholarships matched to your profile" caption. `st.set_page_config(page_title=...)` names
  the tab. Each page opens with its own h1 and a one-line caption that says what the page is for.

---

## Components (native Streamlit)

### Award card (Find, and anywhere an award is listed)

```
┌ st.container(border=True, key=f"award_{id}") ───────────────┐
│ **Title** (bold body)                      [badge: deadline]│
│ Sponsor (caption)                                           │
│ ### $2,000 – $4,000   (h3, the money is the headline)       │
│ [badge: 1 essay] [badge: 1 letter] [badge: local, violet]   │
│ Why it fits: one line, max three reasons joined by " · "    │
│ ⚠ Confirm you meet: gender (yellow badge, only when set)    │
│ [ Save ] [ Apply ↗ ]   one horizontal row, Save = primary   │
│ ▸ More about this award (expander: details, signal detail)  │
└─────────────────────────────────────────────────────────────┘
```

- Target ≤ 260px tall at 400px width (today it is about 600px).
- Reasons are one joined line, not one paragraph per bullet.
- One primary action per card: **Save** until saved, then the card shows a green "Saved" badge and
  **Apply** becomes primary.
- The urgency reason must not contradict the deadline badge. Only say "deadline soon" when the
  badge is orange or red.

### Status badge
`st.badge(label, icon=":material/…:", color=…)` using the deadline and status mappings above.
Never `st.markdown` HTML with inline hex.

### Metrics
`st.metric` with `border=True` inside a horizontal container. When the value is zero because nothing
has been recorded yet, show an empty state instead of "$0 / 0 / 0".

### Empty state
One `st.container(border=True)` holding: a sentence about what *would* be here, a next-best fact
("Next up: Regeneron, due Nov 5") and one button or `st.page_link` to the page that fills it.
Replaces bare `st.success` / `st.info` boxes.

### Alerts
`st.info` for neutral guidance, `st.warning` for "check this", `st.error` for failures only.
Success boxes only confirm an action the user just took, never "nothing is due".

### Forms
Visible labels, helper text via `help=`, and related fields grouped with a bold subheading inside
one bordered container. Long forms are split into `st.tabs` or expanders (Parent Catalog form).

---

## Voice and formatting

- Dates: `Fri, Oct 10` within the school year, `Oct 10, 2027` beyond it. Never ISO on a student
  surface.
- Money: `$2,000 – $4,000`, `Up to $10,000`, `$500+`. "Unknown" becomes "Amount not listed".
- Counts in words a person uses: "1 essay, 1 letter", not "Challenge: 1" or "0/3 done". Prefer
  "3 steps left".
- Encouraging, not cute: "Nice, nothing due this week." with the next item and a link, not
  exclamation marks or confetti.

---

## Where styling lives (order is binding; see CLAUDE.md "UI/UX rules")

1. `.streamlit/config.toml` `[theme]`, `[theme.light]`, `[theme.dark]`, `[theme.sidebar]`
2. Native layout: `st.navigation`, `st.container(border, gap, horizontal, key)`, `st.badge`,
   `st.metric`, `st.tabs`, `st.expander`, `st.page_link`, `st.set_page_config(layout=...)`
3. CSS only in `phone_width_css()` in `app/helpers.py`, targeting `data-testid` or
   `.st-key-<key>`. Never `st-emotion-cache-*`.

Inline HTML is allowed only where no native element does the job, and must escape any catalog or
user text.

---

## Motion

Streamlit's built-in transitions only. No custom animation, no GSAP, no scroll reveal. Respect
`prefers-reduced-motion` in any CSS transition that ever gets added.

---

## Anti-patterns (do not use)

- ❌ Emoji as status or navigation icons
- ❌ "URGENT", all-caps alarms, or red for anything that isn't overdue or an error
- ❌ ML or pipeline vocabulary on a student surface (pipeline, snapshot, parquet, Top-N, weights,
  eligible=, P(Win), signal)
- ❌ A global title/caption block repeated on every page
- ❌ Success-green boxes for "nothing here"
- ❌ Raw hex in Python (`#FF4444` in `urgency_indicator`), so all color comes from theme tokens
- ❌ `st-emotion-cache-*` selectors, shadows, gradients, glassmorphism
- ❌ Google Fonts hot-linking
- ❌ Primary color that fails 4.5:1 against white button text

---

## Pre-delivery checklist (Streamlit)

- [ ] Screenshot every touched page at 400px and 1366px, in **both** light and dark system themes
- [ ] White-on-primary button text ≥ 4.5:1 in both themes (sample pixels, don't trust config)
- [ ] No emoji in any `st.badge`, nav label, alert or button
- [ ] Every badge has text and an icon; no meaning carried by color alone
- [ ] Tap targets ≥ 44px at 400px; no horizontal scroll at 400px
- [ ] Every empty state names a next action and links to it
- [ ] Student pages contain none of the operator vocabulary listed above
- [ ] No request to fonts.googleapis.com / fonts.gstatic.com in the browser network log
- [ ] `python -m pytest tests/ -q`, `ruff check src/ scripts/ app/ tests/`, `python -m mypy src/`,
      `python -m mypy app/`, `python scripts/validate_catalog.py` all green
