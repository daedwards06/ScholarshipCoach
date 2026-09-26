"""The whole money picture: what a year of school costs, minus what was won.

Outside scholarships are the small lever.  A school's net price and its
outside-award displacement policy are the big ones, and neither arrives in any
feed this project ingests -- the family types them in off the school's own net
price calculator.  So this module does arithmetic over two hand-kept tables
(``colleges``) and one earned one (``outcomes``), and does it as pure
functions over dataclasses so the numbers can be tested without Streamlit.

Two things it deliberately does not do:

* It does not apply a displacement policy.  "Reduces institutional grant
  dollar for dollar" is written next to the school as text a parent reads
  before the student spends a weekend on an essay; turning it into a
  subtraction would invent precision the family does not have.
* It does not spread a one-time award across four years.  ``remaining`` is
  this college's net price minus everything won to date, which is the honest
  first-year number and is captioned as such in the UI.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from src.profile.grade_levels import school_year_end
from src.store.milestones import school_year_label
from src.store.repo import Application, College, Outcome, net_price

WON_RESULT = "won"

UNDATED_YEAR_LABEL = "Year not recorded"


@dataclass(frozen=True, slots=True)
class YearTotal:
    """What was won in one school year, undated awards in a bucket of their own."""

    year_end: int | None
    label: str
    total: float
    count: int


@dataclass(frozen=True, slots=True)
class RenewalCondition:
    """A won award that keeps paying only while the student holds up terms."""

    application_id: int
    award_title: str
    terms: str
    amount: float | None


@dataclass(frozen=True, slots=True)
class CollegeCost:
    """One row of the college table, with the gap this family still has to cover."""

    college_id: int
    name: str
    in_state: bool
    sticker_price: float | None
    net_price_estimate: float | None
    net_price: float | None
    deadline_type: str
    merit_aid_notes: str
    outside_award_policy: str
    remaining: float | None


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    """One decided application, as the Outcomes page lists it."""

    application_id: int
    award_title: str
    cycle_year: int | None
    result: str
    amount: float | None
    renewal_terms: str
    paid_to: str
    decided_on: str


@dataclass(frozen=True, slots=True)
class MoneySummary:
    total_won: float
    award_count: int
    by_year: list[YearTotal]
    renewals: list[RenewalCondition]
    colleges: list[CollegeCost]


def is_won(outcome: Outcome) -> bool:
    return str(outcome.result or "").strip().casefold() == WON_RESULT


def won_amount(outcome: Outcome) -> float:
    """The dollars this outcome brought in; a win with no amount yet is zero."""
    if not is_won(outcome) or outcome.amount_awarded is None:
        return 0.0
    return float(outcome.amount_awarded)


def award_year(outcome: Outcome) -> int | None:
    """The school year the decision landed in, or ``None`` when undated."""
    try:
        decided = date.fromisoformat(str(outcome.decided_on or "")[:10])
    except ValueError:
        return None
    return school_year_end(decided)


def total_won(outcomes: Iterable[Outcome]) -> float:
    return sum(won_amount(outcome) for outcome in outcomes)


def won_by_year(outcomes: Iterable[Outcome]) -> list[YearTotal]:
    """Totals per school year, earliest first, undated awards last."""
    totals: dict[int | None, list[float]] = {}
    for outcome in outcomes:
        if not is_won(outcome):
            continue
        totals.setdefault(award_year(outcome), []).append(won_amount(outcome))

    rows = [
        YearTotal(
            year_end=year,
            label=UNDATED_YEAR_LABEL if year is None else school_year_label(year),
            total=sum(amounts),
            count=len(amounts),
        )
        for year, amounts in totals.items()
    ]
    rows.sort(key=lambda row: (row.year_end is None, row.year_end or 0))
    return rows


def renewal_conditions(
    outcomes: Iterable[Outcome], titles: Mapping[int, str] | None = None
) -> list[RenewalCondition]:
    """Won awards that carry terms, so the conditions stay in one visible list."""
    lookup = titles or {}
    rows = [
        RenewalCondition(
            application_id=outcome.application_id,
            award_title=lookup.get(outcome.application_id, "")
            or f"Application {outcome.application_id}",
            terms=str(outcome.renewal_terms or "").strip(),
            amount=outcome.amount_awarded,
        )
        for outcome in outcomes
        if is_won(outcome) and str(outcome.renewal_terms or "").strip()
    ]
    rows.sort(key=lambda row: (row.award_title.casefold(), row.application_id))
    return rows


def _parse_day(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        return None


def cycle_year(application: Application, fallback_deadline: date | None = None) -> int | None:
    """The award cycle an application belongs to, keyed on its deadline.

    Falls back to the snapshot's deadline, then the submission year, then the
    year the award was saved.
    """
    for candidate in (
        _parse_day(application.deadline),
        fallback_deadline,
        _parse_day(application.submitted_on),
        _parse_day(application.created_at),
    ):
        if candidate is not None:
            return candidate.year
    return None


def outcome_rows(
    applications: Iterable[Application], outcomes: Iterable[Outcome]
) -> list[OutcomeRow]:
    """Join each outcome to its application, newest cycle first."""
    by_id = {application.id: application for application in applications}
    rows: list[OutcomeRow] = []
    for outcome in outcomes:
        application = by_id.get(outcome.application_id)
        title = ""
        if application is not None:
            title = application.title or application.catalog_id
        rows.append(
            OutcomeRow(
                application_id=outcome.application_id,
                award_title=title or f"Application {outcome.application_id}",
                cycle_year=None if application is None else cycle_year(application),
                result=str(outcome.result or "").strip().casefold(),
                amount=outcome.amount_awarded,
                renewal_terms=str(outcome.renewal_terms or "").strip(),
                paid_to=str(outcome.paid_to or "").strip(),
                decided_on=str(outcome.decided_on or "")[:10],
            )
        )
    rows.sort(
        key=lambda row: (-(row.cycle_year or 0), row.award_title.casefold(), row.application_id)
    )
    return rows


def college_cost(college: College, won: float) -> CollegeCost:
    """One college's row, with ``won`` subtracted from its net price."""
    price = net_price(college)
    return CollegeCost(
        college_id=college.id,
        name=college.name,
        in_state=college.in_state,
        sticker_price=college.cost_of_attendance,
        net_price_estimate=college.net_price_estimate,
        net_price=price,
        deadline_type=college.deadline_type,
        merit_aid_notes=college.merit_aid_notes,
        outside_award_policy=college.outside_award_policy,
        remaining=None if price is None else price - won,
    )


def money_summary(
    colleges: Iterable[College],
    outcomes: Iterable[Outcome],
    titles: Mapping[int, str] | None = None,
) -> MoneySummary:
    """Everything the Colleges & Money view shows, computed once."""
    results = list(outcomes)
    won = total_won(results)
    return MoneySummary(
        total_won=won,
        award_count=sum(1 for outcome in results if is_won(outcome)),
        by_year=won_by_year(results),
        renewals=renewal_conditions(results, titles),
        colleges=[college_cost(college, won) for college in colleges],
    )
