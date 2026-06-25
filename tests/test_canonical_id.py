from __future__ import annotations

from datetime import date

import pytest

from normalize.canonical_id import generate_scholarship_id

_BASE_PAYLOAD = {
    "title": "Future Leaders Scholarship",
    "sponsor": "Acme Foundation",
    "amount_min": 1000.0,
    "amount_max": 5000.0,
    "deadline": date(2026, 3, 1),
    "source_url": "https://www.example.org/scholarships/future-leaders",
}


def test_generate_scholarship_id_is_stable_for_same_input() -> None:
    first = generate_scholarship_id(**_BASE_PAYLOAD)
    second = generate_scholarship_id(**_BASE_PAYLOAD)

    assert first == second


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("title", "Global Leaders Scholarship"),
        ("deadline", date(2026, 4, 1)),
        ("sponsor", "Other Foundation"),
        ("amount_max", 9999.0),
    ],
)
def test_generate_scholarship_id_changes_when_field_changes(
    field: str, new_value: object
) -> None:
    original = generate_scholarship_id(**_BASE_PAYLOAD)
    changed = generate_scholarship_id(**{**_BASE_PAYLOAD, field: new_value})

    assert original != changed
