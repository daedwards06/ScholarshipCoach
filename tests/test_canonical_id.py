from __future__ import annotations

from datetime import date

from normalize.canonical_id import generate_scholarship_id


def test_generate_scholarship_id_is_stable_for_same_input() -> None:
    payload = {
        "title": "Future Leaders Scholarship",
        "sponsor": "Acme Foundation",
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "deadline": date(2026, 3, 1),
        "source_url": "https://www.example.org/scholarships/future-leaders",
    }

    first = generate_scholarship_id(**payload)
    second = generate_scholarship_id(**payload)

    assert first == second


def test_generate_scholarship_id_changes_when_title_changes() -> None:
    base = {
        "title": "Future Leaders Scholarship",
        "sponsor": "Acme Foundation",
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "deadline": date(2026, 3, 1),
        "source_url": "https://www.example.org/scholarships/future-leaders",
    }

    original = generate_scholarship_id(**base)
    changed = generate_scholarship_id(**{**base, "title": "Global Leaders Scholarship"})

    assert original != changed


def test_generate_scholarship_id_changes_when_deadline_changes() -> None:
    base = {
        "title": "Future Leaders Scholarship",
        "sponsor": "Acme Foundation",
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "deadline": date(2026, 3, 1),
        "source_url": "https://www.example.org/scholarships/future-leaders",
    }

    original = generate_scholarship_id(**base)
    changed = generate_scholarship_id(**{**base, "deadline": date(2026, 4, 1)})

    assert original != changed
