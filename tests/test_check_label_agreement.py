from __future__ import annotations

from scripts.check_label_agreement import label_agreement


def test_label_agreement_flags_sharp_and_orders_worst_first() -> None:
    human = {"p": {"a": 0, "b": 1, "c": 2, "d": 2}}
    proxy = {"p": {"a": 2, "b": 1, "c": 1, "d": 2}}

    rows = label_agreement(human, proxy)

    # One row per comparable pair; ordered by descending delta.
    assert [r["scholarship_id"] for r in rows] == ["a", "c", "b", "d"]
    worst = rows[0]
    assert worst == {
        "profile_id": "p",
        "scholarship_id": "a",
        "human": 0,
        "proxy": 2,
        "delta": 2,
        "sharp": True,
    }
    # delta 1 is a mismatch but not sharp; delta 0 is agreement.
    assert rows[1]["sharp"] is False and rows[1]["delta"] == 1
    assert rows[3]["delta"] == 0 and rows[3]["sharp"] is False


def test_label_agreement_skips_pairs_absent_from_proxy() -> None:
    human = {"p": {"a": 2, "missing": 1}}
    proxy = {"p": {"a": 2}}

    rows = label_agreement(human, proxy)

    assert len(rows) == 1
    assert rows[0]["scholarship_id"] == "a"


def test_label_agreement_skips_profiles_without_proxy() -> None:
    human = {"unknown_profile": {"a": 2}}
    proxy: dict[str, dict[str, int]] = {}

    assert label_agreement(human, proxy) == []


def test_label_agreement_respects_custom_sharp_delta() -> None:
    human = {"p": {"a": 1}}
    proxy = {"p": {"a": 0}}

    rows = label_agreement(human, proxy, sharp_delta=1)

    assert rows[0]["sharp"] is True
