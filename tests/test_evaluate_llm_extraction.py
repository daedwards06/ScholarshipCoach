from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_llm_extraction import (
    aggregate_metrics,
    build_markdown_report,
    canonical_value,
    compare_record,
    gold_fields_for_row,
    overall_summary,
    run_extractions,
    scalar_matches,
    select_sample,
)


class _CountingClient:
    """Fake client that records calls and returns a scripted JSON body."""

    model = "fake-model/v1"

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response


def _snapshot_row(**overrides: object) -> dict[str, object]:
    row = {
        "scholarship_id": "a1",
        "title": "Tar Heel STEM Award",
        "description": "For NC undergraduates studying computer science.",
        "eligibility_text": "Minimum 3.0 GPA. Essay required.",
        "deadline": date(2026, 11, 30),
        "amount_min": 1000.0,
        "amount_max": 5000.0,
        "min_gpa": 3.0,
        "states_allowed": ["NC"],
        "majors_allowed": ["Computer Science"],
        "education_level": "Undergraduate",
        "citizenship": "US",
        "essay_required": True,
        "keywords": ["stem", "engineering"],
        "llm_enriched_fields": [],
    }
    row.update(overrides)
    return row


def test_canonical_value_puts_gold_and_prediction_in_the_same_form() -> None:
    assert canonical_value("deadline", date(2026, 11, 30)) == "2026-11-30"
    assert canonical_value("deadline", "2026-11-30") == "2026-11-30"
    assert canonical_value("deadline", pd.Timestamp("2026-11-30T12:00:00")) == "2026-11-30"
    assert canonical_value("amount_max", "5000") == 5000.0
    assert canonical_value("education_level", "Undergraduate") == "undergraduate"
    assert canonical_value("states_allowed", np.array(["nc", "SC"])) == ("NC", "SC")
    assert canonical_value("keywords", ["STEM", "stem"]) == ("stem",)


def test_canonical_value_treats_missing_and_unparsable_values_as_absent() -> None:
    assert canonical_value("deadline", None) is None
    assert canonical_value("deadline", "not a date") is None
    assert canonical_value("min_gpa", float("nan")) is None
    assert canonical_value("states_allowed", []) is None
    assert canonical_value("essay_required", "yes") is None, "gold booleans must be real bools"
    assert canonical_value("essay_required", False) is False


def test_scalar_matches_allows_one_dollar_tolerance_on_amounts() -> None:
    assert scalar_matches("amount_max", 5000.0, 5000.5)
    assert not scalar_matches("amount_max", 5000.0, 5002.0)
    assert not scalar_matches("min_gpa", 3.0, 3.5)
    assert not scalar_matches("deadline", "2026-11-30", None)


def test_gold_excludes_llm_filled_fields_and_survives_a_missing_column() -> None:
    row = _snapshot_row(llm_enriched_fields=["min_gpa", "citizenship"])
    gold = gold_fields_for_row(row)

    assert "min_gpa" not in gold
    assert "citizenship" not in gold
    assert gold["deadline"] == "2026-11-30"

    legacy_row = {
        key: value for key, value in _snapshot_row().items() if key != "llm_enriched_fields"
    }
    assert gold_fields_for_row(legacy_row)["min_gpa"] == 3.0


def test_select_sample_keeps_only_records_with_gold_and_source_text() -> None:
    frame = pd.DataFrame(
        [
            _snapshot_row(scholarship_id="has-both"),
            _snapshot_row(
                scholarship_id="no-source-text",
                description="",
                eligibility_text=None,
            ),
            _snapshot_row(
                scholarship_id="no-gold",
                deadline=None,
                amount_min=None,
                amount_max=None,
                min_gpa=None,
                states_allowed=[],
                majors_allowed=[],
                education_level=None,
                citizenship=None,
                essay_required=None,
                keywords=[],
            ),
            _snapshot_row(scholarship_id="all-gold-llm-filled", llm_enriched_fields=list(_snapshot_row())),
        ]
    )

    sample = select_sample(frame, sample_size=10, seed=0)

    assert list(sample["scholarship_id"]) == ["has-both"]


def test_select_sample_is_deterministic_and_capped() -> None:
    frame = pd.DataFrame([_snapshot_row(scholarship_id=f"id-{i:02d}") for i in range(10)])

    first = select_sample(frame, sample_size=4, seed=7)
    second = select_sample(frame, sample_size=4, seed=7)

    assert len(first) == 4
    assert list(first["scholarship_id"]) == list(second["scholarship_id"])
    assert list(first["scholarship_id"]) == sorted(first["scholarship_id"])


def test_scalar_metrics_separate_wrong_answers_from_abstentions() -> None:
    gold = {"min_gpa": 3.0}
    comparisons = [
        compare_record(gold, {"min_gpa": 3.0}),
        compare_record(gold, {"min_gpa": 2.5}),
        compare_record(gold, {}),
    ]

    metrics = aggregate_metrics(comparisons)["min_gpa"]

    assert metrics["gold_present"] == 3
    assert metrics["correct"] == 1
    assert metrics["answered"] == 2
    assert metrics["abstained"] == 1
    assert metrics["exact_match_accuracy"] == 1 / 3
    assert metrics["accuracy_when_answered"] == 1 / 2
    assert metrics["abstention_rate"] == 1 / 3


def test_values_supplied_where_gold_is_empty_are_counted_separately() -> None:
    comparisons = [
        compare_record({}, {"citizenship": "us"}),
        compare_record({}, {}),
    ]

    metrics = aggregate_metrics(comparisons)["citizenship"]

    assert metrics["gold_present"] == 0
    assert metrics["exact_match_accuracy"] is None, "undecidable cases must not become accuracy"
    assert metrics["gold_absent"] == 2
    assert metrics["value_added_when_gold_absent"] == 1
    assert metrics["unverified_value_added_rate"] == 0.5


def test_list_metrics_use_micro_averaged_set_precision_and_recall() -> None:
    comparisons = [
        compare_record({"states_allowed": ("NC", "SC")}, {"states_allowed": ["NC", "VA"]}),
        compare_record({"states_allowed": ("TX",)}, {"states_allowed": ["TX"]}),
        compare_record({"states_allowed": ("GA",)}, {}),
    ]

    metrics = aggregate_metrics(comparisons)["states_allowed"]

    assert metrics["gold_items"] == 4
    assert metrics["predicted_items"] == 3
    assert metrics["true_positives"] == 2
    assert metrics["set_precision"] == 2 / 3
    assert metrics["set_recall"] == 2 / 4
    assert metrics["set_f1"] == 2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5)
    assert metrics["exact_set_match"] == 1
    assert metrics["abstained"] == 1


def test_overall_summary_rolls_up_scalar_agreement_and_rates() -> None:
    comparisons = [
        compare_record(
            {"min_gpa": 3.0, "education_level": "undergraduate"},
            {"min_gpa": 3.0, "education_level": "graduate", "citizenship": "us"},
        ),
        compare_record({"min_gpa": 3.0}, {}),
    ]

    summary = overall_summary(aggregate_metrics(comparisons))

    assert summary["scalar_gold_present"] == 3
    assert summary["scalar_correct"] == 1
    assert summary["scalar_exact_match_accuracy"] == 1 / 3
    assert summary["abstained"] == 1, "a wrong education_level is an error, not an abstention"
    assert summary["value_added_when_gold_absent"] == 1


def test_markdown_report_states_the_gold_caveat_and_lists_every_field() -> None:
    comparisons = [compare_record({"min_gpa": 3.0}, {"min_gpa": 3.0})]
    metrics = aggregate_metrics(comparisons)

    markdown = build_markdown_report(
        generated_at="2026-08-13T00:00:00Z",
        snapshot_path=Path("data/processed/scholarships_snapshot_20260813.parquet"),
        snapshot_date="2026-08-13",
        snapshot_count=35,
        sample_size=1,
        model_name="fake-model",
        prompt_version="v1",
        provenance_column_present=True,
        extraction_stats={"cache_hits": 1, "api_calls": 0, "empty_extractions": 0},
        metrics=metrics,
        summary=overall_summary(metrics),
    )

    assert "not** human judgements" in markdown
    assert "unverified value-added" in markdown
    assert "Snapshot date: 2026-08-13" in markdown
    for field in ("min_gpa", "states_allowed", "keywords"):
        assert f"`{field}`" in markdown


def test_run_extractions_reuses_the_cache_and_respects_the_call_cap(tmp_path: Path) -> None:
    client = _CountingClient(json.dumps({"min_gpa": 3.0}))
    frame = pd.DataFrame(
        [
            _snapshot_row(scholarship_id="a", eligibility_text="Minimum 3.0 GPA."),
            _snapshot_row(scholarship_id="b", eligibility_text="Minimum 3.5 GPA."),
        ]
    )

    first, first_stats = run_extractions(
        frame,
        client=client,
        model_name=client.model,
        processed_dir=tmp_path,
        max_api_calls=1,
    )

    assert first_stats["api_calls"] == 1
    assert first_stats["skipped_at_cap"] == 1
    assert first == [{"min_gpa": 3.0}, None], "a capped record is unanswered, not empty"

    _, second_stats = run_extractions(
        frame,
        client=client,
        model_name=client.model,
        processed_dir=tmp_path,
        max_api_calls=5,
    )

    assert len(client.calls) == 2, "the first record must come from the cache on the re-run"
    assert second_stats["cache_hits"] == 1
    assert second_stats["api_calls"] == 1


def test_run_extractions_without_a_client_makes_no_calls(tmp_path: Path) -> None:
    frame = pd.DataFrame([_snapshot_row()])

    extractions, stats = run_extractions(
        frame,
        client=None,
        model_name="fake-model",
        processed_dir=tmp_path,
        max_api_calls=10,
    )

    assert extractions == [None], "no client and no cache means unanswered, not empty"
    assert stats == {
        "cache_hits": 0,
        "api_calls": 0,
        "api_failures": 0,
        "empty_extractions": 0,
        "skipped_at_cap": 0,
    }
