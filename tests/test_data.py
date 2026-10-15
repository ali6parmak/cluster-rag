"""Tests for the Record model, the JSONL loader and the record summary."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cluster_rag.data import Record, load_jsonl, summarize_records


def test_load_sample_jsonl(sample_jsonl) -> None:
    """Smoke test: the fixture loads cleanly and preserves order."""
    report = load_jsonl(sample_jsonl)
    assert report.n_invalid == 0
    assert report.n_valid == 20
    ids = [record.id for record in report.records]
    assert len(set(ids)) == len(ids)
    for record in report.records:
        assert record.text.strip()
        assert record.language
        assert len(record.labels) >= 1


def test_record_rejects_unknown_fields() -> None:
    """Schema drift (extra keys) must fail loudly, not be ignored."""
    payload = {"id": "x", "text": "t", "language": "en", "labels": ["a"], "extra": 1}
    with pytest.raises(ValidationError):
        Record.model_validate(payload)


def test_record_rejects_blank_text_and_labels() -> None:
    """Whitespace-only text and blank labels are malformed."""
    with pytest.raises(ValidationError):
        Record.model_validate({"id": "x", "text": "   ", "language": "en", "labels": ["a"]})
    with pytest.raises(ValidationError):
        Record.model_validate({"id": "x", "text": "t", "language": "en", "labels": ["a", "  "]})
    with pytest.raises(ValidationError):
        Record.model_validate({"id": "x", "text": "t", "language": "en", "labels": []})


def test_malformed_lines_are_reported_and_dropped(tmp_path) -> None:
    """Bad rows are collected with line numbers; the load itself does not fail."""
    source = tmp_path / "malformed.jsonl"
    rows = [
        json.dumps({"id": "ok-1", "text": "valid", "language": "en", "labels": ["a"]}),
        "{not valid json",
        json.dumps({"id": "ok-2", "language": "en", "labels": ["a"]}),  # missing text
        json.dumps({"id": "ok-3", "text": "no labels", "language": "en", "labels": []}),
        "   ",  # blank line: skipped silently, not an error
    ]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = load_jsonl(source)
    assert report.n_valid == 1
    assert report.records[0].id == "ok-1"
    assert set(report.errors) == {2, 3, 4}
    assert report.errors[2].startswith("invalid JSON")
    assert "text" in report.errors[3]
    assert "labels" in report.errors[4]


def test_summarize_records() -> None:
    """Hand-computed summary on three tiny records."""
    rows = [
        {"id": "1", "text": "alpha", "language": "en", "labels": ["A", "B"]},
        {"id": "2", "text": "beta", "language": "en", "labels": ["A", "C"]},
        {"id": "3", "text": "gamma gamma", "language": "es", "labels": ["D"]},
    ]
    records = [Record.model_validate(row) for row in rows]
    summary = summarize_records(records)
    assert summary["n_records"] == 3
    assert summary["n_unique_labels"] == 4
    assert summary["language_counts"] == {"en": 2, "es": 1}
    assert summary["labels_per_record"] == {"min": 1, "mean": 5 / 3, "max": 2}
    assert summary["text_length_chars"]["max"] == 11  # len("gamma gamma")
    assert summary["top_labels"][0] == ("A", 2)


def test_summarize_records_empty() -> None:
    """The summary handles an empty record list without dividing by zero."""
    summary = summarize_records([])
    assert summary["n_records"] == 0
    assert summary["labels_per_record"]["mean"] == 0.0
    assert summary["top_labels"] == []
