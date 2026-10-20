"""Tests for the auxiliary pairwise loader and stats (Phase 1.5)."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from cluster_rag.cli import main
from cluster_rag.pairs import PairRecord, load_pairs, summarize_pairs


def _good_pair() -> dict:
    return {
        "pair_id": "P-1",
        "source_csv": "s.csv",
        "document": "Doc",
        "paragraph_number": "1",
        "paragraph_text": "para",
        "question_number": 7,
        "question_text": "question",
        "label": 1,
        "label_origin": "merged",
    }


def test_load_sample_pairs(sample_pairs) -> None:
    """The fixture loads cleanly and passes top-level metadata through."""
    report = load_pairs(sample_pairs)
    assert report.n_invalid == 0
    assert report.n_valid == 6
    assert report.metadata["n_pairs"] == 6
    assert report.metadata["name"] == "sample-pairs"
    assert "pairs" not in report.metadata


def test_pair_record_validation() -> None:
    """Binary labels only; blank texts and extra keys are rejected."""
    good = _good_pair()
    assert PairRecord.model_validate(good).label == 1

    with pytest.raises(ValidationError, match="0 or 1"):
        PairRecord.model_validate(good | {"label": 2})
    with pytest.raises(ValidationError, match="blank"):
        PairRecord.model_validate(good | {"question_text": "   "})
    with pytest.raises(ValidationError):
        PairRecord.model_validate(good | {"unexpected": 1})


def test_load_pairs_reports_malformed_entries(tmp_path) -> None:
    """Bad entries are reported by 0-based index and dropped; the load does not fail."""
    good = _good_pair()
    payload = {
        "name": "tiny",
        "pairs": [
            good,
            good | {"pair_id": ""},  # blank id
            good | {"label": 2},  # non-binary label
            good | {"question_text": "  "},  # blank text
        ],
    }
    source = tmp_path / "pairs.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    report = load_pairs(source)
    assert report.n_valid == 1
    assert report.metadata["name"] == "tiny"
    assert set(report.errors) == {1, 2, 3}


def test_load_pairs_rejects_bad_top_level(tmp_path) -> None:
    source = tmp_path / "pairs.json"
    source.write_text(json.dumps({"pairs": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="top-level 'pairs'"):
        load_pairs(source)


def test_summarize_pairs(sample_pairs) -> None:
    """Hand-computed stats on the fixture: balance, questions, documents."""
    records = load_pairs(sample_pairs).records
    stats = summarize_pairs(records)

    assert stats["n_pairs"] == 6
    assert stats["label_counts"] == {"1": 3, "0": 3}
    assert stats["positive_share"] == 0.5
    assert stats["n_distinct_question_numbers"] == 3  # 10, 11, 12
    assert stats["n_distinct_question_texts"] == 4  # two questions are re-judged on new paragraphs
    assert stats["n_documents"] == 3
    assert stats["document_counts"] == {"Doc A": 2, "Doc B": 3, "Doc C": 1}
    assert stats["positive_by_document"]["Doc B"] == {"n": 3, "positive": 1}
    assert stats["n_duplicate_pair_ids"] == 0
    # cross-document text repeats (Phase 2.5 log correction): the fixture itself
    # has none (q10's shared number hides two different texts), so 0/0 here; the
    # real dataset's 0 paragraphs / 186 questions show up in the aux-base stats run
    assert stats["n_paragraph_texts_in_multiple_documents"] == 0
    assert stats["n_question_texts_in_multiple_documents"] == 0


def test_summarize_pairs_counts_cross_document_repeats(tmp_path) -> None:
    """A question text judged in several documents is counted once, docs > 1."""
    content = {
        "pairs": [
            {
                "pair_id": "K-1",
                "source_csv": "csvX",
                "document": "Doc A",
                "paragraph_number": "1",
                "paragraph_text": "para one",
                "question_number": 1,
                "question_text": "shared question",
                "label": 1,
                "label_origin": "merged",
            },
            {
                "pair_id": "K-2",
                "source_csv": "csvY",
                "document": "Doc B",
                "paragraph_number": "1",
                "paragraph_text": "para two",
                "question_number": 1,
                "question_text": "shared question",
                "label": 0,
                "label_origin": "reviewer",
            },
        ],
        "n_pairs": 2,
        "n_positive": 1,
        "n_negative": 1,
        "name": "cross-doc",
    }
    path = tmp_path / "cross.json"
    path.write_text(json.dumps(content), encoding="utf-8")
    stats = summarize_pairs(load_pairs(path).records)
    assert stats["n_paragraph_texts_in_multiple_documents"] == 0  # paras repeat within docs only
    assert stats["n_question_texts_in_multiple_documents"] == 1  # "shared question" in Both docs


def test_stats_pairs_command_end_to_end(sample_pairs, write_config, tmp_path, capsys) -> None:
    """`stats --pairs` writes the omct-prefixed report next to the resolved config."""
    config_file = write_config(sample_pairs, run_id="pairs", data_external=sample_pairs)
    assert main(["stats", "--pairs", "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "Pairs: 6 valid, 0 malformed entry/entries skipped" in out
    assert "3 positive / 3 negative" in out

    results_dir = tmp_path / "results" / "omct-pairs"
    payload = json.loads((results_dir / "pairs_stats.json").read_text("utf-8"))
    assert payload["run_id"] == "omct-pairs"
    assert payload["n_valid_pairs"] == 6
    assert payload["pair_stats"]["n_documents"] == 3
    assert (results_dir / "config.yaml").is_file()


def test_stats_pairs_warns_on_prefixed_run_id(sample_pairs, write_config, tmp_path, caplog) -> None:
    """A run_id that already carries a dataset prefix would double up: warn, don't fail."""
    config_file = write_config(sample_pairs, run_id="omct-pairs", data_external=sample_pairs)
    with caplog.at_level(logging.WARNING):
        assert main(["stats", "--pairs", "--config", str(config_file)]) == 0
    assert any("omct-omct-pairs" in record.message for record in caplog.records)
    assert (tmp_path / "results" / "omct-omct-pairs" / "pairs_stats.json").is_file()


def test_stats_pairs_requires_data_external(sample_pairs, write_config, caplog) -> None:
    """Without paths.data_external the command fails with a clear message."""
    config_file = write_config(sample_pairs, run_id="pairs")
    with caplog.at_level(logging.ERROR):
        assert main(["stats", "--pairs", "--config", str(config_file)]) == 1
    assert any("data_external" in record.message for record in caplog.records)
