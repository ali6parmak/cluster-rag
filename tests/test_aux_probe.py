"""Tests for the auxiliary probe (Phase 2): corpus build, run structure, CLI.

Uses the deterministic fake embedder and the sample_pairs fixture; no network,
no model downloads. Numeric invariants only (fake vectors carry no semantics).
"""

from __future__ import annotations

import json
import logging

import pytest

from cluster_rag import cli
from cluster_rag.aux_probe import (
    build_probe_corpus,
    gold_by_paragraph,
    run_probe,
    same_document_columns,
)
from cluster_rag.cli import main
from cluster_rag.config import PreprocessConfig, ProbeConfig
from cluster_rag.pairs import load_pairs
from cluster_rag.preprocess import build_preprocess_spec
from fake_embedder import FakeEmbedder

STRIP_PAIRS = """
{"pairs": [
  {"pair_id": "X-1", "source_csv": "s", "document": "Doc A",
   "paragraph_number": "1", "paragraph_text": "Paragraph one about evictions",
   "question_number": 1,
   "question_text": "Question: How many were evicted?\\n\\nAnswer choices:\\n- Often\\n- Never",
   "label": 1, "label_origin": "merged"},
  {"pair_id": "X-2", "source_csv": "s", "document": "Doc A",
   "paragraph_number": "1", "paragraph_text": "Paragraph one about evictions",
   "question_number": 2, "question_text": "How many were evicted?Notes: see manual",
   "label": 0, "label_origin": "merged"},
  {"pair_id": "X-3", "source_csv": "s", "document": "Doc B",
   "paragraph_number": "1", "paragraph_text": "Paragraph two about prisons",
   "question_number": 3, "question_text": "Question: Are prisons overcrowded?",
   "label": 1, "label_origin": "merged"}
],
"n_pairs": 3, "n_positive": 2, "n_negative": 1, "name": "strip-fixture",
"question_text_sources": ["review.csv"]}
"""


@pytest.fixture()
def strip_pairs_file(tmp_path) -> object:
    path = tmp_path / "strip_pairs.json"
    path.write_text(STRIP_PAIRS, encoding="utf-8")
    return path


def _probe_config(**overrides) -> ProbeConfig:
    defaults = {
        "question_transforms": ("keep", "strip_block"),
        "question_block_markers": ("Answer choices:", "Notes:"),
        "recall_ks": (1, 10),
    }
    return ProbeConfig(**{**defaults, **overrides})


def test_build_corpus_dedups_and_keeps_judgement_alignment(sample_pairs) -> None:
    """6 judged pairs -> 6 paragraph rows, 4 question columns; triple alignment."""
    records = load_pairs(sample_pairs).records
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    corpus = build_probe_corpus(
        records, transform="keep", markers=("Answer choices:", "Notes:"), preprocess_spec=spec
    )
    assert len(corpus.paragraphs) == 6
    assert len(corpus.questions) == 4  # q10 appears in Doc A and Doc C (same text)
    assert corpus.n_judged_pairs == 6
    assert corpus.labels.tolist() == [1, 0, 1, 0, 0, 1]
    # the two re-judged questions collapse onto the same column by text
    assert corpus.question_col["How often did forced evictions occur?"] >= 0
    gold = gold_by_paragraph(corpus)
    assert len(gold) == 6
    assert sum(len(g) for g in gold) == 3  # three positive pairs
    # same-document scope exists for every row (each row is judged with its doc)
    scopes = same_document_columns(corpus)
    assert all(scope for scope in scopes)


def test_corpus_question_strip_transform(strip_pairs_file) -> None:
    """strip_block removes the furniture, counts changes and merges collisions."""
    records = load_pairs(strip_pairs_file).records
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    corpus = build_probe_corpus(
        records,
        transform="strip_block",
        markers=("Answer choices:", "Notes:"),
        preprocess_spec=spec,
    )
    assert len(corpus.questions) == 2  # two distinct stripped questions
    assert corpus.n_question_collisions == 1
    assert corpus.n_question_texts_changed == 3  # X-1, X-2 (Notes cut), X-3 lose the label
    assert len(corpus.paragraphs) == 2
    assert corpus.labels.tolist() == [1, 0, 1]
    assert corpus.rows.tolist() == [0, 0, 1]
    assert corpus.columns.tolist() == [0, 0, 1]  # collision kept the judgement alignment
    assert corpus.question_chars_before[0] > corpus.question_chars_after[0]


def test_corpus_keep_transform_is_identity(strip_pairs_file) -> None:
    """keep: no changes, no collisions, no text removed."""
    records = load_pairs(strip_pairs_file).records
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    corpus = build_probe_corpus(
        records, transform="keep", markers=("Answer choices:", "Notes:"), preprocess_spec=spec
    )
    assert corpus.n_question_texts_changed == 0
    assert corpus.n_question_collisions == 0
    assert len(corpus.questions) == 3
    assert min(corpus.question_chars_before) <= min(corpus.question_chars_after)


def test_run_probe_structure_and_invariants(strip_pairs_file) -> None:
    """Structural contract of the report; fake embedder with a documented prefix."""
    records = load_pairs(strip_pairs_file).records
    embedder = FakeEmbedder(
        model_id="fake-embedder", dimension=8, prefixes={"query": "q: ", "document": "d: "}
    )
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    report = run_probe(records, embedder, preprocess_spec=spec, probe_config=_probe_config())

    corpus_stats = report["corpus"]
    assert corpus_stats["n_pairs"] == 3
    assert corpus_stats["n_positive_pairs"] == 2
    assert corpus_stats["n_distinct_paragraph_texts"] == 2
    assert corpus_stats["n_distinct_question_texts"] == 3
    assert corpus_stats["n_documents"] == 2

    assert report["embedder"]["model_id"] == "fake-embedder"
    assert report["embedder"]["backend"] == "fake"
    assert report["embedder"]["dimension"] == 8

    transforms = {variant["transform"]: variant for variant in report["variants"]}
    assert set(transforms) == {"keep", "strip_block"}
    for variant in transforms.values():
        assert 0.0 <= variant["pair_auc"] <= 1.0
        for scope in ("all", "same_document"):
            metrics = variant["retrieval"][scope]
            assert metrics["n_rows"] == 2  # distinct paragraphs
            assert metrics["n_rows_with_gold"] == 2  # both paragraphs have a positive pair
            assert 0.0 <= metrics["hit@1"] <= 1.0
            assert 0.0 <= metrics["hit@10"] <= 1.0
            assert "recall@10" in metrics
        truncation = variant["truncation"]
        assert truncation["context_tokens"] is None  # the fake has no context
        assert "note" in truncation
        assert truncation["token_source"] == "chars-over-4-estimate"
        assert truncation["mean_tokens"] > 0
    # question texts changed only under strip_block
    assert transforms["keep"]["n_question_texts_changed"] == 0
    assert transforms["strip_block"]["n_question_texts_changed"] == 3
    assert transforms["strip_block"]["question_version"].startswith(
        "question-transform-strip_block"
    )


def test_run_probe_truncation_buckets_reported(strip_pairs_file) -> None:
    """A embedder exposing token_counts + context gets over/within bucket metrics."""

    class ContextFake(FakeEmbedder):
        """Fake with one token per character and a tiny context."""

        def __init__(self) -> None:
            super().__init__(model_id="fake-small-context", dimension=8)

        def token_counts(self, texts):

            return [len(text) for text in texts]

    from cluster_rag.embed.base import EmbedderInfo

    embedder = ContextFake()
    info = embedder._info
    embedder._info = EmbedderInfo(  # 28 tokens: "Paragraph one..." (29 ch) is over
        model_id=info.model_id,
        backend=info.backend,
        dimension=info.dimension,
        revision_or_digest=info.revision_or_digest,
        prefixes=info.prefixes,
        context_tokens=28,
        extra=info.extra,
    )
    records = load_pairs(strip_pairs_file).records
    spec = build_preprocess_spec(PreprocessConfig(variant="raw"))
    report = run_probe(records, embedder, preprocess_spec=spec, probe_config=_probe_config())
    truncation = report["variants"][0]["truncation"]
    assert truncation["token_source"] == "tokenizer-exact"
    assert truncation["mean_tokens"] > 0
    # 29-char paragraph is over the 28-token context; 27-char one is within
    assert truncation["n_paragraphs_over_context"] == 1
    assert truncation["share_paragraphs_over_context"] == 0.5
    over, within = truncation["metrics_over_context"], truncation["metrics_within_context"]
    assert over["bucket"] == "paragraphs_over_context" and over["n_rows"] == 1
    assert within["bucket"] == "paragraphs_within_context" and within["n_rows"] == 1
    assert {"hit@1", "recall@1"} <= set(over) and {"hit@1", "recall@1"} <= set(within)


PROBE_CONFIG_EXTRA = """
embed:
  backend: fake
  model_id: fake-embedder
  prefixes:
    query: "query: "
    document: ""
preprocess:
  variant: raw
"""


def test_probe_aux_command_end_to_end(
    sample_pairs, write_config, tmp_path, monkeypatch, capsys
) -> None:
    """CLI: builds the embedder, writes results/aux-<run_id>/ and a frozen config."""
    monkeypatch.setattr(
        cli, "build_embedder", lambda config: FakeEmbedder(model_id="fake-embedder", dimension=8)
    )
    config_file = write_config(
        sample_pairs, run_id="probe-test", data_external=sample_pairs, extra=PROBE_CONFIG_EXTRA
    )
    assert main(["probe-aux", "--config", str(config_file)]) == 0
    out = capsys.readouterr().out
    assert "Auxiliary probe: aux-probe-test" in out
    assert "pair-AUC" in out

    results_dir = tmp_path / "results" / "aux-probe-test"
    payload = json.loads((results_dir / "probe_report.json").read_text("utf-8"))
    assert payload["run_id"] == "aux-probe-test"
    assert payload["n_valid_pairs"] == 6
    assert payload["embedder"]["model_id"] == "fake-embedder"
    assert len(payload["probe"]["variants"]) == 2
    assert (results_dir / "config.yaml").is_file()
    # paragraph vectors were encoded once and shared between the transforms
    assert payload["cache"]["cache_hits"] > 0
    assert payload["cache"]["cache_misses"] == 6 + 4 + 4  # paragraphs + 2 x questions


def test_probe_aux_requires_data_external(sample_pairs, write_config, caplog) -> None:
    """Without paths.data_external the command fails with a clear message."""
    config_file = write_config(sample_pairs, run_id="probe-test", extra=PROBE_CONFIG_EXTRA)
    with caplog.at_level(logging.ERROR):
        assert main(["probe-aux", "--config", str(config_file)]) == 1
    assert any("data_external" in record.message for record in caplog.records)


def test_probe_aux_requires_embed_section(sample_pairs, write_config, caplog, monkeypatch) -> None:
    config_file = write_config(sample_pairs, run_id="probe-test", data_external=sample_pairs)
    with caplog.at_level(logging.ERROR):
        assert main(["probe-aux", "--config", str(config_file)]) == 1
    assert any("embed: section" in record.message for record in caplog.records)
