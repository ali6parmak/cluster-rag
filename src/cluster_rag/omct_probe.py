"""OMCT probe driver (Phase 2, PLAN.md last Phase 2 checkbox).

Evidence for embedding-model choice on the collaborator pairwise dataset
(``data/external/omct_pairs.json``, strictly auxiliary per PLAN.md section 2;
runs get the ``omct-`` dataset prefix and are never reported as UPR metrics):

per candidate embedder and question-text transform:

- **cosine-similarity AUC** separating judged pairs (label 0 vs 1),
- **question retrieval** hit@k / recall@k ranking distinct GTI questions:
  scope ``all`` (every distinct question) and scope ``same_document`` (only
  questions judged alongside a paragraph's document),
- **truncation diagnostics** against the model context: share of paragraphs
  whose token count exceeds the budget and those rows' retrieval metrics vs
  the unaffected rest.

Two question transforms exist (question furniture as signal or noise,
PLAN.md section 7 answer): ``strip_block`` (drop the "Question: " label and
the trailing "Answer choices:"/"Notes:" blocks) is the default, and ``keep``
(text as-is) remains available for an explicit side-by-side comparison by
listing both in ``probe.question_transforms``. Paragraphs are preprocessed
with the config variant (default raw); transforms are scored against the same
paragraph vectors.

Each variant also reports **score distributions** (Phase 4 thresholding
evidence, strictly auxiliary): judged-pair cosines by label, and the best
same-document-candidate similarity of paragraphs with only negative
judgements ("should answer nothing" rows).

The embedding backend is injected (protocol port) so tests run a deterministic
fake; the metrics are pure functions from :mod:`cluster_rag.eval.metrics`.
This module orchestrates backend + metrics for the CLI (like ``cli.py``) and
stays free of heavy model libraries itself.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.config import ProbeConfig
from cluster_rag.embed.base import Embedder
from cluster_rag.embed.cache import CachedEmbedder
from cluster_rag.eval.metrics import pairwise_retrieval_metrics, roc_auc, subset_pair_metrics
from cluster_rag.pairs import PairRecord
from cluster_rag.preprocess import (
    PreprocessSpec,
    apply_question_transform,
    question_transform_version,
)

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE: float = 4.0  # English-prose heuristic when no tokenizer exists
TOKEN_SOURCE_TOKENIZER = "tokenizer-exact"  # HF embedders only
TOKEN_SOURCE_ESTIMATE = f"chars-over-{CHARS_PER_TOKEN_ESTIMATE:g}-estimate"  # Ollama fallback

PROBE_QUANTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _quantile_dict(values: np.ndarray) -> dict[str, float]:
    """Quantile snapshot with JSON-friendly string probability keys."""
    return {f"{p:g}": float(np.quantile(values, p)) for p in PROBE_QUANTILES}


def _distribution_block(values: np.ndarray, n_total: int) -> dict[str, Any]:
    """Count + mean/min/max/quantiles for one judged-score group (may be empty)."""
    block: dict[str, Any] = {"n": n_total}
    if values.size:
        block.update(
            mean=float(np.mean(values)),
            min=float(values.min()),
            max=float(values.max()),
            quantiles=_quantile_dict(values),
        )
    return block


def score_distributions(
    similarities: np.ndarray,
    corpus: ProbeCorpus,
    gold: Sequence[set[int]],
) -> dict[str, Any]:
    """Score-distribution evidence for threshold design (auxiliary line only).

    Report what judged same-topic (label 1) vs not-same-topic (label 0) pair
    cosines look like, plus — the "should answer nothing" rows of PLAN Phase
    4 — the best similarity a paragraph reaches over its own same-document
    candidate questions when *every* judged pair on it is label 0 (no gold).
    Pure numpy; describes the OMCT data only and is never a UPR metric — UPR
    operating points are calibrated on UPR val.
    """
    pair_scores = similarities[corpus.rows, corpus.columns]
    positive = pair_scores[corpus.labels == 1]
    negative = pair_scores[corpus.labels == 0]

    goldless_rows = [row for row, columns in enumerate(gold) if not columns]
    goldless_best: list[float] = []
    without_candidates = 0
    for row in goldless_rows:
        candidates = [
            column
            for column in range(len(corpus.questions))
            if corpus.question_documents[column] & corpus.paragraph_documents[row]
        ]
        if candidates:
            goldless_best.append(float(similarities[row, candidates].max()))
        else:
            without_candidates += 1

    return {
        "note": (
            "auxiliary threshold-calibration evidence (OMCT line, never a UPR "
            "metric): judged pair cosine distributions by label, and the best "
            "same-document-candidate similarity of paragraphs whose judged "
            "pairs are all label 0 (the 'should answer nothing' rows); UPR "
            "operating points are calibrated on UPR val"
        ),
        "judged_label1": _distribution_block(positive, int(positive.size)),
        "judged_label0": _distribution_block(negative, int(negative.size)),
        "goldless_paragraphs": {
            "n_paragraph_rows": len(goldless_rows),
            "n_without_same_doc_candidates": without_candidates,
            "best_same_doc_similarity": _distribution_block(
                np.asarray(goldless_best, dtype=np.float64), len(goldless_best)
            ),
        },
    }


@dataclass(frozen=True)
class ProbeCacheOptions:
    """Disk-cache wiring for the probe (from the config's cache section).

    Paragraphs and questions are cached under *different* preprocessing
    versions (the question transform is part of the question side's
    preprocessing), so the orchestrator builds one wrapper per side.
    """

    root: str


def _cache_totals(*wrappers: object) -> dict[str, int]:
    """Aggregate hit/miss counters across the probe's cache wrappers."""
    return {
        "cache_hits": sum(getattr(w, "cache_hits", 0) for w in wrappers),
        "cache_misses": sum(getattr(w, "cache_misses", 0) for w in wrappers),
        "n_stored": sum(getattr(w, "n_stored", 0) for w in wrappers),
    }


@dataclass(frozen=True)
class ProbeCorpus:
    """Deduplicated texts for one question transform plus the judged indices.

    Texts are deduplicated by *transformed* text in first-occurrence order; the
    judge triples ``rows``/``columns``/``labels`` align to the judged pair list
    in file order. Column collisions introduced by the transform (two questions
    stripped to the same text) collapse into one column; document sets per row
    and column back the same-document retrieval scope.
    """

    transform: str
    paragraphs: list[str]
    questions: list[str]
    paragraph_row: dict[str, int]  # transformed paragraph text -> row index
    question_col: dict[str, int]  # transformed question text -> column index
    paragraph_documents: list[set[str]]  # per row
    question_documents: list[set[str]]  # per column
    labels: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    n_paragraph_texts_changed: int
    n_question_texts_changed: int
    n_question_collisions: int  # original questions collapsed by the transform
    question_chars_before: tuple[int, ...]  # per original question text
    question_chars_after: tuple[int, ...]  # per original question text

    @property
    def n_judged_pairs(self) -> int:
        """Number of judged (paragraph, question) pairs carried along."""
        return int(self.labels.size)


def build_probe_corpus(
    records: Sequence[PairRecord],
    *,
    transform: str,
    markers: Sequence[str],
    preprocess_spec: PreprocessSpec,
) -> ProbeCorpus:
    """Deduplicate texts, apply the question transform and the preprocessing variant.

    Paragraphs get ``preprocess_spec`` (default ``raw``); questions get the
    ``keep``/``strip_block`` transform. Both sides deduplicate by transformed
    text so each unique text is embedded exactly once.
    """
    original_paragraphs = list(dict.fromkeys(record.paragraph_text for record in records))
    original_questions = list(dict.fromkeys(record.question_text for record in records))
    paragraph_texts = preprocess_spec.apply(original_paragraphs)
    transformed_questions = apply_question_transform(original_questions, transform, markers)

    doc_of_paragraph: dict[str, set[str]] = {}
    doc_of_question: dict[str, set[str]] = {}
    for record in records:
        doc_of_paragraph.setdefault(record.paragraph_text, set()).add(record.document)
        doc_of_question.setdefault(record.question_text, set()).add(record.document)

    paragraph_row = {text: row for row, text in enumerate(dict.fromkeys(paragraph_texts))}
    question_col = {text: col for col, text in enumerate(dict.fromkeys(transformed_questions))}

    paragraph_documents: list[set[str]] = [set() for _ in paragraph_row]
    for original, transformed in zip(original_paragraphs, paragraph_texts, strict=True):
        paragraph_documents[paragraph_row[transformed]] |= doc_of_paragraph[original]
    question_documents: list[set[str]] = [set() for _ in question_col]
    for original, transformed in zip(original_questions, transformed_questions, strict=True):
        question_documents[question_col[transformed]] |= doc_of_question[original]

    paragraph_row_of_original = {
        original: paragraph_row[transformed]
        for original, transformed in zip(original_paragraphs, paragraph_texts, strict=True)
    }
    question_col_of_original = {
        original: question_col[transformed]
        for original, transformed in zip(original_questions, transformed_questions, strict=True)
    }
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    rows = np.asarray(
        [paragraph_row_of_original[record.paragraph_text] for record in records], dtype=np.int64
    )
    columns = np.asarray(
        [question_col_of_original[record.question_text] for record in records], dtype=np.int64
    )

    return ProbeCorpus(
        transform=transform,
        paragraphs=list(paragraph_row),
        questions=list(question_col),
        paragraph_row=paragraph_row,
        question_col=question_col,
        paragraph_documents=paragraph_documents,
        question_documents=question_documents,
        labels=labels,
        rows=rows,
        columns=columns,
        n_paragraph_texts_changed=sum(
            before != after
            for before, after in zip(original_paragraphs, paragraph_texts, strict=True)
        ),
        n_question_texts_changed=sum(
            before != after
            for before, after in zip(original_questions, transformed_questions, strict=True)
        ),
        n_question_collisions=len(original_questions) - len(question_col),
        question_chars_before=tuple(len(text) for text in original_questions),
        question_chars_after=tuple(len(transformed) for transformed in transformed_questions),
    )


def gold_by_paragraph(corpus: ProbeCorpus) -> list[set[int]]:
    """Gold question column per paragraph row (questions judged label 1)."""
    gold: list[set[int]] = [set() for _ in corpus.paragraphs]
    for row, column, label in zip(corpus.rows, corpus.columns, corpus.labels, strict=True):
        if label == 1:
            gold[int(row)].add(int(column))
    return gold


def same_document_columns(corpus: ProbeCorpus) -> list[list[int]]:
    """Per-row candidate columns: questions judged alongside the row's document.

    A question column is a candidate for a paragraph row when their document
    sets intersect (same-report judging). A document set only grows from pairs
    that judged the two texts together, so every row with at least one judged
    pair keeps its own partner columns; the all-columns fallback only guards
    degenerate cases.
    """
    all_columns = list(range(len(corpus.questions)))
    scopes: list[list[int]] = []
    for row in range(len(corpus.paragraphs)):
        docs = corpus.paragraph_documents[row]
        columns = [
            column for column in all_columns if corpus.question_documents[column] & docs
        ] or all_columns
        scopes.append(columns)
    return scopes


def group_breakdown(
    records: Sequence[PairRecord],
    corpus: ProbeCorpus,
    similarities: np.ndarray,
    gold: Sequence[set[int]],
    *,
    same_doc_columns: Sequence[Sequence[int]],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Per (document, source_csv) judged-pair breakdown with counts beside numbers.

    The overall probe numbers can hide document-level effects: per-document
    class balance differs strongly (e.g. 80% positive in one column-source
    document vs 25% in another) and the smallest document carries only ~99 of
    the 1520 pairs. For each group this reports the pair counts, the pair-AUC
    restricted to the group's judged pairs, and the same-document retrieval
    metrics restricted to the group's paragraph rows — reusing the pure
    metrics (:func:`roc_auc`, :func:`subset_pair_metrics`) with group masks.
    A group without both classes reports ``pair_auc: null`` with a note
    instead of inventing a number.
    """
    pair_indices_by_group: dict[tuple[str, str], list[int]] = {}
    for pair_index, record in enumerate(records):
        key = (record.document, record.source_csv)
        pair_indices_by_group.setdefault(key, []).append(pair_index)

    groups: list[dict[str, Any]] = []
    for (document, source_csv), pair_indices in pair_indices_by_group.items():
        pair_rows = np.asarray(corpus.rows[pair_indices], dtype=np.int64)
        labels = [int(corpus.labels[pair_index]) for pair_index in pair_indices]
        scores = similarities[pair_rows, np.asarray(corpus.columns)[pair_indices]].tolist()
        try:
            pair_auc: float | None = roc_auc(labels, scores)
            pair_auc_note = None
        except ValueError as exc:
            pair_auc, pair_auc_note = None, str(exc)

        row_mask = np.zeros(len(corpus.paragraphs), dtype=bool)
        row_mask[sorted({int(row) for row in pair_rows})] = True
        metrics = subset_pair_metrics(
            similarities,
            gold,
            ks,
            row_mask,
            bucket=f"document={document!r} source_csv={source_csv!r}",
            candidate_columns=same_doc_columns,
        )
        groups.append(
            {
                "document": document,
                "source_csv": source_csv,
                "n_pairs": len(pair_indices),
                "n_positive": sum(label == 1 for label in labels),
                "n_negative": sum(label == 0 for label in labels),
                "pair_auc": pair_auc,
                "pair_auc_note": pair_auc_note,
                "n_rows": int(metrics["n_rows"]),
                "n_rows_with_gold": int(metrics["n_rows_with_gold"]),
                "same_document": metrics if metrics["n_rows_with_gold"] else None,
            }
        )

    return {
        "n_groups": len(groups),
        "n_pairs": sum(group["n_pairs"] for group in groups),
        "n_positive": sum(group["n_positive"] for group in groups),
        "groups": groups,
    }


def paragraph_token_counts(embedder: Embedder, texts: Sequence[str]) -> tuple[str, np.ndarray]:
    """Token count per text; exact via an HF tokenizer, else a chars/4 estimate.

    HF embedders expose an extra ``token_counts`` helper (duck-typed port,
    outside the :class:`Embedder` protocol); Ollama has no client-side
    tokenizer, so the token count is estimated (documented in the report).
    """
    counts = getattr(embedder, "token_counts", None)
    if callable(counts):
        return TOKEN_SOURCE_TOKENIZER, np.asarray(counts(texts), dtype=np.int64)
    estimated = [math.ceil(len(text) / CHARS_PER_TOKEN_ESTIMATE) for text in texts]
    return TOKEN_SOURCE_ESTIMATE, np.asarray(estimated, dtype=np.int64)


def truncation_report(
    similarities: np.ndarray,
    gold: Sequence[set[int]],
    tokens: np.ndarray,
    context_tokens: int | None,
    token_source: str,
    ks: Sequence[int],
) -> dict[str, Any]:
    """Truncation diagnostics: paragraphs over the model context vs the rest.

    ``metrics_over``/``metrics_within`` carry
    :func:`~cluster_rag.eval.metrics.subset_pair_metrics` for the two row
    groups, so a quality hit caused by server-side truncation is measurable,
    not assumed.
    """
    report: dict[str, Any] = {
        "context_tokens": context_tokens,
        "token_source": token_source,
        "n_paragraphs": int(tokens.size),
        "mean_tokens": float(np.mean(tokens)) if tokens.size else 0.0,
        "max_tokens": int(np.max(tokens)) if tokens.size else 0,
    }
    if not context_tokens:
        report["note"] = "model context unknown: truncation share cannot be measured"
        return report
    over_mask = tokens > context_tokens
    within_mask = ~over_mask
    n_over = int(over_mask.sum())
    over_metrics = (
        subset_pair_metrics(similarities, gold, ks, over_mask, bucket="paragraphs_over_context")
        if n_over
        else None
    )
    within_metrics = (
        subset_pair_metrics(similarities, gold, ks, within_mask, bucket="paragraphs_within_context")
        if n_over < tokens.size
        else None
    )
    report.update(
        {
            "n_paragraphs_over_context": n_over,
            "share_paragraphs_over_context": n_over / tokens.size if tokens.size else 0.0,
            "mean_tokens_within_context": float(np.mean(tokens[within_mask]))
            if n_over < tokens.size
            else None,
            "mean_tokens_over_context": float(np.mean(tokens[over_mask])) if n_over else None,
            "metrics_over_context": over_metrics,
            "metrics_within_context": within_metrics,
        }
    )
    return report


def _embedder_info_dict(embedder: Embedder) -> dict[str, Any]:
    """JSON-serializable provenance of the embedder used."""
    info = embedder.info
    return {
        "backend": info.backend,
        "model_id": info.model_id,
        "dimension": info.dimension,
        "revision_or_digest": info.revision_or_digest,
        "context_tokens": info.context_tokens,
        "prefixes": dict(info.prefixes),
        "extra": info.extra,
    }


def run_probe(
    records: Sequence[PairRecord],
    embedder: Embedder,
    *,
    preprocess_spec: PreprocessSpec,
    probe_config: ProbeConfig,
    cache: ProbeCacheOptions | None = None,
) -> dict[str, Any]:
    """Run the probe: embed paragraphs and questions, score all judged pairs.

    Paragraphs are embedded as ``kind="document"`` (the indexed side of an
    asymmetric model), questions as ``kind="query"`` — matching how the
    classifier will treat its texts and candidates.

    ``cache`` (optional) wires the disk cache: paragraph vectors are keyed by
    the preprocessing spec version (shared across transforms), question
    vectors by the question-transform version (kept per transform).
    """
    records = list(records)
    if not records:
        raise ValueError("the pairwise dataset is empty: nothing to probe")
    if not probe_config.question_transforms:
        raise ValueError("probe.question_transforms must name at least one transform")

    document_counts: dict[str, dict[str, int]] = {}
    for record in records:
        entry = document_counts.setdefault(record.document, {"n": 0, "positive": 0})
        entry["n"] += 1
        entry["positive"] += record.label == 1

    corpus_stats: dict[str, Any] = {
        "n_pairs": len(records),
        "n_positive_pairs": int(sum(record.label == 1 for record in records)),
        "n_negative_pairs": int(sum(record.label == 0 for record in records)),
        "n_distinct_paragraph_texts": len({record.paragraph_text for record in records}),
        "n_distinct_question_texts": len({record.question_text for record in records}),
        "n_documents": len(document_counts),
        "pairs_per_document": dict(sorted(document_counts.items())),
    }

    paragraph_embedder = embedder
    cache_wrappers: list[object] = []
    if cache is not None:
        paragraph_embedder = CachedEmbedder(
            embedder, cache.root, preprocessing_version=preprocess_spec.version
        )
        cache_wrappers.append(paragraph_embedder)

    ks = list(probe_config.recall_ks)
    variant_reports: list[dict[str, Any]] = []
    for transform in probe_config.question_transforms:
        started = time.perf_counter()
        corpus = build_probe_corpus(
            records,
            transform=transform,
            markers=probe_config.question_block_markers,
            preprocess_spec=preprocess_spec,
        )
        question_version = question_transform_version(
            transform, probe_config.question_block_markers
        )
        question_embedder = embedder
        if cache is not None:
            question_embedder = CachedEmbedder(
                embedder, cache.root, preprocessing_version=question_version
            )
            cache_wrappers.append(question_embedder)
        paragraph_matrix = paragraph_embedder.encode(corpus.paragraphs, "document")
        question_matrix = question_embedder.encode(corpus.questions, "query")
        similarities = paragraph_matrix @ question_matrix.T  # rows = paragraphs

        pair_scores = similarities[corpus.rows, corpus.columns]
        pair_auc = roc_auc(corpus.labels.tolist(), pair_scores.tolist())

        gold = gold_by_paragraph(corpus)
        same_doc_columns = same_document_columns(corpus)
        retrieval = {
            "all": pairwise_retrieval_metrics(similarities, gold, ks),
            "same_document": pairwise_retrieval_metrics(
                similarities, gold, ks, candidate_columns=same_doc_columns
            ),
        }
        breakdown = group_breakdown(
            records,
            corpus,
            similarities,
            gold,
            same_doc_columns=same_doc_columns,
            ks=ks,
        )
        token_source, tokens = paragraph_token_counts(embedder, corpus.paragraphs)
        truncation = truncation_report(
            similarities, gold, tokens, embedder.info.context_tokens, token_source, ks
        )

        variant_reports.append(
            {
                "transform": transform,
                "question_version": question_version,
                "paragraph_version": preprocess_spec.version,
                "n_distinct_question_texts_after": len(corpus.questions),
                "n_question_collisions": corpus.n_question_collisions,
                "n_question_texts_changed": corpus.n_question_texts_changed,
                "n_paragraph_texts_changed": corpus.n_paragraph_texts_changed,
                "question_chars_mean_before": float(np.mean(corpus.question_chars_before)),
                "question_chars_mean_after": float(np.mean(corpus.question_chars_after)),
                "pair_auc": pair_auc,
                "retrieval": retrieval,
                "group_breakdown": breakdown,
                "truncation": truncation,
                "score_distributions": score_distributions(similarities, corpus, gold),
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        logger.info(
            "Probe variant %r: AUC %.4f, hit@1 all %.3f same-doc %.3f (%.1fs)",
            transform,
            pair_auc,
            retrieval["all"]["hit@1"],
            retrieval["same_document"]["hit@1"],
            variant_reports[-1]["seconds"],
        )
    return {
        "corpus": corpus_stats,
        "embedder": _embedder_info_dict(embedder),
        "cache": None if cache is None else _cache_totals(*cache_wrappers),
        "variants": variant_reports,
    }
