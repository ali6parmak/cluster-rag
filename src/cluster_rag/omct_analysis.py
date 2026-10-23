"""The text-processing track's Stage-0 analysis driver (``analyze-omct``).

The Stage-0 artifact of TEXT_PROCESSING_PLAN.md section 3 (Stage 0.1): one
JSON file per run that characterizes the OMCT pairwise data as a
**pair-relevance benchmark**. It carries, per model:

- the score of every judged pair (1,520 of them: the verified positives and
  the reviewer-verified hard negatives) and its rank percentile inside the
  paragraph's own same-document candidate list;
- the **control set** (subset B): a seeded, proportional sample of the
  *unjudged* same-document cells, with their scores and their rank
  percentiles, and the full provenance of the draw (seed, rule, per-document
  quotas, the excluded judged cells) so a later stage re-uses exactly the same
  cells;
- the per-document strata with their base rates, the score distributions per
  label, and the negative-enrichment measurement (the rank percentile of the
  verified negatives next to the positives and the control).

The dataset-level part (the grid, the strata, the control draw) is
model-independent and computed once per run; the scores are per model.

Like :mod:`cluster_rag.omct_probe` (the Phase-2 probe driver), this module
orchestrates the embedder (a port, injected by the CLI), the pure metrics and
the track's pure core (:mod:`cluster_rag.text_processors.control`). It never
imports a model client itself: the CLI wires the backends.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder
from cluster_rag.eval.metrics import roc_auc
from cluster_rag.pairs import PairRecord
from cluster_rag.preprocess import (
    PreprocessSpec,
    apply_question_transform,
    question_transform_version,
)
from cluster_rag.text_processors.base import Chunker
from cluster_rag.text_processors.control import (
    SAMPLING_RULE_DESCRIPTIONS,
    SAMPLING_RULE_PROPORTIONAL,
    ControlSample,
    DocumentGrid,
    rank_percentiles,
    sample_control,
)

logger = logging.getLogger(__name__)

# The quantile grid of the reported distributions (mirrors the probe's).
DISTRIBUTION_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

# The pairwise file's own id convention: "<code>-<paragraph number>-Q<question
# number>" (e.g. "NU-10-Q2"). The code is the document's short handle.
_CELL_ID_PATTERN = re.compile(r"^(?P<code>[^-]+)-\d+-Q\d+$")
_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _slug(text: str) -> str:
    """A readable, filesystem-safe slug of a name (empty collapses to "-")."""
    slug = "".join(character if character in _SLUG_SAFE else "_" for character in text)
    return slug[:60] or "-"


def document_code(document: str, pair_ids: Sequence[str]) -> str:
    """The short code of a document, used in the artifact's cell ids.

    The pairwise file's pair ids follow the convention
    ``<code>-<paragraph number>-Q<question number>``, so a document's code is
    recovered from its ids when they agree on one. Otherwise a readable slug
    of the document name is used. The mapping is recorded in the artifact, so
    every cell id can be resolved back to its document, paragraph and
    question.
    """
    codes: set[str] = set()
    for pair_id in pair_ids:
        match = _CELL_ID_PATTERN.match(pair_id)
        if match is None:
            return _slug(document)
        codes.add(match.group("code"))
    if len(codes) == 1:
        return codes.pop()
    return _slug(document)


def _distribution(values: np.ndarray) -> dict[str, Any]:
    """Count + mean/min/max + quantiles of one score group (empty groups are safe)."""
    block: dict[str, Any] = {"n": int(values.size)}
    if values.size:
        block.update(
            {
                "mean": float(np.mean(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "quantiles": {
                    f"{p:g}": float(np.quantile(values, p)) for p in DISTRIBUTION_QUANTILES
                },
            }
        )
    return block


def _quantile(values: np.ndarray, probability: float) -> float | None:
    """One quantile of a group (``None`` for an empty group)."""
    if not values.size:
        return None
    return float(np.quantile(values, probability))


@dataclass(frozen=True)
class DatasetView:
    """The model-independent view of the pairwise dataset.

    Holds the deduplicated paragraph and question texts (the rows and columns
    of the score matrix), the per-document grids, the drawn control sample
    and the per-document strata. The model-independent part of the artifact
    is built from this object.
    """

    pairs: tuple[PairRecord, ...]
    paragraphs: tuple[str, ...]
    questions: tuple[str, ...]
    paragraph_row: dict[str, int]
    question_col: dict[str, int]
    paragraph_numbers: tuple[str, ...]
    question_numbers: tuple[int, ...]
    paragraph_documents: tuple[frozenset[str], ...]
    question_documents: tuple[frozenset[str], ...]
    grids: tuple[DocumentGrid, ...]
    control: ControlSample
    strata: tuple[dict[str, Any], ...]
    documents: tuple[dict[str, Any], ...]
    pooled: dict[str, Any]
    judged_cells: tuple[dict[str, Any], ...]

    def candidate_columns(self) -> list[tuple[int, ...]]:
        """The candidate questions of every paragraph row (its document's questions)."""
        by_document = {grid.document: grid.question_cols for grid in self.grids}
        candidates: list[tuple[int, ...]] = []
        for documents in self.paragraph_documents:
            columns: set[int] = set()
            for document in documents:
                columns.update(by_document.get(document, ()))
            candidates.append(tuple(sorted(columns)))
        return candidates


def build_dataset_view(
    records: Sequence[PairRecord],
    *,
    control_size: int,
    seed: int,
    sampling_rule: str = SAMPLING_RULE_PROPORTIONAL,
) -> DatasetView:
    """Build the model-independent view: the grid, the strata and the control.

    Args:
        records: The judged pairs of the pairwise dataset.
        control_size: The number of control cells to draw.
        seed: The seed of the control draw (from the run's config).
        sampling_rule: The control's sampling rule.

    Returns:
        The :class:`DatasetView` of the run.

    Raises:
        ValueError: If the dataset is empty or the control cannot be drawn.
    """
    pairs = tuple(records)
    if not pairs:
        raise ValueError("the pairwise dataset is empty: nothing to analyze")

    paragraphs = tuple(dict.fromkeys(record.paragraph_text for record in pairs))
    questions = tuple(dict.fromkeys(record.question_text for record in pairs))
    paragraph_row = {text: index for index, text in enumerate(paragraphs)}
    question_col = {text: index for index, text in enumerate(questions)}

    paragraph_numbers: list[str] = [""] * len(paragraphs)
    question_numbers: list[int] = [0] * len(questions)
    paragraph_documents: list[set[str]] = [set() for _ in paragraphs]
    question_documents: list[set[str]] = [set() for _ in questions]
    for record in pairs:
        row = paragraph_row[record.paragraph_text]
        column = question_col[record.question_text]
        if not paragraph_numbers[row]:
            paragraph_numbers[row] = record.paragraph_number
        if not question_numbers[column]:
            question_numbers[column] = record.question_number
        paragraph_documents[row].add(record.document)
        question_documents[column].add(record.document)

    documents = sorted({record.document for record in pairs})
    grids: list[DocumentGrid] = []
    for document in documents:
        rows = tuple(index for index, docs in enumerate(paragraph_documents) if document in docs)
        columns = tuple(index for index, docs in enumerate(question_documents) if document in docs)
        judged = frozenset(
            (paragraph_row[record.paragraph_text], question_col[record.question_text])
            for record in pairs
            if record.document == document
        )
        grids.append(
            DocumentGrid(
                document=document,
                paragraph_rows=rows,
                question_cols=columns,
                judged=judged,
            )
        )

    control = sample_control(grids, size=control_size, seed=seed, rule=sampling_rule)

    positive_paragraphs: set[int] = set()
    pairs_per_document: dict[str, dict[str, int]] = {}
    source_csvs_per_document: dict[str, set[str]] = {}
    for record in pairs:
        entry = pairs_per_document.setdefault(
            record.document, {"n": 0, "positive": 0, "negative": 0}
        )
        entry["n"] += 1
        entry["positive"] += record.label == 1
        entry["negative"] += record.label == 0
        source_csvs_per_document.setdefault(record.document, set()).add(record.source_csv)
        row = paragraph_row[record.paragraph_text]
        column = question_col[record.question_text]
        if record.label == 1:
            positive_paragraphs.add(row)

    codes = {
        document: document_code(
            document,
            [record.pair_id for record in pairs if record.document == document],
        )
        for document in documents
    }
    judged_cells: list[dict[str, Any]] = []
    for record in pairs:
        row = paragraph_row[record.paragraph_text]
        column = question_col[record.question_text]
        judged_cells.append(
            {
                "pair_id": record.pair_id,
                "cell_id": _cell_id(
                    codes[record.document],
                    paragraph_numbers[row],
                    question_numbers[column],
                ),
                "document": record.document,
                "paragraph_number": record.paragraph_number,
                "question_number": record.question_number,
                "label": record.label,
            }
        )

    document_reports: list[dict[str, Any]] = []
    strata: list[dict[str, Any]] = []
    for grid in grids:
        counts = pairs_per_document.get(grid.document, {"n": 0, "positive": 0, "negative": 0})
        n_pairs = counts["n"]
        rows = grid.paragraph_rows
        n_without_positive = sum(1 for row in rows if row not in positive_paragraphs)
        document_reports.append(
            {
                "document": grid.document,
                "code": codes[grid.document],
                "n_paragraphs": len(rows),
                "n_questions": len(grid.question_cols),
                "n_source_csvs": len(source_csvs_per_document.get(grid.document, set())),
            }
        )
        strata.append(
            {
                "document": grid.document,
                "n_pairs": n_pairs,
                "n_positive": counts["positive"],
                "n_negative": counts["negative"],
                "positive_share": counts["positive"] / n_pairs if n_pairs else 0.0,
                "n_paragraphs": len(rows),
                "n_questions": len(grid.question_cols),
                "n_paragraphs_with_positive": len(rows) - n_without_positive,
                "n_paragraphs_without_positive": n_without_positive,
                "grid_cells": grid.n_cells,
                "judged_cells": grid.n_judged,
                "unjudged_cells": grid.n_unjudged,
                "grid_coverage": grid.n_judged / grid.n_cells if grid.n_cells else 0.0,
                "control_quota": control.quotas.get(grid.document, 0),
                "control_drawn": sum(1 for cell in control.cells if cell[0] == grid.document),
            }
        )

    total_pairs = len(pairs)
    total_positive = sum(record.label == 1 for record in pairs)
    total_cells = sum(grid.n_cells for grid in grids)
    pooled = {
        "n_pairs": total_pairs,
        "n_positive": total_positive,
        "n_negative": total_pairs - total_positive,
        "positive_share": total_positive / total_pairs if total_pairs else 0.0,
        "macro_positive_share": (
            sum(stratum["positive_share"] for stratum in strata) / len(strata) if strata else 0.0
        ),
        "n_paragraphs": len(paragraphs),
        "n_questions": len(questions),
        "n_paragraphs_without_positive": sum(
            stratum["n_paragraphs_without_positive"] for stratum in strata
        ),
        "grid_cells": total_cells,
        "judged_cells": sum(grid.n_judged for grid in grids),
        "unjudged_cells": sum(grid.n_unjudged for grid in grids),
        "grid_coverage": sum(grid.n_judged for grid in grids) / total_cells if total_cells else 0.0,
        "n_documents": len(grids),
    }

    return DatasetView(
        pairs=pairs,
        paragraphs=paragraphs,
        questions=questions,
        paragraph_row=paragraph_row,
        question_col=question_col,
        paragraph_numbers=tuple(paragraph_numbers),
        question_numbers=tuple(question_numbers),
        paragraph_documents=tuple(frozenset(docs) for docs in paragraph_documents),
        question_documents=tuple(frozenset(docs) for docs in question_documents),
        grids=tuple(grids),
        control=control,
        strata=tuple(strata),
        documents=tuple(document_reports),
        pooled=pooled,
        judged_cells=tuple(judged_cells),
    )


def side_cache_versions(
    chunker: Chunker,
    preprocess_spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
) -> tuple[str, str]:
    """The embedding-cache version slot of each side (the chunking composed in).

    The paragraph side's base version is the preprocessing version, the
    question side's the question-transform version; the chunker's
    ``cache_version`` composes the chunking version into that **existing**
    slot, so the cache key layout never changes (the full-text chunker keeps
    the version unchanged, which is what keeps the run warm).
    """
    question_version = question_transform_version(question_transform, question_block_markers)
    return (
        chunker.cache_version(preprocess_spec.version),
        chunker.cache_version(question_version),
    )


def _encode_side(
    embedder: Embedder,
    chunker: Chunker,
    texts: Sequence[str],
    kind: str,
) -> tuple[np.ndarray, int]:
    """Embed one side through the chunker; returns the vectors and the chunk count.

    The Stage-0 baseline aggregates a text's chunks by taking its first
    chunk: with the full-text chunker that *is* the whole text, so the score
    is the plain full-text cosine the plan's indicative numbers used. The
    chunk count is reported, so a later stage's aggregation is never implied.
    """
    chunk_lists = chunker.segment(list(texts))
    chunks: list[str] = []
    for text, chunk_list in zip(texts, chunk_lists, strict=True):
        if not chunk_list:
            raise ValueError(f"the chunker produced no chunk for a text: {text[:60]!r}")
        chunks.append(chunk_list[0])
    vectors = embedder.encode(chunks, kind)  # type: ignore[arg-type]
    return vectors, len(chunks)


def _cache_counters(embedder: object) -> tuple[int, int, int]:
    """The cache hit/miss/stored counters of an embedder (0 without a cache)."""
    return (
        getattr(embedder, "cache_hits", 0),
        getattr(embedder, "cache_misses", 0),
        getattr(embedder, "n_stored", 0),
    )


def score_model(
    view: DatasetView,
    *,
    model_name: str,
    paragraph_embedder: Embedder,
    question_embedder: Embedder,
    chunker: Chunker,
    preprocess_spec: PreprocessSpec,
    question_transform: str,
    question_block_markers: Sequence[str],
) -> dict[str, Any]:
    """Score the judged pairs and the control cells with one model.

    Paragraphs are embedded under the ``document`` kind, questions under
    ``query`` (the same assignment the probe uses: the paragraph is the
    indexed text, the question the query it is matched against). Both sides
    go through the track's chunker, so the artifact records the chunker and
    its version next to the scores.
    """
    paragraph_texts = preprocess_spec.apply(list(view.paragraphs))
    question_texts = apply_question_transform(
        list(view.questions), question_transform, question_block_markers
    )
    paragraph_vectors, n_paragraph_chunks = _encode_side(
        paragraph_embedder, chunker, paragraph_texts, "document"
    )
    question_vectors, n_question_chunks = _encode_side(
        question_embedder, chunker, question_texts, "query"
    )
    similarities = paragraph_vectors @ question_vectors.T

    candidate_columns = view.candidate_columns()
    rows = np.asarray(
        [view.paragraph_row[record.paragraph_text] for record in view.pairs], dtype=np.int64
    )
    columns = np.asarray(
        [view.question_col[record.question_text] for record in view.pairs], dtype=np.int64
    )
    labels = np.asarray([record.label for record in view.pairs], dtype=np.int64)
    pair_scores = similarities[rows, columns]
    pair_percentiles = rank_percentiles(similarities, rows, columns, candidate_columns)

    control_rows = np.asarray([row for _, row, _ in view.control.cells], dtype=np.int64)
    control_columns = np.asarray([col for _, _, col in view.control.cells], dtype=np.int64)
    control_scores = similarities[control_rows, control_columns]
    control_percentiles = rank_percentiles(
        similarities, control_rows, control_columns, candidate_columns
    )

    codes = {report["document"]: report["code"] for report in view.documents}
    judged_pairs: list[dict[str, Any]] = []
    for position, cell in enumerate(view.judged_cells):
        judged_pairs.append(
            {
                **cell,
                "score": float(pair_scores[position]),
                "rank_percentile": float(pair_percentiles[position]),
            }
        )

    control_cells: list[dict[str, Any]] = []
    for position, (document, row, column) in enumerate(view.control.cells):
        control_cells.append(
            {
                "cell_id": _cell_id(
                    codes[document],
                    view.paragraph_numbers[row],
                    view.question_numbers[column],
                ),
                "document": document,
                "paragraph_number": view.paragraph_numbers[row],
                "question_number": view.question_numbers[column],
                "score": float(control_scores[position]),
                "rank_percentile": float(control_percentiles[position]),
            }
        )

    positive_scores = pair_scores[labels == 1]
    negative_scores = pair_scores[labels == 0]
    positive_percentiles = pair_percentiles[labels == 1]
    negative_percentiles = pair_percentiles[labels == 0]

    score_distributions: dict[str, Any] = {
        "label1": _distribution(positive_scores),
        "label0": _distribution(negative_scores),
        "control": _distribution(control_scores),
        "by_document": {},
    }
    enrichment_by_document: dict[str, Any] = {}
    for stratum in view.strata:
        document = stratum["document"]
        mask = np.asarray([record.document == document for record in view.pairs], dtype=bool)
        document_cells = [
            (index, cell) for index, cell in enumerate(view.control.cells) if cell[0] == document
        ]
        document_control_scores = np.asarray(
            [control_scores[index] for index, _ in document_cells], dtype=np.float64
        )
        document_control_percentiles = np.asarray(
            [control_percentiles[index] for index, _ in document_cells], dtype=np.float64
        )
        score_distributions["by_document"][document] = {
            "label1": _distribution(pair_scores[mask & (labels == 1)]),
            "label0": _distribution(pair_scores[mask & (labels == 0)]),
            "control": _distribution(document_control_scores),
        }
        enrichment_by_document[document] = {
            "rank_percentile": {
                "label1": _distribution(pair_percentiles[mask & (labels == 1)]),
                "label0": _distribution(pair_percentiles[mask & (labels == 0)]),
                "control": _distribution(document_control_percentiles),
            },
            "top_quintile_share": {
                "label1": _top_quintile_share(pair_percentiles[mask & (labels == 1)]),
                "label0": _top_quintile_share(pair_percentiles[mask & (labels == 0)]),
                "control": _top_quintile_share(document_control_percentiles),
            },
        }

    pair_auc: dict[str, Any] = {"overall": None, "by_document": {}}
    try:
        pair_auc["overall"] = roc_auc(labels.tolist(), pair_scores.tolist())
    except ValueError as exc:  # a document without both classes
        pair_auc["overall_note"] = str(exc)
    for stratum in view.strata:
        document = stratum["document"]
        mask = np.asarray([record.document == document for record in view.pairs], dtype=bool)
        try:
            pair_auc["by_document"][document] = roc_auc(
                labels[mask].tolist(), pair_scores[mask].tolist()
            )
        except ValueError as exc:
            pair_auc["by_document"][document] = None
            pair_auc.setdefault("notes", {})[document] = str(exc)

    top1 = _top1_provenance(view, similarities, candidate_columns)

    paragraph_hits, paragraph_misses, _ = _cache_counters(paragraph_embedder)
    question_hits, question_misses, _ = _cache_counters(question_embedder)
    info = paragraph_embedder.info

    return {
        "name": model_name,
        "embedder": {
            "backend": info.backend,
            "model_id": info.model_id,
            "dimension": info.dimension,
            "revision_or_digest": info.revision_or_digest,
            "context_tokens": info.context_tokens,
            "prefixes": dict(info.prefixes),
        },
        "sides": {
            "paragraph": {
                "kind": "document",
                "preprocess": preprocess_spec.variant,
                "preprocess_version": preprocess_spec.version,
                "chunker": chunker.version,
                "chunker_name": _chunker_name(chunker),
                "cache_version": chunker.cache_version(preprocess_spec.version),
                "n_texts": len(paragraph_texts),
                "n_chunks": n_paragraph_chunks,
                "aggregation": "first chunk per text (the full-text baseline: one chunk per text)",
                "cache_hits": paragraph_hits,
                "cache_misses": paragraph_misses,
            },
            "question": {
                "kind": "query",
                "transform": question_transform,
                "transform_version": question_transform_version(
                    question_transform, question_block_markers
                ),
                "chunker": chunker.version,
                "chunker_name": _chunker_name(chunker),
                "cache_version": chunker.cache_version(
                    question_transform_version(question_transform, question_block_markers)
                ),
                "n_texts": len(question_texts),
                "n_chunks": n_question_chunks,
                "aggregation": "first chunk per text (the full-text baseline: one chunk per text)",
                "cache_hits": question_hits,
                "cache_misses": question_misses,
            },
        },
        "pair_auc": pair_auc,
        "score_distributions": score_distributions,
        "negative_enrichment": {
            "rank_percentile": {
                "label1": _distribution(positive_percentiles),
                "label0": _distribution(negative_percentiles),
                "control": _distribution(control_percentiles),
            },
            "by_document": enrichment_by_document,
            "top_quintile_share": {
                "label1": _top_quintile_share(positive_percentiles),
                "label0": _top_quintile_share(negative_percentiles),
                "control": _top_quintile_share(control_percentiles),
            },
        },
        "top1_candidate_provenance": top1,
        "judged_pairs": judged_pairs,
        "control_cells": control_cells,
    }


def _cell_id(code: str, paragraph_number: str, question_number: int) -> str:
    """The id of one grid cell (the file's own pair-id convention)."""
    return f"{code}-{paragraph_number}-Q{question_number}"


def _chunker_name(chunker: Chunker) -> str:
    """The chunker's config-facing name (its registry key)."""
    return getattr(chunker, "name", chunker.version)


def _top_quintile_share(percentiles: np.ndarray) -> float | None:
    """Share of a group in the top fifth of its paragraph's candidate list."""
    if not percentiles.size:
        return None
    return float((percentiles < 0.2).mean())


def _top1_provenance(
    view: DatasetView,
    similarities: np.ndarray,
    candidate_columns: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Where every paragraph's best same-document candidate comes from.

    This is the plan's "top-1 same-document candidate per paragraph" row —
    a **model-dependent** measurement: it says what *this model's* first
    choice for a paragraph is, and whether the file has a record for it. It
    is not a property of the dataset and must never be read as one.
    """
    judged_labels: dict[tuple[str, int, int], int] = {}
    for record in view.pairs:
        key = (
            record.document,
            view.paragraph_row[record.paragraph_text],
            view.question_col[record.question_text],
        )
        judged_labels[key] = record.label

    counts = {"given_positive": 0, "given_negative": 0, "unjudged": 0}
    for row, candidates in enumerate(candidate_columns):
        if not candidates:
            continue
        best = max(candidates, key=lambda col: float(similarities[row, col]))
        documents = view.paragraph_documents[row]
        label: int | None = None
        for document in sorted(documents):
            if (document, row, best) in judged_labels:
                label = judged_labels[(document, row, best)]
                break
        if label == 1:
            counts["given_positive"] += 1
        elif label == 0:
            counts["given_negative"] += 1
        else:
            counts["unjudged"] += 1
    return {
        "n_paragraphs": len(candidate_columns),
        **counts,
        "note": (
            "model-dependent measurement: this model's best same-document candidate "
            "per paragraph, intersected with the file's coverage. It is not a "
            "dataset property and says nothing about paragraphs that have no match."
        ),
    }


def build_analysis_report(
    view: DatasetView,
    models: Sequence[dict[str, Any]],
    *,
    run_id: str,
    source: str,
    source_sha256: str,
    seed: int,
    control_size: int,
    sampling_rule: str,
    question_transform: str,
    chunker_name: str,
    n_valid_pairs: int,
    n_malformed_pairs: int,
    total_seconds: float,
) -> dict[str, Any]:
    """Assemble the Stage-0 artifact (the JSON document of one run)."""
    excluded = [
        {
            "cell_id": cell["cell_id"],
            "pair_id": cell["pair_id"],
            "document": cell["document"],
            "label": cell["label"],
        }
        for cell in view.judged_cells
    ]
    return {
        "run_id": run_id,
        "command": "analyze-omct",
        "source": source,
        "source_sha256": source_sha256,
        "n_valid_pairs": n_valid_pairs,
        "n_malformed_pairs": n_malformed_pairs,
        "seed": seed,
        "analysis": {
            "control": {
                **view.control.describe(),
                "n_judged_cells_excluded": len(excluded),
                "excluded_judged_cells": excluded,
                "numpy_version": np.__version__,
            },
            "grid": {
                "n_documents": view.pooled["n_documents"],
                "n_paragraphs": view.pooled["n_paragraphs"],
                "n_questions": view.pooled["n_questions"],
                "n_cells": view.pooled["grid_cells"],
                "n_judged_cells": view.pooled["judged_cells"],
                "n_unjudged_cells": view.pooled["unjudged_cells"],
                "coverage": view.pooled["grid_coverage"],
            },
            "strata": list(view.strata),
            "pooled": view.pooled,
            "documents": list(view.documents),
            "sampling_rule": sampling_rule,
            "sampling_rule_description": SAMPLING_RULE_DESCRIPTIONS.get(
                sampling_rule, sampling_rule
            ),
            "question_transform": question_transform,
            "chunker": chunker_name,
            "notes": [
                "OMCT is auxiliary data (PLAN.md section 2): these numbers annotate "
                "method choice, they are never a UPR metric.",
                "Subset A (the reviewer-verified hard negatives) and subset B (the "
                "unjudged control) are never pooled: every number names its subset.",
                "The control is 'unjudged' (presumed-negative only where a label-shaped "
                "number is unavoidable); its positive rate is unmeasured.",
                "rank_percentile is the position from the top of the paragraph's own "
                "same-document candidate list (0 = best candidate, 1 = worst; a random "
                "candidate sits at about 0.5).",
            ],
        },
        "models": list(models),
        "total_seconds": round(total_seconds, 3),
    }
