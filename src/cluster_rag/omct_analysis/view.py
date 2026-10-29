"""The dataset view and the encoding: the substrate every arm scores on.

``DatasetView`` is the model-independent picture of the OMCT grid (the
paragraphs, the questions, the candidate columns, the control cells) and
``build_dataset_view`` computes it once per run. The encoding functions
turn one side (a paragraph or a question) into its matrix, through the
embedder the CLI injects.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder, token_capability
from cluster_rag.omct_analysis.helpers import (
    _cell_id,
    _chunker_name,
    _chunker_slug,
    document_code,
)
from cluster_rag.pairs import PairRecord
from cluster_rag.preprocess import PreprocessSpec, question_transform_version
from cluster_rag.text_processors.aggregation import Aggregation, aggregate_chunk_similarities
from cluster_rag.text_processors.base import Chunker, SpanChunker
from cluster_rag.text_processors.calibration import THRESHOLD_CRITERION_FLOOR_GAP
from cluster_rag.text_processors.control import (
    SAMPLING_RULE_PROPORTIONAL,
    ControlSample,
    DocumentGrid,
    sample_control,
)
from cluster_rag.text_processors.late_chunking import (
    LATE_POOLING_MEAN,
    LATE_POOLINGS,
    pool_chunk_vectors,
    pool_document_vector,
    pooling_report,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportingSpec:
    """The Stage-3 reporting's knobs (CS-6e).

    The plan's Stage-3 discipline asks for *"calibration curves + thresholds
    per document; pooled and per-document macro; both negative subsets
    separately"*, and CS-6e adds the **safe-negative margin** measured
    against the **control-role floor** (the track's durable deliverable,
    TEXT_PROCESSING_PLAN.md section 8).

    Attributes:
        enabled: Whether the reporting blocks are computed at all. The
            default is off, so every earlier run's artifact keeps its exact
            content.
        bins: The number of the reliability diagram's bins.
        threshold_criterion: The fit-side criterion of the per-document
            threshold (the track's two negative subsets are never pooled, so
            the criterion names which one the FPR is measured on).
        threshold_quantiles: The number of the threshold's candidate
            quantiles.
        margin_quantiles: The number of the margin rule's candidate
            quantiles.
    """

    enabled: bool = False
    bins: int = 10
    threshold_criterion: str = THRESHOLD_CRITERION_FLOOR_GAP
    threshold_quantiles: int = 99
    margin_quantiles: int = 49


def paragraph_length_report(view: DatasetView) -> dict[str, Any]:
    """The paragraph lengths: the per-document summary and the per-paragraph records.

    The lengths are the raw material of the plan's long-standing "length ↔
    control shift" question. They are **recorded**, never fed to the model:
    length is a document proxy on this data (NU's paragraphs are long, AI's
    are short), so a length feature could memorize the document instead of
    the topic.
    """
    by_document: dict[str, Any] = {}
    for grid in view.grids:
        lengths = np.asarray(
            [view.paragraph_char_lengths[row] for row in grid.paragraph_rows], dtype=np.float64
        )
        words = np.asarray(
            [view.paragraph_word_counts[row] for row in grid.paragraph_rows], dtype=np.float64
        )
        by_document[grid.document] = {
            "n_paragraphs": int(lengths.size),
            "mean_chars": float(lengths.mean()) if lengths.size else None,
            "median_chars": float(np.median(lengths)) if lengths.size else None,
            "p90_chars": float(np.quantile(lengths, 0.9)) if lengths.size else None,
            "mean_words": float(words.mean()) if words.size else None,
            "median_words": float(np.median(words)) if words.size else None,
        }
    records = [
        {
            "paragraph_number": view.paragraph_numbers[row],
            "document": sorted(view.paragraph_documents[row])[0],
            "n_chars": view.paragraph_char_lengths[row],
            "n_words": view.paragraph_word_counts[row],
        }
        for row in range(len(view.paragraphs))
    ]
    return {
        "by_document": by_document,
        "per_paragraph": records,
        "note": (
            "the paragraph lengths, recorded for the length-vs-shift question; "
            "they are never a feature (length is a document proxy on this data)"
        ),
    }


@dataclass(frozen=True)
class DatasetView:
    """The model-independent view of the pairwise dataset.

    Holds the deduplicated paragraph and question texts (the rows and columns
    of the score matrix), the per-document grids, the drawn control sample
    and the per-document strata. The model-independent part of the artifact
    is built from this object.

    The paragraph **lengths** are part of the view: the Stage-0 caveat asks
    for the correlation between a paragraph's length and a method's per-cell
    score shift, and the lengths live in the raw pair file — so the command
    records them (the lengths are *recorded*, never fed to the model: length
    is a document proxy on this data).
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
    paragraph_char_lengths: tuple[int, ...]
    paragraph_word_counts: tuple[int, ...]
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
    paragraph_char_lengths = tuple(len(text) for text in paragraphs)
    paragraph_word_counts = tuple(len(text.split()) for text in paragraphs)

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
        paragraph_char_lengths=paragraph_char_lengths,
        paragraph_word_counts=paragraph_word_counts,
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


@dataclass(frozen=True)
class SideEncoding:
    """One side's chunked encoding: the chunk lists, the vectors, the metadata.

    ``vectors`` holds one row per chunk (the chunks of a text are consecutive
    rows, in text order), so :meth:`group_rows` gives every text's rows. The
    cache counters are the ones of the embedder that produced the vectors
    (the CLI gives every side its own cache wrapper, so they are that side's).
    """

    texts: tuple[str, ...]
    chunk_lists: tuple[tuple[str, ...], ...]
    vectors: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def n_texts(self) -> int:
        """Number of texts on this side."""
        return len(self.texts)

    @property
    def n_chunks(self) -> int:
        """Number of chunks on this side (the vectors' rows)."""
        return int(self.vectors.shape[0])

    @property
    def one_chunk_per_text(self) -> bool:
        """Whether every text contributed exactly one chunk."""
        return self.n_chunks == self.n_texts

    def group_rows(self) -> tuple[tuple[int, ...], ...]:
        """The row indices of every text's chunks (in text order)."""
        groups: list[tuple[int, ...]] = []
        cursor = 0
        for chunk_list in self.chunk_lists:
            groups.append(tuple(range(cursor, cursor + len(chunk_list))))
            cursor += len(chunk_list)
        return tuple(groups)


def _cache_counters(embedder: object) -> tuple[int, int, int]:
    """The cache hit/miss/stored counters of an embedder (0 without a cache)."""
    return (
        getattr(embedder, "cache_hits", 0),
        getattr(embedder, "cache_misses", 0),
        getattr(embedder, "n_stored", 0),
    )


def encode_side(
    embedder: Embedder,
    chunker: Chunker,
    texts: Sequence[str],
    kind: str,
    *,
    meta: Mapping[str, Any] | None = None,
) -> SideEncoding:
    """Embed one side through the chunker (every chunk of every text).

    The Stage-0 baseline chunker yields one chunk per text, so its encoding
    is byte-identical to the Stage-0 run's (the warm cache stays valid); a
    chunking chunker yields many chunks per text, and the arm's aggregation
    folds their scores (see :func:`side_similarities`).

    Raises:
        ValueError: If the chunker produced no chunk for a text.
    """
    chunk_lists = chunker.segment(list(texts))
    chunks: list[str] = []
    for text, chunk_list in zip(texts, chunk_lists, strict=True):
        if not chunk_list:
            raise ValueError(f"the chunker produced no chunk for a text: {text[:60]!r}")
        chunks.extend(chunk_list)
    vectors = embedder.encode(chunks, kind)  # type: ignore[arg-type]
    hits, misses, _ = _cache_counters(embedder)
    resolved_meta = dict(meta or {})
    # The semantic chunker finds its boundaries from the atom embeddings, so
    # its atom cache counters are part of the run's provenance (read after the
    # chunking, which is what encodes the atoms).
    atom_embedder = getattr(chunker, "atom_embedder", None)
    if atom_embedder is not None:
        atom_hits, atom_misses, atom_stored = _cache_counters(atom_embedder)
        resolved_meta["atom_cache"] = {
            "hits": atom_hits,
            "misses": atom_misses,
            "stored": atom_stored,
        }
    return SideEncoding(
        texts=tuple(texts),
        chunk_lists=tuple(tuple(chunk_list) for chunk_list in chunk_lists),
        vectors=np.asarray(vectors),
        meta=resolved_meta,
        cache_hits=hits,
        cache_misses=misses,
    )


def encode_late_side(
    embedder: Embedder,
    chunker: Chunker,
    texts: Sequence[str],
    kind: str,
    *,
    poolings: Sequence[str] = (LATE_POOLING_MEAN,),
) -> dict[str, SideEncoding]:
    """Encode one side with **late chunking** (Stage 2.7).

    The whole text is embedded **once** at the token level and every chunk's
    vector is pooled out of the tokens that fall in its character span, so
    each chunk carries the document's context. The pooling rules share the
    one token pass: each rule gets its own encoding.

    The two capabilities this path needs are **probed, never assumed**:

    - the embedder must be able to return token-level embeddings (the HF
      backend can; Ollama's ``/api/embed`` cannot);
    - the chunker must be able to say where its chunks are (the word windows
      do, exactly; a chunker whose chunks are not substrings of the text
      does not).

    Args:
        embedder: The embedder (a port: the CLI wires the backend in).
        chunker: The chunker whose spans the tokens are pooled over.
        texts: The texts of the side.
        kind: The encode kind (``"document"`` or ``"query"``).
        poolings: The pooling rules (each one an encoding of its own).

    Returns:
        One :class:`SideEncoding` per pooling rule, keyed by the rule.

    Raises:
        ValueError: If the embedder has no token-level capability, the
            chunker has no character spans, or a chunk has no token.
    """
    capability = token_capability(embedder)
    if capability is None:
        raise ValueError(
            "the late-chunked arm needs token-level embeddings; this backend "
            "cannot return them (Ollama's /api/embed returns one vector per "
            "text and nothing else)"
        )
    if not isinstance(chunker, SpanChunker):
        raise ValueError(
            f"the late-chunked arm needs the chunker's character spans; the "
            f"chunker {getattr(chunker, 'name', chunker)!r} does not provide them"
        )
    unknown = [rule for rule in poolings if rule not in LATE_POOLINGS]
    if unknown:
        raise ValueError(f"Unknown late pooling rule(s) {unknown}; expected {list(LATE_POOLINGS)}")

    text_list = list(texts)
    token_embeddings = capability.encode_tokens(text_list, kind)  # type: ignore[arg-type]
    span_lists = chunker.segment_spans(text_list)

    chunk_lists: list[tuple[str, ...]] = []
    pooled_by_rule: dict[str, list[np.ndarray]] = {rule: [] for rule in poolings}
    reports: list[dict[str, Any]] = []
    for text, tokens, spans in zip(text_list, token_embeddings, span_lists, strict=True):
        chunk_lists.append(tuple(text[start:end] for start, end in spans))
        for rule in poolings:
            pooled_by_rule[rule].append(pool_chunk_vectors(tokens, spans, pooling=rule))
        reports.append(pooling_report(tokens, spans))

    # The consistency check: the document's own pooled vector against the
    # model's own embedding of the same text — the *chunks' view* (the
    # content tokens only; no chunk can own the prompt prefix's tokens) and
    # the *model's formula* (the content tokens plus the prefix's own, which
    # is what the model itself averages).
    consistency: dict[str, Any] = {"cosine_chunks_view": None, "cosine_model_formula": None}
    if text_list:
        own = np.asarray(embedder.encode([text_list[0]], kind))[0]
        unit_own = own / max(float(np.linalg.norm(own)), 1e-12)
        first = token_embeddings[0]
        consistency["cosine_chunks_view"] = float(np.dot(pool_document_vector(first), unit_own))
        consistency["cosine_model_formula"] = float(
            np.dot(pool_document_vector(first, prefix_tokens=first.prefix_tokens), unit_own)
        )

    coverages = [report["token_coverage"] for report in reports]
    encodings: dict[str, SideEncoding] = {}
    for rule in poolings:
        matrix = (
            np.concatenate(pooled_by_rule[rule], axis=0)
            if pooled_by_rule[rule]
            else np.zeros((0, 0), dtype=np.float32)
        )
        encodings[rule] = SideEncoding(
            texts=tuple(text_list),
            chunk_lists=tuple(chunk_lists),
            vectors=matrix,
            meta={
                "kind": kind,
                "late_chunking": True,
                "pooling": rule,
                "chunker": chunker.version,
                "chunker_name": _chunker_name(chunker),
                "chunker_slug": _chunker_slug(chunker),
                "side_slug": f"late_{rule}_{_chunker_slug(chunker)}",
                "token_level": True,
                "cache": (
                    "the token-level pass is not cached: the embedding cache keys "
                    "on the chunk's text, and a late-chunked vector is a function of "
                    "the whole document, not of one chunk"
                ),
                "token_coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
                "token_coverage_min": float(np.min(coverages)) if coverages else 0.0,
                "n_content_tokens": int(sum(report["n_content_tokens"] for report in reports)),
                "pooling_consistency": {
                    "note": (
                        "the cosine between the document's pooled token vector and "
                        "the model's own embedding of the same text: the 'chunks' "
                        "view' pools the content tokens only (no chunk can own the "
                        "prompt prefix's tokens), the 'model formula' pools them "
                        "with the prefix's own (what the model itself averages) and "
                        "is the exact check of the token-level path"
                    ),
                    **consistency,
                },
            },
            cache_hits=0,
            cache_misses=0,
        )
    return encodings


def fold_question_chunks(
    matrix: np.ndarray, question_groups: Sequence[Sequence[int]]
) -> np.ndarray:
    """Fold a chunked question side: every question keeps its best chunk.

    The symmetric variants of Stage 2 (exp05/exp10) chunk **both** sides. The
    question side is folded with the maximum — the best matching chunk of a
    question is the question's evidence — and the paragraph side is then
    folded with the arm's own rule.

    Args:
        matrix: The ``(n_paragraph_chunks, n_question_chunks)`` similarity matrix.
        question_groups: The chunk rows of every question (in text order).

    Returns:
        The ``(n_paragraph_chunks, n_questions)`` matrix of folded scores.

    Raises:
        ValueError: If a question has no chunk.
    """
    starts: list[int] = []
    for position, group in enumerate(question_groups):
        indices = list(group)
        if not indices:
            raise ValueError(f"question {position} has no chunk")
        starts.append(indices[0])
    folded = np.maximum.reduceat(matrix, np.asarray(starts, dtype=np.int64), axis=1)
    return folded


def side_similarities(
    paragraph_side: SideEncoding,
    question_side: SideEncoding,
    rule: Aggregation | None,
    *,
    global_similarities: np.ndarray | None = None,
) -> np.ndarray:
    """The (paragraphs x questions) score matrix of one arm.

    The question side may be chunked (the Stage-2 symmetric variants): its
    chunks are folded with the maximum first, and the paragraph side is then
    folded with the arm's rule. The baseline arm (``rule=None``) is the
    full-text scoring and needs one chunk per text on both sides.

    Args:
        paragraph_side: The encoded paragraph side (possibly chunked).
        question_side: The encoded question side (possibly chunked).
        rule: The aggregation rule that folds the paragraph chunks (None = the
            full-text baseline, which needs one chunk per paragraph).
        global_similarities: The full-text score matrix (one row per
            paragraph) — the second input of the local+global blend
            (``exp06``).

    Returns:
        The ``(n_paragraphs, n_questions)`` score matrix.

    Raises:
        ValueError: If the baseline arm's sides are chunked, or the rule needs
            the full-text score and none was given.
    """
    matrix = paragraph_side.vectors @ question_side.vectors.T
    if rule is None:
        if not question_side.one_chunk_per_text:
            raise ValueError("the baseline arm needs one chunk per question")
        if not paragraph_side.one_chunk_per_text:
            raise ValueError("an arm without an aggregation rule needs one chunk per paragraph")
        return matrix
    if not question_side.one_chunk_per_text:
        matrix = fold_question_chunks(matrix, question_side.group_rows())
    return aggregate_chunk_similarities(
        matrix,
        paragraph_side.group_rows(),
        rule,
        global_similarities=global_similarities,
    )


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
