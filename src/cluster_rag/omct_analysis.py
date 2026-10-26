"""The text-processing track's analysis driver (``analyze-omct``).

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

Stage 1.1a adds the **arms**: one run scores the *baseline arm* (full text
against full text — the Stage-0 scoring) **and** the structural arms
(paragraph atoms folded into one pair score by an aggregation rule) **and**
the question-transform arms (the free ``keep`` vs ``strip_block``
comparison), so every artifact holds the comparison and the pairing is by
construction. Each arm is compared with the baseline arm by a **paired**
statistic (the paired bootstrap of the AUC difference) and by the
**control-side counterpart** (the control's score level and rank position on
the same 4,000 cells) — the track's reporting rule: no discrimination number
is quoted without the control side next to it, and the de-shifted AUC (the
control's per-document mean shift subtracted from the method's reviewed-set
scores) says whether a discrimination gain survives the scale shift.

The dataset-level part (the grid, the strata, the control draw) is
model-independent and computed once per run; the scores are per model.

Like :mod:`cluster_rag.omct_probe` (the Phase-2 probe driver), this module
orchestrates the embedder (a port, injected by the CLI), the pure metrics and
the track's pure core (:mod:`cluster_rag.text_processors.control`,
:mod:`cluster_rag.text_processors.aggregation`). It never imports a model
client itself: the CLI wires the backends.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder
from cluster_rag.eval.metrics import (
    grouped_auc_decomposition,
    paired_auc_bootstrap,
    paired_within_group_auc_bootstrap,
    roc_auc,
)
from cluster_rag.pairs import PairRecord
from cluster_rag.preprocess import (
    PreprocessSpec,
    apply_question_transform,
    question_transform_version,
)
from cluster_rag.text_processors.aggregation import (
    Aggregation,
    aggregate_chunk_similarities,
    describe_aggregation,
)
from cluster_rag.text_processors.base import Chunker, Combiner, FeatureExtractor, TextPair
from cluster_rag.text_processors.control import (
    SAMPLING_RULE_DESCRIPTIONS,
    SAMPLING_RULE_PROPORTIONAL,
    ControlSample,
    DocumentGrid,
    percentiles_from_positions,
    rank_positions,
    sample_control,
)

logger = logging.getLogger(__name__)

# The quantile grid of the reported distributions (mirrors the probe's).
DISTRIBUTION_QUANTILES: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

# The hit@k grid of the per-pair rank report (the "is the pair's question
# among the paragraph's top-k candidates" measure). The same grid the probe
# reports.
HIT_KS: tuple[int, ...] = (1, 10)

# The name of the baseline arm: the full-text scoring every other arm is
# compared with.
BASELINE_ARM = "full_text"

# The chunker of the baseline arm (the track's identity chunker).
BASELINE_CHUNKER = "full_text"

# The description of the baseline arm's scoring, recorded in the artifact.
FULL_TEXT_AGGREGATION = "one chunk per text (the plain full-text cosine)"

# The fit-side criteria of a swept parameter (the semantic threshold of
# Stage 2.3): the reviewed set's AUC or its floor gap.
SEMANTIC_CRITERION_AUC = "auc"
SEMANTIC_CRITERION_FLOOR_GAP = "floor_gap"
SEMANTIC_CRITERIA: tuple[str, ...] = (SEMANTIC_CRITERION_AUC, SEMANTIC_CRITERION_FLOOR_GAP)

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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """The average ranks of the values (1-based; ties share the mean rank)."""
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or ordered[index] != ordered[start]:
            ranks[order[start:index]] = (start + index - 1) / 2.0 + 1.0
            start = index
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    """The Pearson correlation of two vectors (``None`` when undefined)."""
    if left.size < 2 or left.size != right.size:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(float((left_centered**2).sum()) * float((right_centered**2).sum())))
    if denominator <= 0.0:
        return None
    return float(float((left_centered * right_centered).sum()) / denominator)


def length_correlation(
    lengths: np.ndarray, scores: np.ndarray, groups: np.ndarray | None = None
) -> dict[str, Any]:
    """The correlation between a paragraph's length and a cell's score.

    The Stage-0 caveat asks for it: a chunked method takes a maximum over
    more candidates, so its scale shift is partly a **length** effect and
    partly a **document** effect — and a single global shift is only an
    approximation. The correlation says how much of a cell's score travels
    with its paragraph's length (the Pearson on the raw values and the
    Spearman on the ranks), overall and per group (the document).

    Args:
        lengths: The paragraph lengths (in characters) of the cells.
        scores: The cells' scores.
        groups: The cells' groups (the document); ``None`` skips the
            per-group breakdown.

    Returns:
        A JSON-serializable dict with the correlations and the counts.
    """
    report: dict[str, Any] = {
        "overall": {
            "n": int(lengths.size),
            "pearson": _pearson(lengths, scores),
            "spearman": _pearson(_average_ranks(lengths), _average_ranks(scores)),
        },
        "note": (
            "the correlation between a cell's paragraph length and the cell's "
            "score: it is how much of a method's scale shift travels with the "
            "paragraph's length (the Stage-0 caveat's length effect)"
        ),
    }
    if groups is not None:
        by_group: dict[str, Any] = {}
        for group in sorted(set(groups.tolist())):
            mask = groups == group
            by_group[group] = {
                "n": int(mask.sum()),
                "pearson": _pearson(lengths[mask], scores[mask]),
                "spearman": _pearson(_average_ranks(lengths[mask]), _average_ranks(scores[mask])),
            }
        report["by_group"] = by_group
    return report


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


@dataclass(frozen=True)
class ArmScores:
    """The measured scores of one arm: the matrix and the per-pair records."""

    similarities: np.ndarray
    labels: np.ndarray
    pair_scores: np.ndarray
    control_scores: np.ndarray
    pair_ranks: np.ndarray
    control_ranks: np.ndarray
    pair_percentiles: np.ndarray
    control_percentiles: np.ndarray
    documents: np.ndarray
    control_documents: np.ndarray
    rows: np.ndarray
    columns: np.ndarray
    control_rows: np.ndarray
    control_columns: np.ndarray


def measure_arm(view: DatasetView, similarities: np.ndarray) -> ArmScores:
    """Measure one arm's scores: the judged pairs, the control cells, the ranks."""
    candidate_columns = view.candidate_columns()
    rows = np.asarray(
        [view.paragraph_row[record.paragraph_text] for record in view.pairs], dtype=np.int64
    )
    columns = np.asarray(
        [view.question_col[record.question_text] for record in view.pairs], dtype=np.int64
    )
    labels = np.asarray([record.label for record in view.pairs], dtype=np.int64)
    control_rows = np.asarray([row for _, row, _ in view.control.cells], dtype=np.int64)
    control_columns = np.asarray([col for _, _, col in view.control.cells], dtype=np.int64)

    pair_ranks = rank_positions(similarities, rows, columns, candidate_columns)
    control_ranks = rank_positions(similarities, control_rows, control_columns, candidate_columns)
    return ArmScores(
        similarities=similarities,
        labels=labels,
        pair_scores=similarities[rows, columns],
        control_scores=similarities[control_rows, control_columns],
        pair_ranks=pair_ranks,
        control_ranks=control_ranks,
        pair_percentiles=percentiles_from_positions(pair_ranks, rows, candidate_columns),
        control_percentiles=percentiles_from_positions(
            control_ranks, control_rows, candidate_columns
        ),
        documents=np.asarray([record.document for record in view.pairs]),
        control_documents=np.asarray([document for document, _, _ in view.control.cells]),
        rows=rows,
        columns=columns,
        control_rows=control_rows,
        control_columns=control_columns,
    )


def _hit_at_k(ranks: np.ndarray, hit_ks: Sequence[int]) -> dict[str, float]:
    """The share of pairs whose question is among the paragraph's top-k candidates."""
    return {f"hit@{k}": float((ranks < k).mean()) if ranks.size else 0.0 for k in hit_ks}


def score_arm(
    view: DatasetView,
    *,
    arm_name: str,
    similarities: np.ndarray,
    paragraph_side: SideEncoding,
    question_side: SideEncoding,
    aggregation: Aggregation | None = None,
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """Score one arm: the pairs, the control cells, the distributions, the ranks.

    The arm's block is the same shape for every arm (the baseline arm and the
    structural arms), so the comparisons are made on identical fields.
    """
    measured = measure_arm(view, similarities)
    labels = measured.labels

    codes = {report["document"]: report["code"] for report in view.documents}
    judged_pairs: list[dict[str, Any]] = []
    for position, cell in enumerate(view.judged_cells):
        judged_pairs.append(
            {
                **cell,
                "score": float(measured.pair_scores[position]),
                "rank_percentile": float(measured.pair_percentiles[position]),
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
                "score": float(measured.control_scores[position]),
                "rank_percentile": float(measured.control_percentiles[position]),
            }
        )

    positive_scores = measured.pair_scores[labels == 1]
    negative_scores = measured.pair_scores[labels == 0]
    positive_percentiles = measured.pair_percentiles[labels == 1]
    negative_percentiles = measured.pair_percentiles[labels == 0]

    score_distributions: dict[str, Any] = {
        "label1": _distribution(positive_scores),
        "label0": _distribution(negative_scores),
        "control": _distribution(measured.control_scores),
        "by_document": {},
    }
    enrichment_by_document: dict[str, Any] = {}
    for stratum in view.strata:
        document = stratum["document"]
        mask = measured.documents == document
        control_mask = measured.control_documents == document
        document_control_scores = measured.control_scores[control_mask]
        document_control_percentiles = measured.control_percentiles[control_mask]
        score_distributions["by_document"][document] = {
            "label1": _distribution(measured.pair_scores[mask & (labels == 1)]),
            "label0": _distribution(measured.pair_scores[mask & (labels == 0)]),
            "control": _distribution(document_control_scores),
        }
        enrichment_by_document[document] = {
            "rank_percentile": {
                "label1": _distribution(measured.pair_percentiles[mask & (labels == 1)]),
                "label0": _distribution(measured.pair_percentiles[mask & (labels == 0)]),
                "control": _distribution(document_control_percentiles),
            },
            "top_quintile_share": {
                "label1": _top_quintile_share(measured.pair_percentiles[mask & (labels == 1)]),
                "label0": _top_quintile_share(measured.pair_percentiles[mask & (labels == 0)]),
                "control": _top_quintile_share(document_control_percentiles),
            },
        }

    pair_auc: dict[str, Any] = {"overall": None, "by_document": {}}
    try:
        pair_auc["overall"] = roc_auc(labels.tolist(), measured.pair_scores.tolist())
    except ValueError as exc:  # a document without both classes
        pair_auc["overall_note"] = str(exc)
    for stratum in view.strata:
        document = stratum["document"]
        mask = measured.documents == document
        try:
            pair_auc["by_document"][document] = roc_auc(
                labels[mask].tolist(), measured.pair_scores[mask].tolist()
            )
        except ValueError as exc:
            pair_auc["by_document"][document] = None
            pair_auc.setdefault("notes", {})[document] = str(exc)
    document_aucs = [value for value in pair_auc["by_document"].values() if value is not None]
    pair_auc["macro"] = float(np.mean(document_aucs)) if document_aucs else None

    hit_at_k = {
        "label1": _hit_at_k(measured.pair_ranks[labels == 1], hit_ks),
        "label0": _hit_at_k(measured.pair_ranks[labels == 0], hit_ks),
        "control": _hit_at_k(measured.control_ranks, hit_ks),
        "note": (
            "the share of the group's pairs whose question is among the top-k "
            "candidates of its own paragraph (the negatives' hit@1 rising means "
            "the arm lifts the hard negatives as well)"
        ),
    }

    top1 = _top1_provenance(view, similarities, view.candidate_columns())

    paragraph_meta = {**paragraph_side.meta}
    if aggregation is not None:
        paragraph_meta["aggregation"] = (
            f"the paragraph's {paragraph_side.n_chunks} chunks are folded into one "
            f"pair score by '{aggregation.spec}'"
        )

    return {
        "arm": arm_name,
        "aggregation": None if aggregation is None else dict(describe_aggregation(aggregation)),
        "pair_auc_decomposition": _auc_decomposition(
            labels, measured.pair_scores, measured.documents
        ),
        "sides": {
            "paragraph": {
                **paragraph_meta,
                "n_texts": paragraph_side.n_texts,
                "n_chunks": paragraph_side.n_chunks,
                "cache_hits": paragraph_side.cache_hits,
                "cache_misses": paragraph_side.cache_misses,
            },
            "question": {
                **question_side.meta,
                "n_texts": question_side.n_texts,
                "n_chunks": question_side.n_chunks,
                "cache_hits": question_side.cache_hits,
                "cache_misses": question_side.cache_misses,
            },
        },
        "pair_auc": pair_auc,
        "score_distributions": score_distributions,
        "negative_enrichment": {
            "rank_percentile": {
                "label1": _distribution(positive_percentiles),
                "label0": _distribution(negative_percentiles),
                "control": _distribution(measured.control_percentiles),
            },
            "by_document": enrichment_by_document,
            "top_quintile_share": {
                "label1": _top_quintile_share(positive_percentiles),
                "label0": _top_quintile_share(negative_percentiles),
                "control": _top_quintile_share(measured.control_percentiles),
            },
        },
        "hit_at_k": hit_at_k,
        "length_correlation": length_correlation(
            np.asarray(
                [view.paragraph_char_lengths[row] for _, row, _ in view.control.cells],
                dtype=np.float64,
            ),
            measured.control_scores,
            measured.control_documents,
        ),
        "top1_candidate_provenance": top1,
        "judged_pairs": judged_pairs,
        "control_cells": control_cells,
    }


def _paragraph_meta(preprocess_spec: PreprocessSpec, chunker: Chunker) -> dict[str, Any]:
    """The artifact's paragraph-side provenance (the chunker and the preprocess)."""
    return {
        "kind": "document",
        "preprocess": preprocess_spec.variant,
        "preprocess_version": preprocess_spec.version,
        "chunker": chunker.version,
        "chunker_name": _chunker_name(chunker),
        "chunker_slug": _chunker_slug(chunker),
        "cache_version": chunker.cache_version(preprocess_spec.version),
        "aggregation": FULL_TEXT_AGGREGATION,
    }


def _question_meta(
    question_transform: str,
    question_block_markers: Sequence[str],
    chunker: Chunker,
) -> dict[str, Any]:
    """The artifact's question-side provenance (the transform and the chunker)."""
    version = question_transform_version(question_transform, question_block_markers)
    return {
        "kind": "query",
        "transform": question_transform,
        "transform_version": version,
        "chunker": chunker.version,
        "chunker_name": _chunker_name(chunker),
        "cache_version": chunker.cache_version(version),
        "aggregation": FULL_TEXT_AGGREGATION,
    }


def compare_arms(
    view: DatasetView,
    *,
    arm_name: str,
    reference_similarities: np.ndarray,
    arm_similarities: np.ndarray,
    resamples: int,
    seed: int,
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """Compare one arm with the baseline arm (the paired statistic + the control side).

    The comparison carries three things the track's reporting rule demands:

    - the **paired** difference of the reviewed set's AUC (the same 1,520
      pairs, the same labels, a paired bootstrap over them);
    - the **control-side counterpart**: how much the control's score level and
      rank position moved (overall and per document) — a method that only
      re-scales shows up here;
    - the **de-shifted AUC**: the control's *per-document* mean shift
      subtracted from the method's reviewed-set scores, so a discrimination
      gain can be checked against the scale shift (the plan's rule: a method
      counts as a win only if the discrimination gain survives the scale
      shift).
    """
    reference = measure_arm(view, reference_similarities)
    arm = measure_arm(view, arm_similarities)
    labels = arm.labels

    paired = paired_auc_bootstrap(
        labels.tolist(),
        reference.pair_scores.tolist(),
        arm.pair_scores.tolist(),
        n_resamples=resamples,
        seed=seed,
    )

    per_document: dict[str, Any] = {}
    shifts: dict[str, float] = {}
    deshifted_pair_scores = np.empty_like(arm.pair_scores)
    for stratum in view.strata:
        document = stratum["document"]
        pair_mask = arm.documents == document
        control_mask = arm.control_documents == document
        reference_control = reference.control_scores[control_mask]
        arm_control = arm.control_scores[control_mask]
        shift = float(arm_control.mean() - reference_control.mean()) if arm_control.size else 0.0
        shifts[document] = shift
        deshifted_pair_scores[pair_mask] = arm.pair_scores[pair_mask] - shift

        reference_auc = _safe_auc(labels[pair_mask], reference.pair_scores[pair_mask])
        arm_auc = _safe_auc(labels[pair_mask], arm.pair_scores[pair_mask])
        deshifted_auc = _safe_auc(labels[pair_mask], deshifted_pair_scores[pair_mask])
        reference_positive = reference.pair_scores[pair_mask & (labels == 1)]
        arm_positive = arm.pair_scores[pair_mask & (labels == 1)]
        per_document[document] = {
            "n_pairs": int(pair_mask.sum()),
            "control_shift": {
                "mean_delta": shift,
                "p50_delta": _quantile_delta(arm_control, reference_control),
                "rank_p50_delta": _quantile_delta(
                    arm.control_percentiles[control_mask],
                    reference.control_percentiles[control_mask],
                ),
            },
            "positives_shift": {
                "mean_delta": (
                    float(arm_positive.mean() - reference_positive.mean())
                    if arm_positive.size
                    else None
                ),
            },
            "floor_gap": {
                "reference": _floor_gap(reference_positive, reference_control),
                "arm": _floor_gap(arm_positive, arm_control),
            },
            "pair_auc": {
                "reference": reference_auc,
                "arm": arm_auc,
                "delta": (
                    arm_auc - reference_auc
                    if arm_auc is not None and reference_auc is not None
                    else None
                ),
            },
            "deshifted_auc": deshifted_auc,
        }

    reference_positive = reference.pair_scores[labels == 1]
    arm_positive = arm.pair_scores[labels == 1]
    reference_control_all = reference.control_scores
    arm_control_all = arm.control_scores
    deshifted_overall = _safe_auc(labels, deshifted_pair_scores)

    reference_auc_by_document = {
        document: block["pair_auc"]["reference"] for document, block in per_document.items()
    }
    arm_auc_by_document = {
        document: block["pair_auc"]["arm"] for document, block in per_document.items()
    }
    reference_macro = _macro(reference_auc_by_document)
    arm_macro = _macro(arm_auc_by_document)

    reference_decomposition = _auc_decomposition(labels, reference.pair_scores, reference.documents)
    arm_decomposition = _auc_decomposition(labels, arm.pair_scores, arm.documents)
    within_reference = reference_decomposition.get("within_group_auc")
    within_arm = arm_decomposition.get("within_group_auc")
    cross_reference = reference_decomposition.get("cross_group_auc")
    cross_arm = arm_decomposition.get("cross_group_auc")
    within_paired = paired_within_group_auc_bootstrap(
        labels.tolist(),
        reference.pair_scores.tolist(),
        arm.pair_scores.tolist(),
        arm.documents.tolist(),
        n_resamples=resamples,
        seed=seed,
    )

    return {
        "arm": arm_name,
        "reference_arm": BASELINE_ARM,
        "paired_auc": paired,
        "pair_auc": {
            "reference": paired["auc_reference"],
            "arm": paired["auc_arm"],
            "delta": paired["delta_auc"],
            "macro": {
                "reference": reference_macro,
                "arm": arm_macro,
                "delta": (
                    arm_macro - reference_macro
                    if arm_macro is not None and reference_macro is not None
                    else None
                ),
            },
            "by_document": {
                document: {
                    "reference": reference_auc_by_document[document],
                    "arm": arm_auc_by_document[document],
                    "delta": per_document[document]["pair_auc"]["delta"],
                }
                for document in per_document
            },
        },
        "auc_decomposition": {
            "reference": reference_decomposition,
            "arm": arm_decomposition,
            "within_group_delta": (
                within_arm - within_reference
                if within_arm is not None and within_reference is not None
                else None
            ),
            "cross_group_delta": (
                cross_arm - cross_reference
                if cross_arm is not None and cross_reference is not None
                else None
            ),
            "within_group_paired": within_paired,
            "note": (
                "the pooled AUC mixes the ranking inside a document with the "
                "ranking across documents (58.7% of the positive-negative "
                "pairs are cross-document): the within-document delta is the "
                "part that transfers to a corpus without document structure"
            ),
        },
        "control_shift": {
            "mean_delta": float(arm_control_all.mean() - reference_control_all.mean()),
            "p50_delta": _quantile_delta(arm_control_all, reference_control_all),
            "rank_p50_delta": _quantile_delta(
                arm.control_percentiles, reference.control_percentiles
            ),
            "top_fifth_share_delta": _top_fifth_share_delta(
                arm.control_percentiles, reference.control_percentiles
            ),
            "by_document": {
                document: block["control_shift"] for document, block in per_document.items()
            },
        },
        "positives_shift": {
            "mean_delta": float(arm_positive.mean() - reference_positive.mean()),
            "rank_p50_delta": _quantile_delta(
                arm.pair_percentiles[labels == 1], reference.pair_percentiles[labels == 1]
            ),
            "by_document": {
                document: block["positives_shift"] for document, block in per_document.items()
            },
        },
        "floor_gap": {
            "reference": _floor_gap(reference_positive, reference_control_all),
            "arm": _floor_gap(arm_positive, arm_control_all),
            "delta": (
                _floor_gap(arm_positive, arm_control_all)
                - _floor_gap(reference_positive, reference_control_all)
            ),
            "note": (
                "the gap between the verified positives' mean score and the "
                "control's mean score: a positive delta means the ordinary "
                "same-document pairs fell relative to the true matches"
            ),
        },
        "deshifted_auc": {
            "overall": deshifted_overall,
            "macro": _macro(
                {document: block["deshifted_auc"] for document, block in per_document.items()}
            ),
            "by_document": {
                document: block["deshifted_auc"] for document, block in per_document.items()
            },
            "per_document_shift": shifts,
            "note": (
                "the control's per-document mean shift subtracted from the "
                "method's reviewed-set scores (the plan's rule: a discrimination "
                "gain counts only if it survives the scale shift)"
            ),
        },
        "hit_at_k": {
            "reference": {
                "label1": _hit_at_k(reference.pair_ranks[labels == 1], hit_ks),
                "label0": _hit_at_k(reference.pair_ranks[labels == 0], hit_ks),
                "control": _hit_at_k(reference.control_ranks, hit_ks),
            },
            "arm": {
                "label1": _hit_at_k(arm.pair_ranks[labels == 1], hit_ks),
                "label0": _hit_at_k(arm.pair_ranks[labels == 0], hit_ks),
                "control": _hit_at_k(arm.control_ranks, hit_ks),
            },
        },
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """The AUC of a group (None when a class is missing)."""
    try:
        return roc_auc(labels.tolist(), scores.tolist())
    except ValueError:
        return None


def _mean_of(values: Sequence[float | None]) -> float | None:
    """The mean of the usable values (None when there is none)."""
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _difference(left: float | None, right: float | None) -> float | None:
    """The difference of two optional numbers (None when either is missing)."""
    if left is None or right is None:
        return None
    return float(left - right)


def _weighted_mean(
    values: Sequence[tuple[str, float | None]], weights: Mapping[str, int]
) -> float | None:
    """The weighted mean of per-group values (None when no group has a value).

    The weight of a group is its number of positive-negative pairs, so the
    result is the same statistic as the within-group AUC (which averages the
    groups' AUCs with that weight).
    """
    weighted = 0.0
    total = 0
    for group, value in values:
        if value is None:
            continue
        weight = weights.get(group, 0)
        weighted += value * weight
        total += weight
    return float(weighted / total) if total else None


def _auc_decomposition(
    labels: np.ndarray, scores: np.ndarray, documents: np.ndarray
) -> dict[str, Any]:
    """The within-document / cross-document split of a pooled AUC.

    The pooled AUC of the reviewed set compares every positive with every
    negative, and 58.7% of those pairs are cross-document. A method that
    changes a document's score *level* therefore moves the pooled number
    without changing the ranking inside a document — so the two parts are
    reported separately (the track's per-document discipline).
    """
    try:
        return grouped_auc_decomposition(labels.tolist(), scores.tolist(), documents.tolist())
    except ValueError as exc:
        return {"note": str(exc)}


def _macro(values: Mapping[str, float | None]) -> float | None:
    """The macro average of the per-document values (None when none exists)."""
    present = [value for value in values.values() if value is not None]
    return float(np.mean(present)) if present else None


def _quantile_delta(arm: np.ndarray, reference: np.ndarray) -> float | None:
    """The difference of the medians of two groups (None for an empty group)."""
    if not arm.size or not reference.size:
        return None
    return float(np.quantile(arm, 0.5) - np.quantile(reference, 0.5))


def _floor_gap(positive_scores: np.ndarray, control_scores: np.ndarray) -> float | None:
    """The gap between the verified positives' mean and the control's mean."""
    if not positive_scores.size or not control_scores.size:
        return None
    return float(positive_scores.mean() - control_scores.mean())


def _top_fifth_share_delta(
    arm_percentiles: np.ndarray, reference_percentiles: np.ndarray
) -> float | None:
    """The difference of the top-fifth shares of two rank-percentile groups."""
    if not arm_percentiles.size or not reference_percentiles.size:
        return None
    return _top_quintile_share(arm_percentiles) - _top_quintile_share(reference_percentiles)


@dataclass(frozen=True)
class ChunkedSide:
    """One chunked side of a run: the chunker, its embedder and its question side.

    The driver never builds a chunker itself: the CLI wires the (cached)
    embedders in and hands the driver a ready-made side. That keeps the model
    clients at the edge (AGENTS.md dependency direction) and lets the same
    side describe a fixed chunker and one point of a swept grid.

    Attributes:
        chunker: The paragraph side's chunker.
        paragraph_embedder: The embedder of the paragraph chunks.
        question_chunker: The question side's chunker (``None`` = the question
            side stays whole: the one-sided variant).
        question_embedder: The embedder of the question chunks.
        threshold: The swept parameter this side was built with (the semantic
            threshold; ``None`` for a fixed chunker).
    """

    chunker: Chunker
    paragraph_embedder: Embedder
    question_chunker: Chunker | None = None
    question_embedder: Embedder | None = None
    threshold: float | None = None


def _chunker_slug(chunker: Chunker) -> str:
    """The chunker's short handle (its arm-name prefix)."""
    slug = getattr(chunker, "slug", None)
    return slug if isinstance(slug, str) and slug else _chunker_name(chunker)


def _chunker_name(chunker: Chunker) -> str:
    """The chunker's config-facing name (its registry key)."""
    return getattr(chunker, "name", chunker.version)


def _variant_question_side(
    side: ChunkedSide,
    baseline_question_side: SideEncoding,
    question_texts: Sequence[str],
    question_transform: str,
    question_block_markers: Sequence[str],
) -> SideEncoding:
    """The question side of a chunked variant (the whole one, or the chunked)."""
    if side.question_chunker is None:
        return baseline_question_side
    if side.question_embedder is None:
        raise ValueError("a chunked question side needs its own embedder")
    return encode_side(
        side.question_embedder,
        side.question_chunker,
        list(question_texts),
        "query",
        meta=_question_meta(question_transform, question_block_markers, side.question_chunker),
    )


def _fixed_variant_arms(
    view: DatasetView,
    *,
    side: ChunkedSide,
    aggregations: Sequence[Aggregation],
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    baseline_question_side: SideEncoding,
    baseline_similarities: np.ndarray,
    question_transform: str,
    question_block_markers: Sequence[str],
    preprocess_spec: PreprocessSpec,
    hit_ks: Sequence[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """The arms of one fixed chunker variant (one arm per aggregation rule)."""
    paragraph_side = encode_side(
        side.paragraph_embedder,
        side.chunker,
        list(paragraph_texts),
        "document",
        meta=_paragraph_meta(preprocess_spec, side.chunker),
    )
    question_side = _variant_question_side(
        side, baseline_question_side, question_texts, question_transform, question_block_markers
    )
    arms: dict[str, Any] = {}
    similarities_by_arm: dict[str, np.ndarray] = {}
    for rule in aggregations:
        arm_name = f"{_chunker_slug(side.chunker)}_{rule.slug}"
        similarities = side_similarities(
            paragraph_side,
            question_side,
            rule,
            global_similarities=baseline_similarities,
        )
        arms[arm_name] = score_arm(
            view,
            arm_name=arm_name,
            similarities=similarities,
            paragraph_side=paragraph_side,
            question_side=question_side,
            aggregation=rule,
            hit_ks=hit_ks,
        )
        similarities_by_arm[arm_name] = similarities
    return arms, similarities_by_arm


def _pair_indices(view: DatasetView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The row, column and label of every judged pair."""
    rows = np.asarray(
        [view.paragraph_row[record.paragraph_text] for record in view.pairs], dtype=np.int64
    )
    columns = np.asarray(
        [view.question_col[record.question_text] for record in view.pairs], dtype=np.int64
    )
    labels = np.asarray([record.label for record in view.pairs], dtype=np.int64)
    return rows, columns, labels


def _control_indices(view: DatasetView) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The row, column and document of every control cell."""
    rows = np.asarray([row for _, row, _ in view.control.cells], dtype=np.int64)
    columns = np.asarray([column for _, _, column in view.control.cells], dtype=np.int64)
    documents = np.asarray([document for document, _, _ in view.control.cells])
    return rows, columns, documents


def _row_documents(view: DatasetView) -> tuple[str, ...]:
    """The document of every paragraph row (a row belongs to one document)."""
    return tuple(min(documents) for documents in view.paragraph_documents)


def _side_numbers(
    labels: np.ndarray,
    pair_scores: np.ndarray,
    pair_mask: np.ndarray,
    control_scores: np.ndarray,
    control_mask: np.ndarray,
) -> dict[str, Any]:
    """The numbers of one side of a document (the reviewed set + the control)."""
    positives = pair_mask & (labels == 1)
    control = control_scores[control_mask]
    return {
        "n_pairs": int(pair_mask.sum()),
        "n_positive": int(positives.sum()),
        "auc": _safe_auc(labels[pair_mask], pair_scores[pair_mask]),
        "positives_mean": float(pair_scores[positives].mean()) if positives.any() else None,
        "control_mean": float(control.mean()) if control.size else None,
        "floor_gap": _floor_gap(pair_scores[positives], control),
    }


def _argmax_threshold(values: Mapping[float, float | None], thresholds: Sequence[float]) -> float:
    """The threshold with the largest criterion value (the first one on a tie).

    Raises:
        ValueError: If no threshold has a usable criterion value.
    """
    best: float | None = None
    best_value = -math.inf
    for threshold in thresholds:
        value = values.get(threshold)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best = threshold
    if best is None:
        raise ValueError(
            "the swept parameter could not be selected: the fit side has no usable criterion value"
        )
    return best


def _cross_fit(
    view: DatasetView,
    matrices: Mapping[float, Mapping[str, np.ndarray]],
    thresholds: Sequence[float],
    aggregations: Sequence[Aggregation],
    criterion: str,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Choose the swept parameter of every document on the **fit side** only.

    The document-level holdout of the plan's Stage 3, applied to a swept
    parameter: for every document the threshold is chosen on the other two
    documents (the fit side), never on the document it is applied to. The
    criterion is the fit side's pair AUC (``"auc"``) or its floor gap
    (``"floor_gap"``).

    Returns:
        The chosen threshold per rule and document, and the rotation report.
    """
    documents = [stratum["document"] for stratum in view.strata]
    rows, columns, labels = _pair_indices(view)
    control_rows, control_columns, control_documents = _control_indices(view)
    pair_documents = np.asarray([record.document for record in view.pairs])
    chosen: dict[str, dict[str, float]] = {rule.slug: {} for rule in aggregations}
    rotations: list[dict[str, Any]] = []
    for holdout in documents:
        fit_pairs = pair_documents != holdout
        holdout_pairs = pair_documents == holdout
        fit_control = control_documents != holdout
        holdout_control = control_documents == holdout
        grid: dict[str, dict[str, Any]] = {}
        for threshold in thresholds:
            per_rule: dict[str, Any] = {}
            for rule in aggregations:
                scores = matrices[threshold][rule.slug]
                pair_scores = scores[rows, columns]
                control_scores = scores[control_rows, control_columns]
                per_rule[rule.slug] = {
                    "fit": _side_numbers(
                        labels, pair_scores, fit_pairs, control_scores, fit_control
                    ),
                    "holdout": _side_numbers(
                        labels, pair_scores, holdout_pairs, control_scores, holdout_control
                    ),
                }
            grid[str(threshold)] = per_rule
        rotation: dict[str, Any] = {
            "holdout": holdout,
            "fit_documents": [document for document in documents if document != holdout],
            "n_fit_pairs": int(fit_pairs.sum()),
            "n_holdout_pairs": int(holdout_pairs.sum()),
            "grid": grid,
            "chosen": {},
        }
        for rule in aggregations:
            values = {
                threshold: grid[str(threshold)][rule.slug]["fit"].get(criterion)
                for threshold in thresholds
            }
            best = _argmax_threshold(values, thresholds)
            chosen[rule.slug][holdout] = best
            rotation["chosen"][rule.slug] = {
                "threshold": best,
                "criterion": criterion,
                "fit": grid[str(best)][rule.slug]["fit"],
                "holdout": grid[str(best)][rule.slug]["holdout"],
            }
        rotations.append(rotation)
    return chosen, rotations


def _swept_variant_arms(
    view: DatasetView,
    *,
    sides: Sequence[ChunkedSide],
    aggregations: Sequence[Aggregation],
    paragraph_texts: Sequence[str],
    question_texts: Sequence[str],
    baseline_question_side: SideEncoding,
    baseline_similarities: np.ndarray,
    question_transform: str,
    question_block_markers: Sequence[str],
    preprocess_spec: PreprocessSpec,
    hit_ks: Sequence[int],
    criterion: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """The arms of a swept-parameter variant (the semantic threshold).

    Every threshold of the grid is scored on **every** document, and the
    threshold of a document is chosen on the fit side (the other two
    documents) — the document-level cross-fit. The arm's score matrix is
    therefore assembled row by row: every document's rows come from the
    threshold that was chosen without seeing that document.
    """
    thresholds: list[float] = []
    for side in sides:
        if side.threshold is None:
            raise ValueError("a swept side needs a threshold")
        thresholds.append(side.threshold)
    matrices: dict[float, dict[str, np.ndarray]] = {}
    sides_by_threshold: dict[float, tuple[SideEncoding, SideEncoding]] = {}
    for side in sides:
        assert side.threshold is not None
        paragraph_side = encode_side(
            side.paragraph_embedder,
            side.chunker,
            list(paragraph_texts),
            "document",
            meta=_paragraph_meta(preprocess_spec, side.chunker),
        )
        question_side = _variant_question_side(
            side,
            baseline_question_side,
            question_texts,
            question_transform,
            question_block_markers,
        )
        matrices[side.threshold] = {
            rule.slug: side_similarities(
                paragraph_side,
                question_side,
                rule,
                global_similarities=baseline_similarities,
            )
            for rule in aggregations
        }
        sides_by_threshold[side.threshold] = (paragraph_side, question_side)

    chosen, rotations = _cross_fit(view, matrices, thresholds, aggregations, criterion)
    row_documents = _row_documents(view)
    chunk_counts = {str(t): sides_by_threshold[t][0].n_chunks for t in thresholds}
    question_chunk_counts = {str(t): sides_by_threshold[t][1].n_chunks for t in thresholds}
    arms: dict[str, Any] = {}
    similarities_by_arm: dict[str, np.ndarray] = {}
    for rule in aggregations:
        matrix = np.zeros((len(view.paragraphs), len(view.questions)), dtype=np.float64)
        for row, document in enumerate(row_documents):
            threshold = chosen[rule.slug][document]
            matrix[row] = matrices[threshold][rule.slug][row]
        arm_name = f"{_chunker_slug(sides[0].chunker)}_{rule.slug}"
        paragraph_side, question_side = sides_by_threshold[thresholds[0]]
        arm = score_arm(
            view,
            arm_name=arm_name,
            similarities=matrix,
            paragraph_side=paragraph_side,
            question_side=question_side,
            aggregation=rule,
            hit_ks=hit_ks,
        )
        arm["sides"]["paragraph"].update(
            {
                "n_chunks": int(round(float(np.mean(list(chunk_counts.values()))))),
                "n_chunks_by_threshold": chunk_counts,
                "swept_parameter": {
                    "name": "threshold",
                    "values": list(thresholds),
                    "selection": (
                        "the document-level cross-fit: the threshold of a document "
                        "is chosen on the fit side (the other two documents)"
                    ),
                },
            }
        )
        arm["sides"]["question"].update(
            {
                "n_chunks": int(round(float(np.mean(list(question_chunk_counts.values()))))),
                "n_chunks_by_threshold": question_chunk_counts,
            }
        )
        arms[arm_name] = arm
        similarities_by_arm[arm_name] = matrix
    selection = {
        "criterion": criterion,
        "grid": list(thresholds),
        "chosen_by_document": chosen,
        "rotations": rotations,
        "note": (
            "the swept parameter is chosen on the fit side only: for every "
            "document the threshold comes from the rotation that holds that "
            "document out, so no document ever sees its own fitted value"
        ),
    }
    return arms, similarities_by_arm, selection


@dataclass
class BaselineArm:
    """The baseline arm's internals (its full-text cosine matrix).

    The Stage-3 arm compares itself with the full-text cosine, so the driver
    hands the matrix out of :func:`score_model_arms` through this holder
    instead of encoding the texts twice.
    """

    similarities: np.ndarray | None = None


def score_model_arms(
    view: DatasetView,
    *,
    model_name: str,
    paragraph_embedder: Embedder,
    question_embedders: Mapping[str, Embedder],
    baseline_chunker: Chunker,
    question_transform: str,
    question_block_markers: Sequence[str],
    preprocess_spec: PreprocessSpec,
    chunked_side: ChunkedSide | None = None,
    swept_sides: Sequence[ChunkedSide] = (),
    aggregations: Sequence[Aggregation] = (),
    semantic_selection: str = SEMANTIC_CRITERION_AUC,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
    hit_ks: Sequence[int] = HIT_KS,
    baseline: BaselineArm | None = None,
) -> dict[str, Any]:
    """Score the baseline arm and every structural / question arm of one model.

    The baseline arm (full text against full text) is always computed; it is
    the reference every other arm is compared with, in the same run and on
    the same cells, so the comparison is paired by construction. Paragraphs
    are embedded under the ``document`` kind, questions under ``query`` (the
    same assignment the probe uses: the paragraph is the indexed text, the
    question the query it is matched against).

    Args:
        view: The model-independent dataset view.
        model_name: The short handle of the model (the artifact's row key).
        paragraph_embedder: The embedder of the baseline (full-text) paragraph side.
        question_embedders: The embedder of every question side, keyed by its
            transform (the baseline transform plus the extra ones).
        baseline_chunker: The chunker of the baseline arms (the full-text one).
        question_transform: The baseline question transform.
        question_block_markers: The markers of the question transform.
        preprocess_spec: The paragraph side's preprocess spec.
        chunked_side: The chunked paragraph side (and, for the symmetric
            variants, the chunked question side).
        swept_sides: One chunked side per point of a swept grid (the semantic
            threshold); the arms are then cross-fitted per document.
        aggregations: The aggregation rules the chunked arms fold with.
        semantic_selection: The fit-side criterion of the swept parameter
            (``"auc"`` or ``"floor_gap"``).
        bootstrap_resamples: The number of bootstrap resamples of the paired
            statistics.
        bootstrap_seed: The seed of the bootstrap.
        hit_ks: The hit@k grid of the rank report.
        baseline: An optional holder that receives the baseline arm's
            full-text cosine matrix (the Stage-3 arm's reference).

    Returns:
        The model's artifact block: the baseline arm's fields at the top
        level, the other arms under ``arms`` and their comparisons against
        the baseline under ``comparisons``.
    """
    paragraph_texts = preprocess_spec.apply(list(view.paragraphs))
    paragraph_side = encode_side(
        paragraph_embedder,
        baseline_chunker,
        paragraph_texts,
        "document",
        meta=_paragraph_meta(preprocess_spec, baseline_chunker),
    )
    question_sides: dict[str, SideEncoding] = {}
    question_texts_by_transform: dict[str, list[str]] = {}
    for transform, embedder in question_embedders.items():
        question_texts = apply_question_transform(
            list(view.questions), transform, question_block_markers
        )
        question_texts_by_transform[transform] = question_texts
        question_sides[transform] = encode_side(
            embedder,
            baseline_chunker,
            question_texts,
            "query",
            meta=_question_meta(transform, question_block_markers, baseline_chunker),
        )
    if question_transform not in question_sides:
        raise ValueError(f"the baseline question transform {question_transform!r} has no embedder")

    baseline_question_side = question_sides[question_transform]
    baseline_similarities = side_similarities(paragraph_side, baseline_question_side, None)
    if baseline is not None:
        baseline.similarities = baseline_similarities
    baseline_block = score_arm(
        view,
        arm_name=BASELINE_ARM,
        similarities=baseline_similarities,
        paragraph_side=paragraph_side,
        question_side=baseline_question_side,
        hit_ks=hit_ks,
    )

    arms: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    arm_similarities: dict[str, np.ndarray] = {}
    selection: dict[str, Any] | None = None

    # The question-side arms: the full-text paragraph side against another
    # question transform (the free keep-vs-strip_block comparison).
    for transform, question_side in question_sides.items():
        if transform == question_transform:
            continue
        arm_name = f"{BASELINE_ARM}_{transform}"
        similarities = side_similarities(paragraph_side, question_side, None)
        arms[arm_name] = score_arm(
            view,
            arm_name=arm_name,
            similarities=similarities,
            paragraph_side=paragraph_side,
            question_side=question_side,
            hit_ks=hit_ks,
        )
        arm_similarities[arm_name] = similarities

    # The chunked arms: the chunked paragraph side folded by an aggregation
    # rule, against the baseline's question side.
    if swept_sides:
        if not aggregations:
            raise ValueError("the swept arms need aggregation rules")
        swept_arms, swept_similarities, selection = _swept_variant_arms(
            view,
            sides=swept_sides,
            aggregations=aggregations,
            paragraph_texts=paragraph_texts,
            question_texts=question_texts_by_transform[question_transform],
            baseline_question_side=baseline_question_side,
            baseline_similarities=baseline_similarities,
            question_transform=question_transform,
            question_block_markers=question_block_markers,
            preprocess_spec=preprocess_spec,
            hit_ks=hit_ks,
            criterion=semantic_selection,
        )
        arms.update(swept_arms)
        arm_similarities.update(swept_similarities)
    elif chunked_side is not None:
        if not aggregations:
            raise ValueError("the chunked arms need aggregation rules")
        chunked_arms, chunked_similarities = _fixed_variant_arms(
            view,
            side=chunked_side,
            aggregations=aggregations,
            paragraph_texts=paragraph_texts,
            question_texts=question_texts_by_transform[question_transform],
            baseline_question_side=baseline_question_side,
            baseline_similarities=baseline_similarities,
            question_transform=question_transform,
            question_block_markers=question_block_markers,
            preprocess_spec=preprocess_spec,
            hit_ks=hit_ks,
        )
        arms.update(chunked_arms)
        arm_similarities.update(chunked_similarities)

    for arm_name, similarities in arm_similarities.items():
        comparisons[arm_name] = compare_arms(
            view,
            arm_name=arm_name,
            reference_similarities=baseline_similarities,
            arm_similarities=similarities,
            resamples=bootstrap_resamples,
            seed=bootstrap_seed,
            hit_ks=hit_ks,
        )

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
        **baseline_block,
        "arms": arms,
        "comparisons": comparisons,
        "threshold_selection": selection,
    }


def _cell_id(code: str, paragraph_number: str, question_number: int) -> str:
    """The id of one grid cell (the file's own pair-id convention)."""
    return f"{code}-{paragraph_number}-Q{question_number}"


# ---------------------------------------------------------------------------
# Stage 3: the learned pair scorer (the document-level holdout)
# ---------------------------------------------------------------------------

# The name of the Stage-3 arm (the learned scorer beside the full-text
# baseline arm).
STAGE3_ARM = "stage3"


@dataclass(frozen=True)
class Stage3Grid:
    """The same-document grid the learned scorer scores.

    The Stage-3 arm is a **per-pair** model, so it scores the whole grid: the
    rank of a pair inside its paragraph's candidate list needs every
    candidate's score, and the candidate list is the paragraph's document's
    questions — the same scope the Stage-2 arms use. The cells are ordered
    (document, row, column), so the artifact's numbers are reproducible.
    """

    rows: np.ndarray
    columns: np.ndarray
    documents: np.ndarray
    pairs: tuple[TextPair, ...]
    index: dict[tuple[int, int], int]

    @property
    def n_cells(self) -> int:
        """The number of grid cells."""
        return int(self.rows.size)


def build_stage3_grid(view: DatasetView) -> Stage3Grid:
    """The grid's cells (documents in the strata's order, rows and columns ascending)."""
    rows: list[int] = []
    columns: list[int] = []
    documents: list[str] = []
    pairs: list[TextPair] = []
    index: dict[tuple[int, int], int] = {}
    for grid in view.grids:
        for row in grid.paragraph_rows:
            for column in grid.question_cols:
                index[(row, column)] = len(pairs)
                rows.append(row)
                columns.append(column)
                documents.append(grid.document)
                pairs.append(
                    TextPair(
                        paragraph_text=view.paragraphs[row],
                        question_text=view.questions[column],
                    )
                )
    return Stage3Grid(
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        documents=np.asarray(documents),
        pairs=tuple(pairs),
        index=index,
    )


def _stage3_control_block(
    control_scores: np.ndarray,
    control_percentiles: np.ndarray,
    control_documents: np.ndarray,
) -> dict[str, Any]:
    """The control-side counterpart of one arm (its level and its rank position)."""
    by_document: dict[str, Any] = {}
    for document in sorted(set(control_documents.tolist())):
        mask = control_documents == document
        scores = control_scores[mask]
        percentiles = control_percentiles[mask]
        by_document[document] = {
            "n": int(mask.sum()),
            "mean": float(scores.mean()) if scores.size else None,
            "p50": _quantile(scores, 0.5),
            "rank_p50": _quantile(percentiles, 0.5),
            "top_fifth_share": _top_quintile_share(percentiles),
        }
    return {
        "n": int(control_scores.size),
        "mean": float(control_scores.mean()) if control_scores.size else None,
        "p50": _quantile(control_scores, 0.5),
        "rank_p50": _quantile(control_percentiles, 0.5),
        "top_fifth_share": _top_quintile_share(control_percentiles),
        "by_document": by_document,
    }


def _deshifted_auc(
    view: DatasetView,
    labels: np.ndarray,
    pair_scores: np.ndarray,
    pair_documents: np.ndarray,
    control_scores: np.ndarray,
    control_documents: np.ndarray,
) -> dict[str, Any]:
    """The AUC after each document's control level is removed.

    The de-shift is computed **inside one arm's own scale**: the control's
    per-document mean minus the control's pooled mean is subtracted from that
    document's reviewed-set scores. A uniform additive shift cannot change an
    AUC, so this removes the *document-level* part of the score level and
    leaves the ranking inside a document (the track's transferable statistic).
    """
    pooled = float(control_scores.mean()) if control_scores.size else 0.0
    shifts: dict[str, float] = {}
    deshifted = np.empty_like(pair_scores, dtype=np.float64)
    for stratum in view.strata:
        document = stratum["document"]
        control_mask = control_documents == document
        shift = float(control_scores[control_mask].mean()) - pooled if control_mask.any() else 0.0
        shifts[document] = shift
        pair_mask = pair_documents == document
        deshifted[pair_mask] = pair_scores[pair_mask] - shift
    return {
        "overall": _safe_auc(labels, deshifted),
        "macro": _macro(
            {
                stratum["document"]: _safe_auc(
                    labels[pair_documents == stratum["document"]],
                    deshifted[pair_documents == stratum["document"]],
                )
                for stratum in view.strata
            }
        ),
        "by_document": {
            stratum["document"]: _safe_auc(
                labels[pair_documents == stratum["document"]],
                deshifted[pair_documents == stratum["document"]],
            )
            for stratum in view.strata
        },
        "per_document_shift": shifts,
    }


def _floor_gap_block(
    positive_scores: np.ndarray, control_scores: np.ndarray
) -> dict[str, float | None]:
    """The floor gap (the positives' mean minus the control's mean) and its standardized form."""
    gap = _floor_gap(positive_scores, control_scores)
    spread = float(control_scores.std()) if control_scores.size else 0.0
    return {
        "gap": gap,
        "standardized": (gap / spread if gap is not None and spread > 0 else None),
    }


def compare_stage3_arm(
    view: DatasetView,
    *,
    arm_name: str,
    reference_similarities: np.ndarray,
    arm_similarities: np.ndarray,
    resamples: int,
    seed: int,
    hit_ks: Sequence[int] = HIT_KS,
) -> dict[str, Any]:
    """Compare the learned arm with the full-text baseline arm (the same blocks).

    The blocks are the Stage-2 ones (the pair AUC pooled / per document /
    macro, the within/cross decomposition, the control-side counterpart, the
    de-shifted AUC, the floor gap and the hit@k), with one difference forced
    by the arms' scales: the learned arm's scores are **probabilities** and
    the baseline's are **cosines**, so the level numbers (the control's mean,
    the floor gap) are reported in each arm's own units and are *not*
    comparable. The scale-free numbers (the AUCs, the rank positions, the
    standardized floor gap) are comparable, and they carry the verdict.
    """
    reference = measure_arm(view, reference_similarities)
    arm = measure_arm(view, arm_similarities)
    labels = arm.labels

    paired = paired_auc_bootstrap(
        labels.tolist(),
        reference.pair_scores.tolist(),
        arm.pair_scores.tolist(),
        n_resamples=resamples,
        seed=seed,
    )
    reference_decomposition = _auc_decomposition(labels, reference.pair_scores, reference.documents)
    arm_decomposition = _auc_decomposition(labels, arm.pair_scores, arm.documents)
    within_reference = reference_decomposition.get("within_group_auc")
    within_arm = arm_decomposition.get("within_group_auc")
    cross_reference = reference_decomposition.get("cross_group_auc")
    cross_arm = arm_decomposition.get("cross_group_auc")
    within_paired = paired_within_group_auc_bootstrap(
        labels.tolist(),
        reference.pair_scores.tolist(),
        arm.pair_scores.tolist(),
        arm.documents.tolist(),
        n_resamples=resamples,
        seed=seed,
    )

    reference_auc_by_document = {
        stratum["document"]: _safe_auc(
            labels[reference.documents == stratum["document"]],
            reference.pair_scores[reference.documents == stratum["document"]],
        )
        for stratum in view.strata
    }
    arm_auc_by_document = {
        stratum["document"]: _safe_auc(
            labels[arm.documents == stratum["document"]],
            arm.pair_scores[arm.documents == stratum["document"]],
        )
        for stratum in view.strata
    }

    control_reference = _stage3_control_block(
        reference.control_scores, reference.control_percentiles, reference.control_documents
    )
    control_arm = _stage3_control_block(
        arm.control_scores, arm.control_percentiles, arm.control_documents
    )
    floor_reference = _floor_gap_block(reference.pair_scores[labels == 1], reference.control_scores)
    floor_arm = _floor_gap_block(arm.pair_scores[labels == 1], arm.control_scores)

    return {
        "arm": arm_name,
        "reference_arm": BASELINE_ARM,
        "paired_auc": paired,
        "pair_auc": {
            "reference": paired["auc_reference"],
            "arm": paired["auc_arm"],
            "delta": paired["delta_auc"],
            "macro": {
                "reference": _macro(reference_auc_by_document),
                "arm": _macro(arm_auc_by_document),
                "delta": (
                    _macro(arm_auc_by_document) - _macro(reference_auc_by_document)
                    if _macro(arm_auc_by_document) is not None
                    and _macro(reference_auc_by_document) is not None
                    else None
                ),
            },
            "by_document": {
                document: {
                    "reference": reference_auc_by_document[document],
                    "arm": arm_auc_by_document[document],
                    "delta": (
                        arm_auc_by_document[document] - reference_auc_by_document[document]
                        if arm_auc_by_document[document] is not None
                        and reference_auc_by_document[document] is not None
                        else None
                    ),
                }
                for document in reference_auc_by_document
            },
        },
        "auc_decomposition": {
            "reference": reference_decomposition,
            "arm": arm_decomposition,
            "within_group_delta": (
                within_arm - within_reference
                if within_arm is not None and within_reference is not None
                else None
            ),
            "cross_group_delta": (
                cross_arm - cross_reference
                if cross_arm is not None and cross_reference is not None
                else None
            ),
            "within_group_paired": within_paired,
            "note": (
                "the pooled AUC mixes the ranking inside a document with the "
                "ranking across documents; the within-document delta is the "
                "part that transfers to a corpus without document structure"
            ),
        },
        "control_side": {
            "reference": control_reference,
            "arm": control_arm,
            "rank_p50_delta": (
                control_arm["rank_p50"] - control_reference["rank_p50"]
                if control_arm["rank_p50"] is not None and control_reference["rank_p50"] is not None
                else None
            ),
            "top_fifth_share_delta": (
                control_arm["top_fifth_share"] - control_reference["top_fifth_share"]
                if control_arm["top_fifth_share"] is not None
                and control_reference["top_fifth_share"] is not None
                else None
            ),
            "note": (
                "the control-side counterpart of the discrimination numbers. The "
                "level numbers (mean, p50) are in each arm's own units — the "
                "learned arm scores probabilities, the baseline scores cosines — "
                "so they are not comparable; the rank numbers (rank_p50, the top-"
                "fifth share) are scale-free and are."
            ),
        },
        "negative_enrichment": {
            "rank_percentile": {
                "label1": _distribution(arm.pair_percentiles[labels == 1]),
                "label0": _distribution(arm.pair_percentiles[labels == 0]),
                "control": _distribution(arm.control_percentiles),
            },
            "top_fifth_share": {
                "label1": _top_quintile_share(arm.pair_percentiles[labels == 1]),
                "label0": _top_quintile_share(arm.pair_percentiles[labels == 0]),
                "control": _top_quintile_share(arm.control_percentiles),
            },
        },
        "floor_gap": {
            "reference": floor_reference,
            "arm": floor_arm,
            "standardized_delta": (
                floor_arm["standardized"] - floor_reference["standardized"]
                if floor_arm["standardized"] is not None
                and floor_reference["standardized"] is not None
                else None
            ),
            "note": (
                "the gap between the verified positives' mean score and the "
                "control's mean score, in each arm's own units; the standardized "
                "form divides it by the control's spread and is comparable"
            ),
        },
        "deshifted_auc": {
            "reference": _deshifted_auc(
                view,
                labels,
                reference.pair_scores,
                reference.documents,
                reference.control_scores,
                reference.control_documents,
            ),
            "arm": _deshifted_auc(
                view,
                labels,
                arm.pair_scores,
                arm.documents,
                arm.control_scores,
                arm.control_documents,
            ),
            "note": (
                "each arm's own per-document control level removed (the control's "
                "per-document mean minus its pooled mean): the ranking inside a "
                "document survives the document's score level"
            ),
        },
        "hit_at_k": {
            "reference": {
                "label1": _hit_at_k(reference.pair_ranks[labels == 1], hit_ks),
                "label0": _hit_at_k(reference.pair_ranks[labels == 0], hit_ks),
                "control": _hit_at_k(reference.control_ranks, hit_ks),
            },
            "arm": {
                "label1": _hit_at_k(arm.pair_ranks[labels == 1], hit_ks),
                "label0": _hit_at_k(arm.pair_ranks[labels == 0], hit_ks),
                "control": _hit_at_k(arm.control_ranks, hit_ks),
            },
        },
    }


def _stage3_rotation(
    view: DatasetView,
    *,
    holdout: str,
    documents: Sequence[str],
    features: np.ndarray,
    grid: Stage3Grid,
    pair_cells: np.ndarray,
    control_cells: np.ndarray,
    labels: np.ndarray,
    pair_documents: np.ndarray,
    control_documents: np.ndarray,
    combiner_factory: Callable[[], Combiner],
    control_weight: float,
    permutation_resamples: int,
    rng: np.random.Generator,
    reference_similarities: np.ndarray,
    feature_names: Sequence[str] = (),
) -> dict[str, Any]:
    """One document-level holdout: fit on two documents, predict the third.

    The fit side's rows are its **judged** pairs (the verified positives and
    the reviewer-verified hard negatives) plus its **unjudged control cells**
    as presumed negatives, down-weighted by ``control_weight`` (the control's
    true positive rate is unmeasured, so it must not dominate the fit). The
    holdout's rows are never seen by the fit.
    """
    fit_pairs = pair_documents != holdout
    holdout_pairs = pair_documents == holdout
    fit_control = control_documents != holdout
    holdout_control = control_documents == holdout
    holdout_grid = grid.documents == holdout

    fit_features = np.concatenate(
        [features[pair_cells[fit_pairs]], features[control_cells[fit_control]]], axis=0
    )
    fit_labels = np.concatenate(
        [labels[fit_pairs], np.zeros(int(fit_control.sum()), dtype=np.int64)], axis=0
    )
    fit_weights = np.concatenate(
        [
            np.ones(int(fit_pairs.sum()), dtype=np.float64),
            np.full(int(fit_control.sum()), float(control_weight), dtype=np.float64),
        ],
        axis=0,
    )

    combiner = combiner_factory().fit(fit_features, fit_labels, fit_weights)
    holdout_scores = combiner.score(features[holdout_grid])

    # The label-permutation floors: the same fit with the labels destroyed.
    #
    # - **chance floor** (the plan's): the fit side's **whole label vector**
    #   is permuted (the judged pairs' labels and the control cells' labels
    #   together, so the class counts and the weights stay exactly as they
    #   are and only the assignment changes). The model then has no label
    #   information at all, so the AUC must land at chance (0.5).
    # - **control-role floor** (the diagnostic): only the judged pairs'
    #   labels are permuted and the control cells keep their presumed-negative
    #   role. The gap to the chance floor is what the *training design* alone
    #   (the "the ordinary unjudged cells are negative" prior) buys on the
    #   reviewed set — a label-free signal the arm inherits.
    chance_auc: list[float | None] = []
    control_role_auc: list[float | None] = []
    control_labels = np.zeros(int(fit_control.sum()), dtype=np.int64)
    for _ in range(permutation_resamples):
        shuffled = labels[fit_pairs].copy()
        rng.shuffle(shuffled)
        permuted_labels = np.concatenate([shuffled, control_labels], axis=0)
        # the control-role floor: the control cells keep their presumed-negative role
        permuted = combiner_factory().fit(fit_features, permuted_labels, fit_weights)
        permuted_scores = permuted.score(features[pair_cells[holdout_pairs]])
        control_role_auc.append(_safe_auc(labels[holdout_pairs], permuted_scores))
        # the chance floor: the whole label vector is permuted
        permuted_pool = permuted_labels.copy()
        rng.shuffle(permuted_pool)
        chance = combiner_factory().fit(fit_features, permuted_pool, fit_weights)
        chance_scores = chance.score(features[pair_cells[holdout_pairs]])
        chance_auc.append(_safe_auc(labels[holdout_pairs], chance_scores))

    reference = measure_arm(view, reference_similarities)
    holdout_reference_scores = reference.pair_scores[holdout_pairs]
    holdout_pair_auc = _safe_auc(
        labels[holdout_pairs], combiner.score(features[pair_cells[holdout_pairs]])
    )
    holdout_reference_auc = _safe_auc(labels[holdout_pairs], holdout_reference_scores)
    # The question overlap between the two sides: the plan's leakage rule (i).
    # 186/294 question numbers are judged in more than one document, so a
    # *question-level* feature (the question-type block) can carry the fit
    # side's label for a question the holdout also carries. The overlap is
    # measured here, so the question-type arm's number can be read against it.
    fit_questions = set(grid.columns[pair_cells[fit_pairs]].tolist())
    holdout_columns = grid.columns[pair_cells[holdout_pairs]]
    holdout_questions = set(holdout_columns.tolist())
    shared_questions = fit_questions & holdout_questions
    return {
        "holdout": holdout,
        "fit_documents": [document for document in documents if document != holdout],
        "n_fit_pairs": int(fit_pairs.sum()),
        "n_fit_positive": int((labels[fit_pairs] == 1).sum()),
        "n_fit_negative": int((labels[fit_pairs] == 0).sum()),
        "n_fit_control": int(fit_control.sum()),
        "control_weight": float(control_weight),
        "fit_weight_mass": {
            "judged": float(fit_weights[: int(fit_pairs.sum())].sum()),
            "control": float(fit_weights[int(fit_pairs.sum()) :].sum()),
        },
        "n_holdout_pairs": int(holdout_pairs.sum()),
        "n_holdout_positive": int((labels[holdout_pairs] == 1).sum()),
        "n_holdout_negative": int((labels[holdout_pairs] == 0).sum()),
        "n_holdout_control": int(holdout_control.sum()),
        "n_holdout_grid_cells": int(holdout_grid.sum()),
        "question_overlap": {
            "n_holdout_questions": len(holdout_questions),
            "n_holdout_questions_also_judged_in_fit": len(shared_questions),
            "n_holdout_pairs_with_a_shared_question": int(
                np.isin(holdout_columns, list(shared_questions)).sum()
            ),
            "note": (
                "how many of the holdout's questions the fit side also judged. "
                "The features are pair-local (no question-level statistic "
                "enters them), but a question-type weight fitted on the fit "
                "side reaches the holdout's copy of the same question, so a "
                "question-type-only arm must be read against this overlap."
            ),
        },
        "holdout_pair_auc": holdout_pair_auc,
        "holdout_reference_pair_auc": holdout_reference_auc,
        "holdout_pair_auc_delta": (
            holdout_pair_auc - holdout_reference_auc
            if holdout_pair_auc is not None and holdout_reference_auc is not None
            else None
        ),
        "holdout_grid_positions": holdout_grid,
        "holdout_scores": holdout_scores,
        "permutation_floor": {
            "n_resamples": int(permutation_resamples),
            "chance_auc": chance_auc,
            "control_role_auc": control_role_auc,
            "note": (
                "the label-permutation floors. The chance floor permutes the fit "
                "side's whole label vector (the class counts and the weights stay "
                "as they are, only the assignment changes), so the model has no "
                "label information and the AUC must land at chance (0.5). The "
                "control-role floor permutes only the judged pairs' labels and "
                "keeps the control cells' presumed-negative role: the gap to the "
                "chance floor is what the training design alone buys on the "
                "reviewed set."
            ),
        },
        "coefficients": _named_coefficients(combiner, feature_names),
        "feature_usage": _named_feature_usage(combiner, feature_names),
        "combiner": combiner.describe(),
    }


def _feature_diagnostics(
    view: DatasetView,
    *,
    features: np.ndarray,
    feature_names: Sequence[str],
    grid: Stage3Grid,
    labels: np.ndarray,
    pair_documents: np.ndarray,
    control_documents: np.ndarray,
    control_lengths: np.ndarray,
) -> dict[str, Any]:
    """Every feature's own numbers: its discrimination and its control side.

    The Stage-3.3 change set asks for **every new feature's control-side
    counterpart next to its discrimination number**, so each feature gets:

    - the **discrimination** number: the within-document AUC of the feature
      *used as a score* (the same statistic the headline is measured with),
      the pooled AUC and the per-document macro beside it;
    - the **control-side counterpart**: the ordinary (unjudged) cells' mean
      and median and their rank percentile inside the paragraph's candidate
      list — a feature that lifts the ordinary cells shows up here next to
      one that separates the reviewed set;
    - the **length correlation**: how much the feature's value travels with
      the paragraph's length. A feature that is a length proxy would
      memorize the document instead of the topic (the plan's leakage rule
      (ii)), so the number is measured, not assumed.

    The features are measured on the **whole grid** (the same-document
    cells), so the reviewed pairs and the control cells are scored on one
    scale.
    """
    per_feature: dict[str, Any] = {}
    for index, name in enumerate(feature_names):
        values = np.asarray(features[:, index], dtype=np.float64)
        matrix = np.full((len(view.paragraphs), len(view.questions)), np.nan, dtype=np.float64)
        matrix[grid.rows, grid.columns] = values
        measured = measure_arm(view, matrix)
        decomposition = _auc_decomposition(labels, measured.pair_scores, measured.documents)
        by_document = {
            stratum["document"]: _safe_auc(
                labels[pair_documents == stratum["document"]],
                measured.pair_scores[pair_documents == stratum["document"]],
            )
            for stratum in view.strata
        }
        per_feature[name] = {
            "within_auc": decomposition.get("within_group_auc"),
            "pooled_auc": _safe_auc(labels, measured.pair_scores),
            "macro_auc": _macro(by_document),
            "by_document": by_document,
            "control": {
                "mean": float(measured.control_scores.mean())
                if measured.control_scores.size
                else None,
                "p50": _quantile(measured.control_scores, 0.5),
                "rank_p50": _quantile(measured.control_percentiles, 0.5),
                "top_fifth_share": _top_quintile_share(measured.control_percentiles),
            },
            "length_correlation": length_correlation(
                control_lengths, measured.control_scores, control_documents
            ),
        }
    return {
        "per_feature": per_feature,
        "note": (
            "every feature the pack computes, measured on its own: the "
            "within-document AUC (the headline's statistic, the feature used "
            "as a score), the pooled AUC and the per-document macro beside "
            "it; the control-side counterpart (the ordinary unjudged cells' "
            "level and rank position); and the correlation between the "
            "feature and the paragraph's length — the length-proxy check, "
            "since a length feature would memorize the document instead of "
            "the topic."
        ),
    }


def _length_check(
    *,
    arm: ArmScores,
    reference: ArmScores,
    control_lengths: np.ndarray,
) -> dict[str, Any]:
    """The length check: does the arm's score travel with the paragraph's length?

    The Stage-0 caveat says a method's scale shift is partly a **length**
    effect and partly a **document** effect. The baseline arm's correlation
    is the reference (it is already recorded per document); the arm's own
    correlation sits beside it, and the **rank shift** (the arm's control
    rank percentile minus the baseline's) is correlated with the length as
    well — the rank shift is scale-free, so it compares the two arms without
    their different units (a probability against a cosine).
    """
    rank_shift = arm.control_percentiles - reference.control_percentiles
    return {
        "arm": length_correlation(control_lengths, arm.control_scores, arm.control_documents),
        "reference": length_correlation(
            control_lengths, reference.control_scores, reference.control_documents
        ),
        "rank_shift": length_correlation(control_lengths, rank_shift, arm.control_documents),
        "note": (
            "the correlation between a control cell's paragraph length and its "
            "score, for the learned arm and for the full-text cosine (the "
            "reference). The rank-shift block is the same correlation for the "
            "difference of the two arms' rank percentiles — it is scale-free, "
            "so it compares the arms without their different units."
        ),
    }


def _named_coefficients(combiner: Combiner, feature_names: Sequence[str]) -> dict[str, Any]:
    """The fitted coefficients, keyed by the feature name.

    The combiner standardizes its features, so the coefficients are
    comparable with each other: their magnitudes say which features the fit
    side leaned on. A coefficient on a **question-type** feature is a
    question-level prior by construction (the feature does not depend on the
    paragraph), so it is worth seeing.
    """
    getter = getattr(combiner, "coefficients", None)
    if getter is None:
        return {}
    values = getter()
    named: dict[str, float] = {}
    for index, value in values.items():
        position = int(index)
        name = feature_names[position] if position < len(feature_names) else f"feature_{position}"
        named[name] = float(value)
    return {
        "values": named,
        "note": (
            "the fit side's standardized coefficients: they say which features "
            "the fit leaned on. A question-type feature does not depend on the "
            "paragraph, so its coefficient is a question-level prior."
        ),
    }


def _named_feature_usage(combiner: Combiner, feature_names: Sequence[str]) -> dict[str, Any]:
    """Which features the fitted trees split on, keyed by the feature name.

    A tree ensemble has no coefficients: the record of its splits is what
    says which features the model used (the Stage-3.2 importances
    capability). The combiner that has no such record (the logistic
    regression) returns an empty block.

    The block is a **usage** diagnostic, not an accuracy-based importance: it
    says what the trees split on, not what would improve the score if a
    feature were removed. It is computed on the fit side's rows only.
    """
    getter = getattr(combiner, "feature_usage", None)
    if getter is None:
        return {}
    values = getter()
    named: dict[str, dict[str, float]] = {}
    for index, usage in values.items():
        position = int(index)
        name = feature_names[position] if position < len(feature_names) else f"feature_{position}"
        named[name] = {key: float(value) for key, value in usage.items()}
    return {
        "per_feature": named,
        "note": (
            "the fit side's trees' split usage: how often each feature was "
            "split on, the gain those splits bought and the shares of all the "
            "splits and all the gain. It is a usage diagnostic (what the "
            "model splits on), not an accuracy-based importance (what would "
            "improve the score if the feature were removed)."
        ),
    }


def score_stage3_model(
    view: DatasetView,
    *,
    model_name: str,
    extractor: FeatureExtractor,
    combiner_factory: Callable[[], Combiner],
    reference_similarities: np.ndarray,
    control_weight: float = 0.3,
    permutation_resamples: int = 5,
    seed: int = 42,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 42,
    hit_ks: Sequence[int] = HIT_KS,
    feature_embeddings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score the learned pair scorer of one model (the Stage-3 arm).

    The protocol is the plan's Stage-3 discipline: **document-level holdout**.
    For every document the combiner is fitted on the other two documents'
    judged pairs (plus their unjudged control cells as down-weighted presumed
    negatives) and predicts the held-out document. The out-of-fold scores of
    every judged pair and of every control cell then form **one** score
    matrix, on which the same comparison blocks the Stage-2 runs use are
    computed (the pair AUC pooled / per document / macro, the within/cross
    decomposition and its paired bootstrap, the control-side counterpart, the
    de-shifted AUC, the floor gap and the hit@k). The reference is the
    full-text cosine (the Stage-0 baseline arm) on the same held-out
    documents.

    Args:
        view: The model-independent dataset view.
        model_name: The short handle of the model (the artifact's row key).
        extractor: The feature pack (it embeds the pair's own chunks through
            the ``Embedder`` port).
        combiner_factory: A callable that builds a fresh combiner (one per
            fit: the rotations and the permutations must not share a model).
        reference_similarities: The full-text cosine matrix (the baseline).
        control_weight: The training weight of the fit side's control cells.
        permutation_resamples: The number of label permutations of the floor.
        seed: The seed of the permutations.
        bootstrap_resamples: The number of bootstrap resamples of the paired
            statistics.
        bootstrap_seed: The seed of the bootstrap.
        hit_ks: The hit@k grid of the rank report.
        feature_embeddings: The provenance of the feature pack's **embedding
            sides** — the run's model (the Stage-3.1 behaviour) or the
            random-vector floor (CS-6d). It is recorded in the artifact, so a
            reader knows which space the arm's embedding features live in.

    Returns:
        The model's Stage-3 block (the rotations, the out-of-fold summary, the
        comparison and the permutation floor).
    """
    grid = build_stage3_grid(view)
    all_features = extractor.extract_all(grid.pairs)
    if all_features.shape[0] != grid.n_cells:
        raise ValueError(
            f"the feature pack returned {all_features.shape[0]} rows for {grid.n_cells} cells"
        )
    # the combiner sees the selected groups' columns; the diagnostics see
    # every column the pack computes
    features = all_features[:, np.asarray(extractor.selected_indices, dtype=np.int64)]

    pair_rows, pair_columns, labels = _pair_indices(view)
    control_rows, control_columns, control_documents = _control_indices(view)
    pair_documents = np.asarray([record.document for record in view.pairs])
    pair_cells = np.asarray(
        [
            grid.index[(int(row), int(column))]
            for row, column in zip(pair_rows, pair_columns, strict=True)
        ],
        dtype=np.int64,
    )
    control_cells = np.asarray(
        [
            grid.index[(int(row), int(column))]
            for row, column in zip(control_rows, control_columns, strict=True)
        ],
        dtype=np.int64,
    )

    documents = [stratum["document"] for stratum in view.strata]
    rng = np.random.default_rng(seed)
    rotations: list[dict[str, Any]] = []
    out_of_fold = np.full((len(view.paragraphs), len(view.questions)), np.nan, dtype=np.float64)
    for holdout in documents:
        rotation = _stage3_rotation(
            view,
            holdout=holdout,
            documents=documents,
            features=features,
            grid=grid,
            pair_cells=pair_cells,
            control_cells=control_cells,
            labels=labels,
            pair_documents=pair_documents,
            control_documents=control_documents,
            combiner_factory=combiner_factory,
            control_weight=control_weight,
            permutation_resamples=permutation_resamples,
            rng=rng,
            reference_similarities=reference_similarities,
            feature_names=extractor.feature_names,
        )
        rotations.append(rotation)
        # The holdout's grid cells take the scores of the model that never
        # saw that document.
        holdout_grid = rotation.pop("holdout_grid_positions")
        holdout_scores = rotation.pop("holdout_scores")
        out_of_fold[grid.rows[holdout_grid], grid.columns[holdout_grid]] = holdout_scores

    comparison = compare_stage3_arm(
        view,
        arm_name=STAGE3_ARM,
        reference_similarities=reference_similarities,
        arm_similarities=out_of_fold,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
        hit_ks=hit_ks,
    )
    chance_values = [
        value
        for rotation in rotations
        for value in rotation["permutation_floor"]["chance_auc"]
        if value is not None
    ]
    control_role_values = [
        value
        for rotation in rotations
        for value in rotation["permutation_floor"]["control_role_auc"]
        if value is not None
    ]
    # The headline is the **within-document** AUC: the pair-count-weighted mean
    # of the documents' own AUCs (the same weighting the metric uses). The
    # floors are measured one document at a time, so they get the same
    # weighting here — otherwise a per-document mean would be compared with a
    # pair-count-weighted one.
    pair_counts = {
        stratum["document"]: stratum["n_positive"] * stratum["n_negative"]
        for stratum in view.strata
    }
    chance_within = _weighted_mean(
        [
            (rotation["holdout"], _mean_of(rotation["permutation_floor"]["chance_auc"]))
            for rotation in rotations
        ],
        pair_counts,
    )
    control_role_within = _weighted_mean(
        [
            (rotation["holdout"], _mean_of(rotation["permutation_floor"]["control_role_auc"]))
            for rotation in rotations
        ],
        pair_counts,
    )
    # The control cells' paragraph lengths (the length check's raw material).
    control_lengths = np.asarray(
        [view.paragraph_char_lengths[int(row)] for row in control_rows], dtype=np.float64
    )
    arm_measured = measure_arm(view, out_of_fold)
    reference_measured = measure_arm(view, reference_similarities)
    diagnostics = _feature_diagnostics(
        view,
        features=all_features,
        feature_names=extractor.all_feature_names,
        grid=grid,
        labels=labels,
        pair_documents=pair_documents,
        control_documents=control_documents,
        control_lengths=control_lengths,
    )
    length_check = _length_check(
        arm=arm_measured,
        reference=reference_measured,
        control_lengths=control_lengths,
    )
    return {
        "model": model_name,
        "feature_pack": extractor.describe(),
        "feature_embeddings": dict(feature_embeddings or {}),
        "combiner": rotations[0]["combiner"] if rotations else None,
        "out_of_fold_scores": {
            "judged_pairs": [
                float(value)
                for value in out_of_fold[grid.rows[pair_cells], grid.columns[pair_cells]]
            ],
            "control_cells": [
                float(value)
                for value in out_of_fold[grid.rows[control_cells], grid.columns[control_cells]]
            ],
            "note": (
                "the arm's out-of-fold score of every judged pair and every "
                "control cell, in the order of the model's 'judged_pairs' and "
                "'control_cells' blocks. They are the per-pair scores two arms "
                "of different runs can be compared on (a paired per-pair "
                "bootstrap), which the per-rotation AUCs alone cannot give."
            ),
        },
        "training": {
            "control_weight": float(control_weight),
            "fit_side": (
                "the fit side's judged pairs (the verified positives and the "
                "reviewer-verified hard negatives) plus its unjudged control "
                "cells as down-weighted presumed negatives"
            ),
            "leakage_rules": [
                "the fit side never contains the holdout document",
                "no corpus-level question statistic is used (the features are pair-local)",
                "no document-identity feature and no length feature",
                "the scaler of the combiner is fitted on the fit side's rows only",
            ],
        },
        "grid": {
            "n_cells": grid.n_cells,
            "n_matrix_positions": int(out_of_fold.size),
            "n_scored_cells": int(np.count_nonzero(~np.isnan(out_of_fold))),
            "n_unscored_positions": int(np.count_nonzero(np.isnan(out_of_fold))),
            "note": (
                "the out-of-fold matrix holds the same-document grid only: the "
                "cross-document positions are not scored (no rotation holds out "
                "two documents at once), and every statistic the comparison uses "
                "lives inside a document"
            ),
        },
        "rotations": rotations,
        "out_of_fold": {
            "pair_auc": comparison["pair_auc"],
            "auc_decomposition": comparison["auc_decomposition"],
            "control_side": comparison["control_side"],
            "floor_gap": comparison["floor_gap"],
            "deshifted_auc": comparison["deshifted_auc"],
            "hit_at_k": comparison["hit_at_k"],
        },
        "comparison": comparison,
        "feature_diagnostics": diagnostics,
        "length_check": length_check,
        "permutation_floor": {
            "n_resamples": int(permutation_resamples),
            "n_measured": len(chance_values),
            "chance": {
                "mean": float(np.mean(chance_values)) if chance_values else None,
                "min": float(np.min(chance_values)) if chance_values else None,
                "max": float(np.max(chance_values)) if chance_values else None,
            },
            "control_role": {
                "mean": float(np.mean(control_role_values)) if control_role_values else None,
                "min": float(np.min(control_role_values)) if control_role_values else None,
                "max": float(np.max(control_role_values)) if control_role_values else None,
            },
            "within_document": {
                "chance": chance_within,
                "control_role": control_role_within,
                "note": (
                    "the pair-count-weighted mean of the documents' own AUCs — "
                    "the same weighting the within-document AUC uses, so these "
                    "two numbers are comparable with the arm's within-document "
                    "AUC and the baseline's"
                ),
            },
            "note": (
                "the label-permutation floors, each a per-document AUC (one per "
                "rotation), so their mean is comparable with the arm's per-"
                "document macro AUC and not with its pooled AUC (the out-of-fold "
                "matrix mixes three models' score levels). A holdout with a "
                "single class has no AUC and is left out (n_measured)."
            ),
        },
        "rotations_summary": [
            {
                "holdout": rotation["holdout"],
                "arm_pair_auc": rotation["holdout_pair_auc"],
                "reference_pair_auc": rotation["holdout_reference_pair_auc"],
                "arm_minus_reference": rotation["holdout_pair_auc_delta"],
                "chance_floor_mean": _mean_of(rotation["permutation_floor"]["chance_auc"]),
                "control_role_floor_mean": _mean_of(
                    rotation["permutation_floor"]["control_role_auc"]
                ),
                "arm_minus_chance_floor": _difference(
                    rotation["holdout_pair_auc"],
                    _mean_of(rotation["permutation_floor"]["chance_auc"]),
                ),
                "arm_minus_control_role_floor": _difference(
                    rotation["holdout_pair_auc"],
                    _mean_of(rotation["permutation_floor"]["control_role_auc"]),
                ),
                "n_holdout_questions": rotation["question_overlap"]["n_holdout_questions"],
                "n_holdout_questions_also_judged_in_fit": rotation["question_overlap"][
                    "n_holdout_questions_also_judged_in_fit"
                ],
            }
            for rotation in rotations
        ],
    }


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
    chunker_params: Mapping[str, Any] | None = None,
    question_chunker_name: str | None = None,
    question_chunker_params: Mapping[str, Any] | None = None,
    semantic_thresholds: Sequence[float] = (),
    semantic_selection: str = SEMANTIC_CRITERION_AUC,
    aggregations: Sequence[str] = (),
    question_transforms: Sequence[str] = (),
    bootstrap: Mapping[str, Any] | None = None,
    stage3: Mapping[str, Any] | None = None,
    baseline_chunker_name: str = BASELINE_ARM,
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
            "question_transforms": list(question_transforms),
            "chunker": chunker_name,
            "chunker_params": dict(chunker_params or {}),
            "question_chunker": question_chunker_name,
            "question_chunker_params": dict(question_chunker_params or {}),
            "semantic_thresholds": list(semantic_thresholds),
            "semantic_selection": semantic_selection,
            "baseline_chunker": baseline_chunker_name,
            "aggregations": list(aggregations),
            "bootstrap": dict(bootstrap or {}),
            "stage3": None if stage3 is None else dict(stage3),
            "paragraph_lengths": paragraph_length_report(view),
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
                "The model's top-level fields are the baseline arm (full text against "
                "full text). The other arms are under 'arms' and their comparisons "
                "against the baseline under 'comparisons' (one per arm).",
                "A swept parameter (the semantic threshold) is chosen on the fit "
                "side only: the model's 'threshold_selection' block records the "
                "grid, the per-rotation fit-side numbers and the chosen value, so "
                "no document ever sees its own fitted parameter.",
                "The paragraph lengths are recorded in 'paragraph_lengths' (the "
                "length-vs-shift question); they are never a feature, because "
                "length is a document proxy on this data.",
                "A learned arm (the Stage-3 block) scores probabilities and the "
                "baseline arm scores cosines: the level numbers of the two arms "
                "are in their own units and are not comparable, while the rank "
                "numbers (the rank percentiles) are.",
            ],
        },
        "models": list(models),
        "total_seconds": round(total_seconds, 3),
    }
