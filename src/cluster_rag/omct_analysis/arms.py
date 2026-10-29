"""The arms: the ladder's variants and the comparison against the baseline.

An **arm** is one way of scoring a pair (the full text, the atoms, a word
window, a late-chunked side, a swept threshold). This module scores them,
measures each one's discrimination against the control side, and compares
two arms with the paired bootstrap.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cluster_rag.embed.base import Embedder
from cluster_rag.eval.metrics import (
    paired_auc_bootstrap,
    paired_within_group_auc_bootstrap,
    roc_auc,
)
from cluster_rag.omct_analysis.helpers import (
    BASELINE_ARM,
    FULL_TEXT_AGGREGATION,
    HIT_KS,
    SEMANTIC_CRITERION_AUC,
    _auc_decomposition,
    _cell_id,
    _chunker_name,
    _chunker_slug,
    _distribution,
    _floor_gap,
    _macro,
    _quantile_delta,
    _safe_auc,
    _top_fifth_share_delta,
    _top_quintile_share,
    length_correlation,
)
from cluster_rag.omct_analysis.view import (
    DatasetView,
    SideEncoding,
    _top1_provenance,
    encode_late_side,
    encode_side,
    side_similarities,
)
from cluster_rag.preprocess import (
    PreprocessSpec,
    apply_question_transform,
    question_transform_version,
)
from cluster_rag.text_processors.aggregation import Aggregation, describe_aggregation
from cluster_rag.text_processors.base import Chunker
from cluster_rag.text_processors.control import percentiles_from_positions, rank_positions
from cluster_rag.text_processors.late_chunking import LATE_POOLING_MEAN

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class LateSide:
    """The **late-chunked** side of a run (Stage 2.7).

    Unlike the early-chunked side, this side's vectors are not the chunks'
    own embeddings: the document is embedded **once** and each chunk's vector
    is pooled out of the document's token-level embeddings, so every chunk
    carries the document's context.

    The driver never builds a chunker or an embedder itself: the CLI wires
    them in (the model clients stay at the edge, AGENTS.md).

    Attributes:
        chunker: The chunker whose character spans the tokens are pooled over.
        paragraph_embedder: The embedder of the paragraph side (it must be
            able to return token-level embeddings).
        question_chunker: The chunker of the **question** side (``None`` = the
            question side stays whole, as the early-chunked reference does).
        question_embedder: The embedder of the late-chunked question side.
        poolings: The pooling rules (each one an arm family of its own).
    """

    chunker: Chunker
    paragraph_embedder: Embedder
    question_chunker: Chunker | None = None
    question_embedder: Embedder | None = None
    poolings: tuple[str, ...] = (LATE_POOLING_MEAN,)

    def slug(self, pooling: str) -> str:
        """The arm-name prefix of one pooling rule (``late_mean_words25``)."""
        return f"late_{pooling}_{_chunker_slug(self.chunker)}"


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


def _late_variant_arms(
    view: DatasetView,
    *,
    side: LateSide,
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
    """The arms of the late-chunked side (one arm per pooling rule and rule).

    The late side's chunks are pooled out of the document's own token-level
    embeddings, so its arms sit **beside** the run's early-chunked arms: the
    late-vs-early comparison is paired by construction (the same cells, the
    same candidate lists, the same control).
    """
    paragraph_encodings = encode_late_side(
        side.paragraph_embedder,
        side.chunker,
        list(paragraph_texts),
        "document",
        poolings=side.poolings,
    )
    question_encodings: dict[str, SideEncoding] = {}
    if side.question_chunker is not None:
        if side.question_embedder is None:
            raise ValueError("a late-chunked question side needs its own embedder")
        question_encodings = encode_late_side(
            side.question_embedder,
            side.question_chunker,
            list(question_texts),
            "query",
            poolings=side.poolings,
        )
    arms: dict[str, Any] = {}
    similarities_by_arm: dict[str, np.ndarray] = {}
    for pooling in side.poolings:
        paragraph_side = paragraph_encodings[pooling]
        question_side = question_encodings.get(pooling, baseline_question_side)
        for rule in aggregations:
            arm_name = f"{side.slug(pooling)}_{rule.slug}"
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
    header_side: ChunkedSide | None = None,
    late_side: LateSide | None = None,
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
        header_side: An optional **second** chunked side whose chunks carry the
            paragraph's own first sentence (the Stage-2.6 chunk header). Its
            arms run beside the plain chunked arms, so the header is measured
            against the no-header arm in the same run.
        late_side: An optional **late-chunked** side (Stage 2.7): the document
            is embedded once and the token-level embeddings are pooled per
            chunk. Its arms run beside the early-chunked arms, so the late
            geometry is measured against the early one in the same run.
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

    if header_side is not None:
        # The chunk header (Stage 2.6): the same aggregation rules over the
        # headered chunks, so the header is measured against the no-header arm
        # of the same run (the pairing is by construction).
        if not aggregations:
            raise ValueError("The header arms need aggregation rules")
        header_arms, header_similarities = _fixed_variant_arms(
            view,
            side=header_side,
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
        arms.update(header_arms)
        arm_similarities.update(header_similarities)

    if late_side is not None:
        # The late chunking (Stage 2.7): the document embedded once, the
        # token-level embeddings pooled per chunk. The arms run beside the
        # run's early-chunked arms, so the late geometry is measured against
        # the early one in the same run (the pairing is by construction).
        if not aggregations:
            raise ValueError("the late-chunked arms need aggregation rules")
        late_arms, late_similarities = _late_variant_arms(
            view,
            side=late_side,
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
        arms.update(late_arms)
        arm_similarities.update(late_similarities)

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
