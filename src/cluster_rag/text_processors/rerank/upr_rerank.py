"""The rerankers' pure side on UPR (the text-processing track's CS-15).

A reranker can **re-order** a candidate set, never enlarge it, so the track's
question is not "what does the reranker find?" but "**does the reranker's
ordering of an existing candidate set beat the ordering that produced the
set?**". This module holds the pure pieces of that measurement:

- the **candidate set** (every text's top-k labels, in the incumbent's own
  order) and its **ceiling** (what a *perfect* reordering of that set could
  reach — the set's own recall is the binding constraint);
- the **label side** (the catalog's description or the label's own name) and
  the **label-blind counterpart** (the same documents, shifted: the model
  keeps its shape and its cost and loses only the pairing);
- the **rank fusion** (the candidates ordered by the reranker's score, the
  cells outside the set below them in the incumbent's order) and the
  **measurement** of the fused arm.

The measurement and the fusion are the **decision side's own functions**
(:func:`~cluster_rag.text_processors.decision.upr_probe.measure_upr_arm` and
:func:`~cluster_rag.text_processors.decision.upr_listwise.fuse_rankings`),
re-used on purpose: the like-for-like comparison ("the same candidate set,
two orderings") is only meaningful if **both sides are measured by the same
code**.

Pure core: numpy + the standard library only (the track's rule — no torch,
sentence-transformers, httpx or sqlite3). The model client lives in
:mod:`cluster_rag.rerank`, at the edge, and the CLI wires it in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.text_processors.decision.upr_listwise import fuse_rankings
from cluster_rag.text_processors.decision.upr_probe import measure_upr_arm

# The label side's variants (the config's names).
LABEL_DOCUMENT_DESCRIPTION = "description"
LABEL_DOCUMENT_NAME = "name"
LABEL_DOCUMENT_VARIANTS: tuple[str, ...] = (LABEL_DOCUMENT_DESCRIPTION, LABEL_DOCUMENT_NAME)

# The pair order variants (the config's names).
ORDER_TEXT_FIRST = "text_first"
ORDER_LABEL_FIRST = "label_first"
ORDER_VARIANTS: tuple[str, ...] = (ORDER_TEXT_FIRST, ORDER_LABEL_FIRST)


def candidate_columns(scores: np.ndarray, k: int) -> list[tuple[int, ...]]:
    """Every text's candidate columns: its top-k labels, in the incumbent's order.

    Args:
        scores: The incumbent's ``(n_texts, n_labels)`` score matrix.
        k: How many labels per text the candidate set holds.

    Returns:
        The candidate columns of every text (the best first).

    Raises:
        ValueError: If the score matrix is not 2-dimensional.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"the score matrix must be 2-dimensional, got {matrix.shape}")
    take = max(1, min(int(k), matrix.shape[1]))
    order = np.argsort(-matrix, axis=1, kind="stable")
    return [tuple(int(column) for column in row[:take]) for row in order]


def label_documents(
    labels: Sequence[str],
    descriptions: Mapping[str, str] | None,
    variant: str = LABEL_DOCUMENT_DESCRIPTION,
) -> list[str]:
    """The label side's document for every label.

    The **description** is the catalog's text (label-side knowledge, frozen
    before the runs, exactly like the zero-shot label vectors); the **name**
    is the label's own name (the cheap ablation).

    Args:
        labels: The label space (the matrix's column order).
        descriptions: The catalog's descriptions (the label's name when the
            catalog has none).
        variant: ``"description"`` or ``"name"``.

    Returns:
        One document per label, in the label space's order.

    Raises:
        ValueError: If the variant is unknown.
    """
    if variant not in LABEL_DOCUMENT_VARIANTS:
        raise ValueError(
            f"unknown label document variant {variant!r}; expected one of "
            f"{list(LABEL_DOCUMENT_VARIANTS)}"
        )
    documents: list[str] = []
    for label in labels:
        if variant == LABEL_DOCUMENT_NAME:
            documents.append(label)
            continue
        description = (descriptions or {}).get(label) or ""
        documents.append(description or label)
    return documents


def shifted_documents(documents: Sequence[str]) -> list[str]:
    """The **label-blind counterpart**' documents: the same list, cyclically shifted.

    The shift is the decision probe's own construction (a deterministic
    cyclic shift of the label space): the model keeps its shape, its cost and
    its calibration and loses only the pairing — the yardstick that says how
    much of a gain is the model's own label reading.

    Args:
        documents: The label side's documents (in the label space's order).

    Returns:
        The shifted documents (the same length, the same order of positions).
    """
    values = list(documents)
    if not values:
        return []
    return [values[(index + 1) % len(values)] for index in range(len(values))]


def rerank_matrix(
    columns: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    n_labels: int,
) -> np.ndarray:
    """The reranker's scores as a full ``(n_texts, n_labels)`` matrix.

    The cells **outside** the candidate set are ``NaN``: the fusion reads them
    as "not a candidate" (the CS-12 listwise arm's convention).

    Args:
        columns: Every text's candidate columns.
        scores: The reranker's score of every candidate, per text (aligned
            with ``columns``).
        n_labels: The label space's size.

    Returns:
        The score matrix (the same shape as the incumbent's).

    Raises:
        ValueError: If the candidate columns and the scores disagree.
    """
    if len(columns) != len(scores):
        raise ValueError(
            f"the candidate columns and the scores must have the same length, got "
            f"{len(columns)} and {len(scores)}"
        )
    matrix = np.full((len(columns), int(n_labels)), np.nan, dtype=np.float64)
    for row, (row_columns, row_scores) in enumerate(zip(columns, scores, strict=True)):
        if len(row_columns) != len(row_scores):
            raise ValueError(
                f"the candidate columns and the scores of row {row} disagree: "
                f"{len(row_columns)} columns and {len(row_scores)} scores"
            )
        for column, value in zip(row_columns, row_scores, strict=True):
            matrix[row, column] = float(value)
    return matrix


def fuse_candidates(
    incumbent: np.ndarray,
    columns: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
) -> np.ndarray:
    """The fused score matrix: the candidates re-ordered, the rest below them.

    This is the CS-12 listwise arm's **rank fusion**, applied to an explicit
    candidate set: the candidates are ordered by the reranker's score and the
    cells outside the set keep the incumbent's own order below them. The
    result holds the **negative final rank**, so every rank-based metric (the
    decision view's top-k, the nDCG, the recall@k) reads the fused order
    directly.

    Args:
        incumbent: The incumbent's ``(n_texts, n_labels)`` score matrix (the
            ordering that produced the candidate set).
        columns: Every text's candidate columns.
        scores: The reranker's score of every candidate, per text.

    Returns:
        The fused score matrix (the same shape).
    """
    matrix = np.asarray(incumbent, dtype=np.float64)
    reranked = rerank_matrix(columns, scores, matrix.shape[1])
    return fuse_rankings(matrix, reranked)


def fill_restricted_scores(
    columns: Sequence[Sequence[int]],
    scores: Sequence[Sequence[float]],
    n_labels: int,
) -> np.ndarray:
    """The **candidate-restricted** feature: the candidates' scores, the rest at the floor.

    The pre-registered **fill value** for the cells outside the candidate set
    is the **row's minimum scored value** (the row's floor): the fill never
    lifts a non-candidate above a candidate and never manufactures a
    separation the reranker did not produce. A constant fill (0.0) is not
    used: it makes the feature's absolute level arbitrary and lets the
    combiner's fitted weight manufacture the separation.

    Args:
        columns: Every text's candidate columns.
        scores: The reranker's score of every candidate, per text.
        n_labels: The label space's size.

    Returns:
        The filled score matrix (the same shape).

    Raises:
        ValueError: If the candidate columns and the scores disagree.
    """
    if len(columns) != len(scores):
        raise ValueError(
            f"the candidate columns and the scores must have the same length, got "
            f"{len(columns)} and {len(scores)}"
        )
    matrix = np.zeros((len(columns), int(n_labels)), dtype=np.float64)
    for row, (row_columns, row_scores) in enumerate(zip(columns, scores, strict=True)):
        if len(row_columns) != len(row_scores):
            raise ValueError(
                f"the candidate columns and the scores of row {row} disagree: "
                f"{len(row_columns)} columns and {len(row_scores)} scores"
            )
        if not row_scores:
            continue
        floor = float(min(row_scores))
        for column, value in zip(row_columns, row_scores, strict=True):
            matrix[row, column] = float(value)
        outside = np.ones(int(n_labels), dtype=bool)
        outside[list(row_columns)] = False
        matrix[row, outside] = floor
    return matrix


def candidate_ceiling(
    columns: Sequence[Sequence[int]],
    gold: np.ndarray,
    *,
    n_predicted: int,
    labels_by_bucket: Mapping[str, Sequence[str]] | None = None,
    bucket_masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """The candidate set's **ceiling**: what a perfect reordering could reach.

    A perfect reordering puts the gold label first whenever it is inside the
    set, so the true positives are the **gold cells inside the set** and the
    top-k rule's false positives are the predicted cells it did not hit:

        ``TP = (the gold cells inside the set)``, ``FP = n_predicted - TP``,
        ``FN = n_gold - TP``, and the micro-F1 is
        ``2*TP / (2*TP + FP + FN)``.

    The **cell-level** share (the gold cells' share of the set) is the
    headline number, because the decision view's micro-F1 is a cell-level
    statistic; the **row-mean** recall (the ranking view's aggregation of the
    same question) is reported beside it, never mixed with it.

    Args:
        columns: Every text's candidate columns.
        gold: The bool gold matrix (the same shape as the incumbent's).
        n_predicted: How many cells the decision rule predicts (the top-k
            rule's ``k`` per text).
        labels_by_bucket: The label buckets' members (the per-bucket view).
        bucket_masks: The label buckets' column masks (the same, as arrays).

    Returns:
        The ceiling block (the counts, the two aggregations, the micro-F1
        ceiling and the per-bucket coverage).
    """
    truth = np.asarray(gold, dtype=bool)
    n_texts, n_labels = truth.shape
    inside = np.zeros_like(truth, dtype=bool)
    for row, row_columns in enumerate(columns):
        for column in row_columns:
            inside[row, column] = True
    hits = int((inside & truth).sum())
    n_gold = int(truth.sum())
    denominator = 2 * hits + (int(n_predicted) - hits) + (n_gold - hits)
    micro_f1_ceiling = (2 * hits / denominator) if denominator else 0.0

    # The row-mean recall (the other aggregation of the same question).
    gold_counts = truth.sum(axis=1)
    contributing = gold_counts > 0
    row_mean_recall = 0.0
    if contributing.any():
        row_hits = (inside & truth).sum(axis=1)[contributing]
        row_mean_recall = float(np.mean(row_hits / gold_counts[contributing]))

    block: dict[str, Any] = {
        "n_gold_cells": n_gold,
        "n_predicted_cells": int(n_predicted),
        "n_gold_cells_in_set": hits,
        "cell_level_share": (hits / n_gold) if n_gold else 0.0,
        "row_mean_recall": row_mean_recall,
        "micro_f1_ceiling": float(micro_f1_ceiling),
        "n_texts": int(n_texts),
        "n_labels": int(n_labels),
        "note": (
            "the ceiling of the candidate set: the gold cells inside it (the "
            "cell-level share) are the true positives a PERFECT reordering "
            "would reach, and the micro-F1 ceiling follows the top-k rule's "
            "accounting. The row-mean recall is the ranking view's aggregation "
            "of the same question and is reported beside it, never mixed with it."
        ),
    }
    if bucket_masks:
        per_bucket: dict[str, Any] = {}
        for name, mask in bucket_masks.items():
            columns_mask = np.asarray(mask, dtype=bool)
            bucket_gold = truth[:, columns_mask]
            bucket_inside = inside[:, columns_mask]
            bucket_hits = int((bucket_inside & bucket_gold).sum())
            bucket_total = int(bucket_gold.sum())
            per_bucket[name] = {
                "n_gold_cells": bucket_total,
                "n_gold_cells_in_set": bucket_hits,
                "cell_level_share": (bucket_hits / bucket_total) if bucket_total else 0.0,
            }
        block["per_bucket"] = per_bucket
    return block


def measure_reranked_arm(
    fused: np.ndarray,
    gold: np.ndarray,
    *,
    decision: Mapping[str, Any],
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
    bucket_masks: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """The reranked arm's measurement (the harness's own metrics).

    The fused matrix holds **ranks**, not probabilities, so the calibration
    and the probability compression are reported as **not applicable** (the
    CS-12 listwise arm's convention).

    Args:
        fused: The fused score matrix (the negative final rank).
        gold: The bool gold matrix (the same shape).
        decision: The decision rule (the ``eval.decision`` mapping).
        ks: The ranking metrics' k values.
        shortlist_ms: The shortlist recall's m values.
        bucket_masks: The label buckets' column masks (the per-bucket view).

    Returns:
        The measurement block.
    """
    return measure_upr_arm(
        fused,
        gold,
        decision=decision,
        ks=ks,
        shortlist_ms=shortlist_ms,
        bucket_masks=bucket_masks,
        probability_scores=False,
    )
