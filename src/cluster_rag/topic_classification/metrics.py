"""Ranking metrics for multi-label shortlist evaluation (EURLEX-4K track).

The dataset has ~4,271 labels, so micro-F1@2 is the wrong instrument.  The
track's primary metrics are:

- **recall@50** — shortlist breadth: the share of a document's gold labels that
  appear in its top 50;
- **R-precision** — cardinality-aware precision: precision at ``|gold|``
  (equivalently, recall at ``|gold|``), averaged over documents;
- **nDCG@10** — ordering at the top, with binary relevance.

The secondary metrics are recall@5/10/20/100 and nDCG@50.

All functions are pure (numpy + the standard library), take a dense
``(n_documents, n_labels)`` score matrix and a per-document set of gold label
indices, and return JSON-serializable numbers.  Ties are broken by the label's
column order (a stable descending sort), so the result is deterministic.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DEFAULT_RECALL_KS: tuple[int, ...] = (5, 10, 20, 50, 100)
DEFAULT_NDCG_KS: tuple[int, ...] = (10, 50)


def ranked_labels(scores: np.ndarray) -> np.ndarray:
    """Rank every row's labels by descending score (stable tie-break)."""
    matrix = np.asarray(scores)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_documents, n_labels); got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("scores contain non-finite values (NaN/inf)")
    return np.argsort(-matrix, axis=1, kind="stable")


def _validate(gold: Sequence[set[int]], n_documents: int, n_labels: int) -> None:
    if len(gold) != n_documents:
        raise ValueError(f"{len(gold)} gold sets for {n_documents} score rows")
    for labels in gold:
        if not labels:
            raise ValueError("every document must have at least one gold label")
        if any(label < 0 or label >= n_labels for label in labels):
            raise ValueError(f"gold label index outside [0, {n_labels})")


def per_document_r_precision(ranked: np.ndarray, gold: Sequence[set[int]]) -> np.ndarray:
    """Per-document precision at ``k = |gold|`` (the paired bootstrap's unit)."""
    ranked = np.asarray(ranked)
    if len(gold) != ranked.shape[0]:
        raise ValueError(f"{len(gold)} gold sets for {ranked.shape[0]} ranked rows")
    values = np.empty(len(gold), dtype=np.float64)
    for row, labels in enumerate(gold):
        if not labels:
            raise ValueError("every document must have at least one gold label")
        k = min(len(labels), ranked.shape[1])
        top = set(ranked[row, :k].tolist())
        values[row] = len(top & labels) / len(labels)
    return values


def per_document_ndcg_at_k(ranked: np.ndarray, gold: Sequence[set[int]], k: int) -> np.ndarray:
    """Per-document nDCG@k with binary relevance (the paired bootstrap's unit)."""
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    ranked = np.asarray(ranked)
    if len(gold) != ranked.shape[0]:
        raise ValueError(f"{len(gold)} gold sets for {ranked.shape[0]} ranked rows")
    values = np.empty(len(gold), dtype=np.float64)
    for row, labels in enumerate(gold):
        if not labels:
            raise ValueError("every document must have at least one gold label")
        top = ranked[row, : min(k, ranked.shape[1])].tolist()
        dcg = sum(
            1.0 / np.log2(position + 2) for position, label in enumerate(top) if label in labels
        )
        ideal = min(k, len(labels))
        idcg = sum(1.0 / np.log2(position + 2) for position in range(ideal))
        values[row] = dcg / idcg if idcg else 0.0
    return values


def recall_at_k(ranked: np.ndarray, gold: Sequence[set[int]], k: int) -> float:
    """Macro recall@k: mean over documents of ``|top-k ∩ gold| / |gold|``."""
    return float(np.mean(per_document_recall_at_k(ranked, gold, k)))


def r_precision(ranked: np.ndarray, gold: Sequence[set[int]]) -> float:
    """Macro R-precision: mean precision at ``k = |gold|`` over documents."""
    return float(np.mean(per_document_r_precision(ranked, gold)))


def ndcg_at_k(ranked: np.ndarray, gold: Sequence[set[int]], k: int) -> float:
    """Macro nDCG@k with binary relevance and the standard log2 discount."""
    return float(np.mean(per_document_ndcg_at_k(ranked, gold, k)))


def per_document_recall_at_k(ranked: np.ndarray, gold: Sequence[set[int]], k: int) -> np.ndarray:
    """Per-document ``|top-k ∩ gold| / |gold|`` (the paired bootstrap's unit).

    ``recall_at_k`` averages these values; keeping the vector lets a paired
    bootstrap resample documents and difference two methods on the same rows.

    Args:
        ranked: ``(n_documents, n_labels)`` label indices, best first.
        gold: One non-empty set of gold label indices per document.
        k: Cut-off.

    Returns:
        The ``(n_documents,)`` float64 recall values.

    Raises:
        ValueError: If ``k`` is not positive or a document has no gold label.
    """
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    ranked = np.asarray(ranked)
    if len(gold) != ranked.shape[0]:
        raise ValueError(f"{len(gold)} gold sets for {ranked.shape[0]} ranked rows")
    values = np.empty(len(gold), dtype=np.float64)
    for row, labels in enumerate(gold):
        if not labels:
            raise ValueError("every document must have at least one gold label")
        top = set(ranked[row, : min(k, ranked.shape[1])].tolist())
        values[row] = len(top & labels) / len(labels)
    return values


def paired_bootstrap_ci(
    differences: np.ndarray,
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile bootstrap of the mean per-document difference.

    The resampling unit is the **document**: the same row indices are drawn
    for both methods, so the interval is the paired difference, not the
    difference of two independent intervals.

    Args:
        differences: ``(n_documents,)`` per-document differences (treatment
            minus baseline).
        n_resamples: Number of bootstrap resamples (2,000 in the track).
        seed: ``numpy`` generator seed, recorded in the artifact.
        alpha: Two-sided error; the interval is
            ``[percentile(alpha/2), percentile(1 - alpha/2)]``.

    Returns:
        ``{"mean", "ci_low", "ci_high", "n_resamples", "seed", "n_documents"}``.

    Raises:
        ValueError: On an empty vector or a non-positive resample count.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("paired_bootstrap_ci needs at least one document")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be positive, got {n_resamples}")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(n_resamples, values.size))
    means = values[indexes].mean(axis=1)
    low, high = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {
        "mean": round(float(values.mean()), 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "n_documents": int(values.size),
    }


def label_recall_at_k(
    ranked: np.ndarray,
    gold: Sequence[set[int]],
    bucket_of_label: Mapping[int, str],
    k: int,
) -> dict[str, dict[str, float | int]]:
    """Label-level recall@k, split by the label's frequency bucket.

    A gold label is a **hit** when it appears in its document's top-k.  The
    denominator is the number of gold assignments in that bucket, so this is
    micro over labels (it shows whether the shortlist reaches the tail).
    """
    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row, labels in enumerate(gold):
        top = set(ranked[row, : min(k, ranked.shape[1])].tolist())
        for label in labels:
            bucket = bucket_of_label[int(label)]
            totals[bucket] += 1
            hits[bucket] += int(label in top)
    return {
        bucket: {
            "hits": hits[bucket],
            "labels": totals[bucket],
            "recall": round(hits[bucket] / totals[bucket], 6) if totals[bucket] else 0.0,
        }
        for bucket in sorted(totals)
    }


def evaluate_ranked(
    ranked: np.ndarray,
    gold: Sequence[set[int]],
    bucket_of_label: Mapping[int, str],
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
) -> dict[str, Any]:
    """All primary and secondary metrics for an already-ranked label matrix.

    The pipeline's shortlist operations (thresholding, a quota, a reranker) can
    change the order directly instead of producing new scores, so the metrics
    take a ranked matrix as well. ``evaluate_ranking`` is this function after
    ``ranked_labels``.
    """
    ranked = np.asarray(ranked)
    if ranked.ndim != 2:
        raise ValueError(f"ranked must be 2-D (n_documents, n_labels); got {ranked.shape}")
    _validate(gold, ranked.shape[0], ranked.shape[1])
    coverage = float(
        np.mean(
            [
                bool(set(ranked[row, : min(50, ranked.shape[1])].tolist()) & labels)
                for row, labels in enumerate(gold)
            ]
        )
    )
    return {
        "documents": int(ranked.shape[0]),
        "labels": int(ranked.shape[1]),
        "primary": {
            "recall@50": round(recall_at_k(ranked, gold, 50), 6),
            "r_precision": round(r_precision(ranked, gold), 6),
            "ndcg@10": round(ndcg_at_k(ranked, gold, 10), 6),
        },
        "secondary": {
            "recall@5": round(recall_at_k(ranked, gold, 5), 6),
            "recall@10": round(recall_at_k(ranked, gold, 10), 6),
            "recall@20": round(recall_at_k(ranked, gold, 20), 6),
            "recall@100": round(recall_at_k(ranked, gold, 100), 6),
            "ndcg@50": round(ndcg_at_k(ranked, gold, 50), 6),
        },
        "diagnostics": {
            "coverage@50": round(coverage, 6),
            "mean_gold_cardinality": round(float(np.mean([len(labels) for labels in gold])), 3),
            "label_recall@50": label_recall_at_k(ranked, gold, bucket_of_label, 50),
        },
    }


def evaluate_ranking(
    scores: np.ndarray,
    gold: Sequence[set[int]],
    bucket_of_label: Mapping[int, str],
    *,
    recall_ks: Sequence[int] = DEFAULT_RECALL_KS,
    ndcg_ks: Sequence[int] = DEFAULT_NDCG_KS,
) -> dict[str, Any]:
    """All primary and secondary metrics for one score matrix."""
    matrix = np.asarray(scores)
    ranked = ranked_labels(matrix)
    return evaluate_ranked(
        ranked,
        gold,
        bucket_of_label,
        recall_ks=recall_ks,
        ndcg_ks=ndcg_ks,
    )
