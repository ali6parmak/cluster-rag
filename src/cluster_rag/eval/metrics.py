"""Evaluation metrics (Phases 3+); pure functions, numpy only.

The pairwise-retrieval subset here (:func:`roc_auc`,
:func:`pairwise_retrieval_metrics`, :func:`subset_pair_metrics`) is exercised
by the Phase 2 auxiliary probe. The full P@k/R@k/F1 set arrives in Phase 3.
Everything takes plain arrays and returns JSON-serializable dicts, with no
file I/O, model clients or global state (AGENTS.md dependency-direction rule).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _midranks(scores: np.ndarray) -> np.ndarray:
    """Tie-aware midranks (1-based): tied scores share the mean of their positions."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.flatnonzero(np.r_[True, sorted_scores[1:] != sorted_scores[:-1]])
    ends = np.r_[starts[1:], sorted_scores.size] - 1
    midranks = (starts + ends) / 2.0 + 1.0
    ranks_sorted = np.repeat(midranks, ends - starts + 1)
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def _auc(y: np.ndarray, scores: np.ndarray) -> float:
    """The AUC of two arrays (both classes present); the tie-aware U statistic."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    ranks = _midranks(scores)
    u = ranks[y == 1].sum() - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve via the tie-aware (Mann-Whitney U) rank statistic.

    ``labels`` are 0/1s, ``scores`` are relevance scores to compare; larger
    scores should indicate label 1. Ties contribute 0.5 via shared midranks.

    Raises:
        ValueError: If both classes are not present (AUC undefined).
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size != s.size:
        raise ValueError("labels and scores must have the same length")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("AUC is undefined unless both label classes are present")
    return _auc(y, s)


def paired_auc_bootstrap(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap of the AUC difference between two scorers.

    The resampling is **paired**: the same resampled pair indices are used for
    both scorers, so the difference is measured on identical pairs (the
    track's rule — TEXT_PROCESSING_PLAN.md section 4: no verdict rests on two
    independent AUCs). The resampling is stratified by class (the positives
    and the negatives are drawn separately), so every resample holds both
    classes and the AUC is always defined.

    Args:
        labels: The 0/1 labels of the pairs.
        scores_a: The scores of the reference scorer (the baseline arm).
        scores_b: The scores of the compared scorer (the method arm).
        n_resamples: The number of bootstrap resamples.
        seed: The seed of the numpy generator (recorded in the artifact).
        confidence: The confidence level of the percentile interval.

    Returns:
        A JSON-serializable dict with both AUCs, the difference
        (``auc_arm - auc_reference``), its percentile confidence interval,
        ``p_delta_positive`` (the share of resamples with a positive
        difference), the number of resamples, the seed and the method.

    Raises:
        ValueError: If the lengths differ or a class is missing.
    """
    y = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if not (y.size == a.size == b.size):
        raise ValueError("labels and both score vectors must have the same length")
    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    if positives.size == 0 or negatives.size == 0:
        raise ValueError("AUC is undefined unless both label classes are present")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    deltas = np.empty(n_resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index in range(n_resamples):
        rows = np.concatenate(
            [
                rng.choice(positives, size=positives.size, replace=True),
                rng.choice(negatives, size=negatives.size, replace=True),
            ]
        )
        resampled_labels = y[rows]
        deltas[index] = _auc(resampled_labels, b[rows]) - _auc(resampled_labels, a[rows])

    alpha = (1.0 - confidence) / 2.0
    return {
        "auc_reference": _auc(y, a),
        "auc_arm": _auc(y, b),
        "delta_auc": float(_auc(y, b) - _auc(y, a)),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "p_delta_positive": float((deltas > 0).mean()),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "method": (
            "stratified paired bootstrap over the pairs; the percentile interval "
            "of the AUC difference (arm - reference)"
        ),
    }


def pairwise_retrieval_metrics(
    similarities: np.ndarray,
    gold: Sequence[set[int]],
    ks: Sequence[int],
    *,
    candidate_columns: Sequence[Sequence[int]] | None = None,
) -> dict[str, float | int]:
    """Question-retrieval quality for paragraph rows against candidate columns.

    For each row (paragraph) the column ranking (descending similarity, ties
    resolved toward earlier columns) is compared with its gold column set,
    and averaged over rows that have a non-empty gold set:

    - ``hit@k``: any gold question ranks within the top-k (success@k).
    - ``recall@k``: the fraction of gold questions inside the top-k.

    Rows without gold questions are excluded from the averages and counted in
    ``n_rows_without_gold`` (they carry no retrieval information).

    Args:
        similarities: ``(n_paragraphs, n_questions)`` cosine-similarity matrix.
        gold: One set of gold column indices per row (may be empty).
        ks: k values for both metrics (clipped at the candidate count).
        candidate_columns: Optional per-row candidate column indices restricting
            the ranking (the omct probe's "same document" scope); rows rank only
            within their candidate list. An empty per-row list ranks nothing.
            Defaults to every column.

    Returns:
        JSON-serializable dict with both metrics per k plus row counts.
    """
    n_questions = similarities.shape[1]
    full_range: np.ndarray = np.arange(n_questions)
    metrics: dict[str, float | int] = {
        "n_rows": similarities.shape[0],
        "n_rows_with_gold": sum(1 for gold_set in gold if gold_set),
        "n_rows_without_gold": sum(1 for gold_set in gold if not gold_set),
    }
    hits: dict[int, list[float]] = {k: [] for k in ks}
    recalls: dict[int, list[float]] = {k: [] for k in ks}
    for row_index, row in enumerate(similarities):
        gold_set = gold[row_index]
        if not gold_set:
            continue
        columns = (
            np.asarray(candidate_columns[row_index], dtype=np.int64)
            if candidate_columns is not None
            else full_range
        )
        ranked = columns[np.argsort(-row[columns], kind="stable")]  # descending, stable ties
        for k in ks:
            top = set(ranked[: min(k, columns.size)].tolist())
            overlap = len(gold_set & top)
            hits[k].append(1.0 if overlap else 0.0)
            recalls[k].append(overlap / len(gold_set))
    for k in ks:
        metrics[f"hit@{k}"] = float(np.mean(hits[k])) if hits[k] else 0.0
        metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    return metrics


def subset_pair_metrics(
    similarities: np.ndarray,
    gold: Sequence[set[int]],
    ks: Sequence[int],
    row_mask: Sequence[bool],
    *,
    bucket: str,
    candidate_columns: Sequence[Sequence[int]] | None = None,
) -> dict[str, float | int | str]:
    """:func:`pairwise_retrieval_metrics` restricted to rows where ``row_mask`` is true.

    Used by the omct probe to compare metrics for row subsets: "paragraphs that
    exceed the model context (truncated by the server)" vs the rest, and (Phase
    2.5) per (document, source_csv) groups with their same-document candidates.

    Args:
        candidate_columns: Optional per-row candidate restriction over the full
            row space (same shape as the probe's same-doc scopes); only the
            masked rows' columns are subset before ranking. ``None`` ranks over
            every column.
    """
    mask = np.asarray(row_mask, dtype=bool)
    kept = np.flatnonzero(mask)
    subset = similarities[mask]
    subset_gold = [gold[index] for index in kept]
    columns = (
        [list(candidate_columns[index]) for index in kept]
        if candidate_columns is not None
        else None
    )
    return {
        "bucket": bucket,
        **pairwise_retrieval_metrics(subset, subset_gold, ks, candidate_columns=columns),
    }
