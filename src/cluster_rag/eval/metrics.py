"""Evaluation metrics (Phases 3+); pure functions, numpy only.

The pairwise-retrieval subset here (:func:`roc_auc`,
:func:`pairwise_retrieval_metrics`, :func:`subset_pair_metrics`) is exercised
by the Phase 2 auxiliary probe. The full P@k/R@k/F1 set arrives in Phase 3.
Everything takes plain arrays and returns JSON-serializable dicts, with no
file I/O, model clients or global state (AGENTS.md dependency-direction rule).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
    # midranks (1-based; tied scores share the mean of their positions)
    order = np.argsort(s, kind="stable")
    sorted_scores = s[order]
    run_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    run_ends = np.r_[run_starts[1:] - 1, s.size - 1]
    ranks = np.empty(s.size, dtype=np.float64)
    for start, end in zip(run_starts, run_ends, strict=True):
        midrank = (start + end) / 2.0 + 1.0  # middle of the 1-based positions
        ranks[order[start : end + 1]] = midrank
    u = ranks[y == 1].sum() - positives * (positives + 1) / 2.0
    return float(u / (positives * negatives))


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
            the ranking (the aux probe's "same document" scope); rows rank only
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

    Used by the aux probe to compare metrics for row subsets: "paragraphs that
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
