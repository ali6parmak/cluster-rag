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


def grouped_auc_decomposition(
    labels: Sequence[int],
    scores: Sequence[float],
    groups: Sequence[str],
) -> dict[str, Any]:
    """Split a pooled AUC into its within-group and its cross-group part.

    A pooled AUC compares **every** positive with **every** negative, so the
    pairs of two different groups (here: two different documents) are part of
    it. On the text-processing track's OMCT data 58.7% of the positive-negative
    pairs are cross-document, so a method that changes a document's score
    *level* moves the pooled number without changing the ranking inside a
    document. This decomposition separates the two: the **within-document**
    AUC (the ranking inside one document) and the **cross-document** AUC (the
    ranking across documents). Ties contribute 0.5, as in :func:`roc_auc`.

    Args:
        labels: The 0/1 labels of the rows.
        scores: The scores of the rows.
        groups: The group (the document) of every row.

    Returns:
        A JSON-serializable dict with the two AUCs, the number of pairs of
        each kind and their shares of all positive-negative pairs.

    Raises:
        ValueError: If the lengths differ or a class is missing.
    """
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray([str(group) for group in groups])
    if not (y.size == s.size == g.size):
        raise ValueError("labels, scores and groups must have the same length")
    positives = y == 1
    negatives = y == 0
    if not positives.any() or not negatives.any():
        raise ValueError("AUC is undefined unless both label classes are present")

    pos_scores = s[positives][:, None]
    neg_scores = s[negatives][None, :]
    same_group = g[positives][:, None] == g[negatives][None, :]
    credit = (pos_scores > neg_scores).astype(np.float64) + 0.5 * (pos_scores == neg_scores).astype(
        np.float64
    )
    n_within = int(same_group.sum())
    n_cross = int((~same_group).sum())
    total = n_within + n_cross
    return {
        "within_group_auc": (float(credit[same_group].sum() / n_within) if n_within else None),
        "cross_group_auc": (float(credit[~same_group].sum() / n_cross) if n_cross else None),
        "n_within_group_pairs": n_within,
        "n_cross_group_pairs": n_cross,
        "within_group_pair_share": float(n_within / total) if total else None,
        "cross_group_pair_share": float(n_cross / total) if total else None,
        "n_groups": int(len(set(g.tolist()))),
        "note": (
            "the pooled AUC compares every positive with every negative; the "
            "within-group part is the ranking inside one group (here: one "
            "document) and the cross-group part the ranking across groups"
        ),
    }


def _within_group_auc(
    y: np.ndarray, s: np.ndarray, g: np.ndarray, group_values: Sequence[str]
) -> float | None:
    """The pooled within-group AUC: the pair-count-weighted mean of the per-group AUCs.

    Every positive-negative pair of the **same** group counts, so the result is
    the pair-count-weighted average of the groups' own AUCs.
    """
    weighted = 0.0
    total = 0
    for group in group_values:
        mask = g == group
        y_group = y[mask]
        s_group = s[mask]
        positives = int((y_group == 1).sum())
        negatives = int((y_group == 0).sum())
        if positives == 0 or negatives == 0:
            continue
        pairs = positives * negatives
        weighted += _auc(y_group, s_group) * pairs
        total += pairs
    if total == 0:
        return None
    return float(weighted / total)


def paired_within_group_auc_bootstrap(
    labels: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    groups: Sequence[str],
    *,
    n_resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap of the **within-group** AUC difference.

    The resampling is paired (the same rows for both scorers) and stratified
    by (group, class), so every group keeps its size and its class balance in
    every resample — the statistic a segmentation method is judged on is the
    ranking *inside* a group, and a pooled AUC that mixes the groups' score
    levels moves for reasons that have nothing to do with the method (see
    :func:`grouped_auc_decomposition`).

    Args:
        labels: The 0/1 labels of the rows.
        scores_a: The scores of the reference scorer (the baseline arm).
        scores_b: The scores of the compared scorer (the method arm).
        groups: The group (the document) of every row.
        n_resamples: The number of bootstrap resamples.
        seed: The seed of the numpy generator (recorded in the artifact).
        confidence: The confidence level of the percentile interval.

    Returns:
        A JSON-serializable dict with both within-group AUCs, the difference,
        its percentile confidence interval, ``p_delta_positive``, the number of
        resamples and the seed.

    Raises:
        ValueError: If the lengths differ or a class is missing.
    """
    y = np.asarray(labels, dtype=np.int64)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    g = np.asarray([str(group) for group in groups])
    if not (y.size == a.size == b.size == g.size):
        raise ValueError("labels, both score vectors and the groups must have the same length")
    if not (y == 1).any() or not (y == 0).any():
        raise ValueError("AUC is undefined unless both label classes are present")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    group_values = sorted(set(g.tolist()))
    strata: list[tuple[np.ndarray, np.ndarray]] = []
    for group in group_values:
        for label in (1, 0):
            rows = np.flatnonzero((g == group) & (y == label))
            if rows.size:
                strata.append((label, rows))  # type: ignore[arg-type]

    deltas = np.empty(n_resamples, dtype=np.float64)
    reference_auc = _within_group_auc(y, a, g, group_values)
    arm_auc = _within_group_auc(y, b, g, group_values)
    rng = np.random.default_rng(seed)
    for index in range(n_resamples):
        rows = np.concatenate(
            [rng.choice(members, size=members.size, replace=True) for _, members in strata]
        )
        resampled_labels = y[rows]
        resampled_groups = g[rows]
        deltas[index] = (
            _within_group_auc(resampled_labels, b[rows], resampled_groups, group_values) or 0.0
        ) - (_within_group_auc(resampled_labels, a[rows], resampled_groups, group_values) or 0.0)

    alpha = (1.0 - confidence) / 2.0
    return {
        "auc_reference": reference_auc,
        "auc_arm": arm_auc,
        "delta_auc": (
            float(arm_auc - reference_auc)
            if arm_auc is not None and reference_auc is not None
            else None
        ),
        "ci_low": float(np.quantile(deltas, alpha)),
        "ci_high": float(np.quantile(deltas, 1.0 - alpha)),
        "p_delta_positive": float((deltas > 0).mean()),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "confidence": float(confidence),
        "strata": [f"{group}/{label}" for group, label in strata],
        "method": (
            "paired bootstrap stratified by (group, class) of the within-group "
            "AUC difference (arm - reference)"
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
