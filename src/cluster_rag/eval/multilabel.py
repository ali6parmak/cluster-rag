"""Multi-label ranking and decision metrics (Phase 3); pure functions, numpy only.

Conventions (settled in the Phase 3 design review):

- **Ranking**: stable descending sort — ties pick the earlier column (= the
  alphabetically earlier label), the same convention as the Phase 2 pairwise
  metrics in :mod:`cluster_rag.eval.metrics`.
- **Restricted view (per-bucket)**: ``column_mask`` restricts both the ranked
  column space and the gold labels; rows contribute only when at least one of
  their gold labels is in view (goldless rows are counted, not averaged —
  same convention as the pairwise metrics). ``k`` clips to the view size. The
  overall report is the same functions with mask ``None``.
- **Shortlist recall@m stays full-view**: ``cell_mask`` only restricts *which
  gold cells count*, never the ranking — the ceiling question is "does the
  gold label appear anywhere in the record's global top-m".
- **nDCG@k**: binary gains (2^rel - 1), discount log2(rank + 1), ideal DCG
  from ``min(n_gold, k)`` ones. Computed on the raw score ranking, not on a
  decision set.
- **F1 zero-division**: a record/label with no positive prediction and no
  gold scores F1 = 0 (reported, never hidden).
- All functions take plain arrays and return JSON-serializable dicts, with no
  file I/O, model clients or global state (AGENTS.md dependency-direction rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DECISION_RULES: tuple[str, ...] = ("top_k", "score_threshold", "top_k_min_score")


def _as_matrix(scores: np.ndarray) -> np.ndarray:
    """Coerce a score matrix to 2-d float64 and reject non-finite values."""
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"scores must be a 2-d (n_rows, n_labels) matrix, got shape {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("scores must be finite (NaN/inf would corrupt every ranking)")
    return matrix


def _as_gold_matrix(gold: np.ndarray, shape: tuple[int, int], name: str = "gold") -> np.ndarray:
    """Coerce a gold matrix to 2-d bool matching ``shape``."""
    matrix = _as_bool_matrix(gold, name)
    if matrix.shape != shape:
        raise ValueError(f"{name} must be a bool matrix of shape {shape}, got {matrix.shape}")
    return matrix


def _as_bool_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a 2-d bool matrix (shape checked by the caller)."""
    result = np.asarray(matrix, dtype=bool)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2-d matrix, got shape {result.shape}")
    return result


def _view(
    scores: np.ndarray, gold: np.ndarray, column_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict both matrices to ``column_mask`` (None = every column)."""
    if column_mask is None:
        return scores, gold
    mask = np.asarray(column_mask, dtype=bool)
    if mask.shape != (scores.shape[1],):
        raise ValueError(f"column_mask must have shape ({scores.shape[1]},), got {mask.shape}")
    return scores[:, mask], gold[:, mask]


def _ranked_columns(scores: np.ndarray) -> np.ndarray:
    """Per-row column indices, descending by score, stable (earlier column wins ties)."""
    return np.argsort(-scores, axis=1, kind="stable")


def apply_decision_rule(
    scores: np.ndarray,
    *,
    rule: str,
    k: int | None = None,
    threshold: float | None = None,
    min_score: float | None = None,
) -> np.ndarray:
    """Turn a score matrix into the binary decision matrix of one operating point.

    Rules (the Phase 4 thresholding options, defined once in Phase 3):
    - ``top_k``: the top-k ranked labels per row (k clips to the label count).
    - ``score_threshold``: every label with ``score >= threshold``.
    - ``top_k_min_score``: the top-k labels per row, each additionally required
      to pass ``score >= min_score`` (predict at most k, only confident ones).

    The result is a bool matrix of the same shape; it feeds the F1 and
    coverage metrics. Scores below a rule's parameters never predict.

    Raises:
        ValueError: For an unknown rule or parameters the rule requires but
            did not get.
    """
    matrix = _as_matrix(scores)
    if rule not in DECISION_RULES:
        raise ValueError(f"Unknown decision rule {rule!r}; expected one of {DECISION_RULES}")
    n_rows, n_cols = matrix.shape
    decision = np.zeros((n_rows, n_cols), dtype=bool)
    rows = np.arange(n_rows)[:, None]

    if rule == "score_threshold":
        if threshold is None:
            raise ValueError("decision rule 'score_threshold' requires a threshold")
        return matrix >= threshold

    if rule in ("top_k", "top_k_min_score"):
        if k is None or k < 1:
            raise ValueError(f"decision rule {rule!r} requires a positive k")
        k_eff = min(k, n_cols)
        if k_eff == 0:
            return decision
        ranked = _ranked_columns(matrix)
        top_columns = ranked[:, :k_eff]
        if rule == "top_k":
            decision[rows, top_columns] = True
            return decision
        if min_score is None:
            raise ValueError("decision rule 'top_k_min_score' requires min_score")
        top_scores = np.take_along_axis(matrix, top_columns, axis=1)
        decision[rows, top_columns] = top_scores >= min_score
        return decision

    raise ValueError(f"Unknown decision rule {rule!r}; expected one of {DECISION_RULES}")


def ranking_metrics(
    scores: np.ndarray,
    gold: np.ndarray,
    ks: Sequence[int],
    *,
    column_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """P@k, R@k and nDCG@k of the score ranking against a multi-hot gold matrix.

    Per row the ranked labels (descending score, stable ties) are compared
    with the row's gold set inside the (possibly restricted) view; all three
    metrics average over rows with at least one gold label in view:

    - ``precision@k``: gold labels inside the top-k, divided by k (clipped to
      the view size).
    - ``recall@k``: gold labels inside the top-k, divided by the row's gold
      count in view.
    - ``ndcg@k``: binary-gain DCG of the top-k over the ideal ``min(n_gold, k)``
      ranks (gains 2^rel - 1, discount log2(rank + 1)).

    Args:
        scores: ``(n_rows, n_labels)`` matrix; higher = more relevant.
        gold: bool matrix of the same shape.
        ks: k values (each clipped to the view size).
        column_mask: optional bool vector restricting the ranked columns (the
            per-bucket restricted view); ``None`` ranks every column.

    Returns:
        JSON-serializable dict with the three metrics per k plus row counts.
    """
    matrix = _as_matrix(scores)
    truth = _as_gold_matrix(gold, matrix.shape)
    if any(k < 1 for k in ks):
        raise ValueError(f"k values must be positive, got {ks}")
    view_scores, view_gold = _view(matrix, truth, column_mask)
    n_view = view_scores.shape[1]
    gold_counts = view_gold.sum(axis=1).astype(np.float64)
    contributing = gold_counts > 0

    metrics: dict[str, float | int] = {
        "n_rows": int(view_scores.shape[0]),
        "n_rows_with_gold": int(contributing.sum()),
        "n_rows_without_gold": int((~contributing).sum()),
    }
    if not contributing.any():
        for k in ks:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        return metrics

    ranked = _ranked_columns(view_scores)[contributing]
    row_gold = view_gold[contributing]
    gold_count = gold_counts[contributing]

    for k in ks:
        k_eff = min(k, n_view)
        top = ranked[:, :k_eff]
        top_gold = np.take_along_axis(row_gold, top, axis=1)
        tp = top_gold.sum(axis=1).astype(np.float64)
        metrics[f"precision@{k}"] = float(np.mean(tp / k_eff))
        metrics[f"recall@{k}"] = float(np.mean(tp / gold_count))
        # Binary-gain DCG over the ranked top-k; ideal = min(n_gold, k) ones
        # packed into the leading ranks.
        discount = 1.0 / np.log2(np.arange(1, k_eff + 1, dtype=np.float64) + 1.0)
        dcg = (top_gold * discount).sum(axis=1)
        ideal_counts = np.minimum(gold_count, k_eff).astype(np.int64)
        idcg = np.cumsum(discount)[ideal_counts - 1]
        metrics[f"ndcg@{k}"] = float(np.mean(dcg / idcg))
    return metrics


def decision_f1_metrics(
    predicted: np.ndarray,
    gold: np.ndarray,
    *,
    column_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """micro / macro / sample F1 (plus micro precision/recall) of a decision matrix.

    The decision matrix is the output of :func:`apply_decision_rule` (bool,
    same shape as ``gold``). Zero-division convention: a record or label with
    no positive prediction and no gold scores F1 = 0.

    Aggregation over the (possibly restricted) view:

    - ``micro``: TP/FP/FN are summed over every row — including rows whose
      gold lies outside the view, whose in-view predictions then count as
      false positives (restricted-cell semantics, i.e. both matrices are
      masked to the view's columns).
    - ``macro``: per-label F1 averaged over the view labels that have at
      least one gold occurrence (evaluable labels); labels without any gold
      are excluded and counted in ``n_labels_without_gold`` — the same
      "goldless units are counted but not averaged" convention as the
      retrieval metrics.
    - ``sample``: per-row F1, averaged over rows with at least one gold label
      in view (rows without gold there are excluded and counted).

    Returns:
        JSON-serializable dict with all three F1 variants and their counts.
    """
    predicted_matrix = _as_bool_matrix(predicted, "predicted")
    truth = _as_gold_matrix(gold, predicted_matrix.shape)
    view_predicted, view_gold = _view(predicted_matrix, truth, column_mask)
    n_rows = view_predicted.shape[0]
    tp = int((view_predicted & view_gold).sum())
    fp = int((view_predicted & ~view_gold).sum())
    fn = int((~view_predicted & view_gold).sum())
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (2 * tp) / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

    # Macro over evaluable labels (>= 1 gold occurrence in view).
    label_tp = (view_predicted & view_gold).sum(axis=0)
    label_fp = (view_predicted & ~view_gold).sum(axis=0)
    label_fn = (~view_predicted & view_gold).sum(axis=0)
    label_gold = label_tp + label_fn
    evaluable = label_gold > 0
    n_evaluable = int(evaluable.sum())
    if n_evaluable:
        per_label_f1 = np.zeros(view_predicted.shape[1], dtype=np.float64)
        denominator = 2 * label_tp[evaluable] + label_fp[evaluable] + label_fn[evaluable]
        per_label_f1[evaluable] = (2 * label_tp[evaluable]) / denominator
        macro_f1 = float(per_label_f1[evaluable].mean())
    else:
        macro_f1 = 0.0

    # Sample F1 over rows with at least one gold label in view.
    row_gold = view_gold.sum(axis=1)
    contributing = row_gold > 0
    if contributing.any():
        rows = view_predicted[contributing] & view_gold[contributing]
        row_tp = rows.sum(axis=1).astype(np.float64)
        row_fp = (view_predicted[contributing] & ~view_gold[contributing]).sum(axis=1)
        row_fn = row_gold[contributing] - row_tp
        denominator = 2 * row_tp + row_fp + row_fn
        per_row_f1 = np.divide(
            2 * row_tp, denominator, out=np.zeros_like(row_tp), where=denominator > 0
        )
        sample_f1 = float(np.mean(per_row_f1))
        n_rows_with_gold = int(contributing.sum())
    else:
        sample_f1 = 0.0
        n_rows_with_gold = 0

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "sample_f1": float(sample_f1),
        "n_rows": int(n_rows),
        "n_rows_with_gold": n_rows_with_gold,
        "n_rows_without_gold": int((~contributing).sum()),
        "n_labels": int(view_predicted.shape[1]),
        "n_labels_in_macro": n_evaluable,
        "n_labels_without_gold": int((~evaluable).sum()),
        "n_gold_cells": tp + fn,
        "n_predicted_cells": tp + fp,
    }


def shortlist_recall(
    scores: np.ndarray,
    gold: np.ndarray,
    ms: Sequence[int],
    *,
    cell_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Gold-in-shortlist recall@m: is each gold label inside the record's global top-m?

    The ranking is always the **global** one (every column); ``cell_mask``
    only restricts which gold cells are counted, so a bucket's numbers ask
    "do this bucket's gold labels sit anywhere in the record's top-m" — the
    ceiling for any reranker that only sees the top-m shortlist. Averaged
    per row over rows with at least one gold cell in the mask (goldless rows
    are counted, not averaged).

    Args:
        scores: ``(n_rows, n_labels)`` matrix; higher = more relevant.
        gold: bool matrix of the same shape.
        ms: m values (each clipped to the label count).
        cell_mask: optional bool vector selecting which gold columns count
            (the per-bucket restricted *cells*); ``None`` counts every cell.
    """
    matrix = _as_matrix(scores)
    truth = _as_gold_matrix(gold, matrix.shape)
    if any(m < 1 for m in ms):
        raise ValueError(f"m values must be positive, got {ms}")
    if cell_mask is not None:
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != (matrix.shape[1],):
            raise ValueError(f"cell_mask must have shape ({matrix.shape[1]},), got {mask.shape}")
        truth = truth & mask
    n_cols = matrix.shape[1]
    gold_counts = truth.sum(axis=1).astype(np.float64)
    contributing = gold_counts > 0
    ranked = _ranked_columns(matrix)[contributing]
    gold_count = gold_counts[contributing]

    metrics: dict[str, float | int] = {
        "n_rows": int(matrix.shape[0]),
        "n_rows_with_gold": int(contributing.sum()),
        "n_rows_without_gold": int((~contributing).sum()),
        "n_gold_cells": int(truth.sum()),
    }
    if not contributing.any():
        metrics.update({f"recall@{m}": 0.0 for m in ms})
        return metrics
    for m in ms:
        m_eff = min(m, n_cols)
        top = ranked[:, :m_eff]
        top_gold = np.take_along_axis(truth[contributing], top, axis=1)
        hits = top_gold.sum(axis=1).astype(np.float64)
        metrics[f"recall@{m}"] = float(np.mean(hits / gold_count))
    return metrics


def label_coverage(
    predicted: np.ndarray,
    gold: np.ndarray,
    *,
    column_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Share of evaluable labels the predictor ever suggests at its operating point.

    ``predicted`` is the decision matrix (output of :func:`apply_decision_rule`).
    A label counts as covered when it appears in at least one record's
    decision set across the whole evaluated split; the denominator is the
    number of view labels with at least one gold occurrence (evaluable
    labels). Unlike the other metrics this is a predictor-diversity
    diagnostic, so it scans *all* rows, not only gold rows in view.

    Returns:
        Dict with ``coverage`` plus the numerator/denominator counts.
    """
    predicted_matrix = _as_bool_matrix(predicted, "predicted")
    truth = _as_gold_matrix(gold, predicted_matrix.shape)
    view_predicted, view_gold = _view(predicted_matrix, truth, column_mask)
    suggested = view_predicted.any(axis=0)
    gold_present = view_gold.any(axis=0)
    n_gold_present = int(gold_present.sum())
    n_covered = int((suggested & gold_present).sum())
    coverage = n_covered / n_gold_present if n_gold_present else 0.0
    return {
        "coverage": float(coverage),
        "n_labels_covered": n_covered,
        "n_labels_with_gold": n_gold_present,
        "n_labels_suggested": int(suggested.sum()),
        "n_labels_in_view": int(view_predicted.shape[1]),
    }


# --- the paired bootstrap of a metric difference (two scorers, the same texts) ---

# The metrics the paired bootstrap can compare.
PAIRED_METRICS: tuple[str, ...] = ("micro_f1", "sample_f1", "ndcg")


def _per_row_f1_counts(
    predicted: np.ndarray, truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every row's (true positives, false positives, false negatives)."""
    tp = (predicted & truth).sum(axis=1).astype(np.float64)
    fp = (predicted & ~truth).sum(axis=1).astype(np.float64)
    fn = (~predicted & truth).sum(axis=1).astype(np.float64)
    return tp, fp, fn


def _per_row_ndcg(scores: np.ndarray, truth: np.ndarray, k: int) -> np.ndarray:
    """Every row's nDCG@k (binary gains, the ideal from the row's gold count)."""
    matrix = _as_matrix(scores)
    n_labels = matrix.shape[1]
    k_eff = min(k, n_labels)
    order = _ranked_columns(matrix)
    top_gold = np.take_along_axis(truth, order[:, :k_eff], axis=1)
    gold_counts = truth.sum(axis=1)
    discount = 1.0 / np.log2(np.arange(1, k_eff + 1, dtype=np.float64) + 1.0)
    dcg = (top_gold * discount).sum(axis=1)
    ideal_counts = np.minimum(gold_counts, k_eff).astype(np.int64)
    idcg = np.cumsum(discount)[ideal_counts - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        values = np.where(idcg > 0, dcg / idcg, 0.0)
    values = np.where(gold_counts > 0, values, np.nan)
    return values


def _micro_f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """The micro-F1 of a set of rows, from their per-row counts."""
    total_tp = tp.sum(axis=0)
    total_fp = fp.sum(axis=0)
    total_fn = fn.sum(axis=0)
    denominator = 2 * total_tp + total_fp + total_fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * total_tp / denominator, 0.0)


def _sample_f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """The per-row F1 of a set of rows, from their per-row counts."""
    denominator = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, 2 * tp / denominator, 0.0)


def paired_metric_bootstrap(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    gold: np.ndarray,
    *,
    metric: str = "micro_f1",
    k: int = 5,
    decision: Mapping[str, Any] | None = None,
    resamples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """The **paired** bootstrap of a metric difference between two scorers.

    Two scorers are compared on the **same texts**: the resampling unit is
    the text (the row), so the interval accounts for the texts' own
    difficulty and the difference is paired (the track's discipline: no
    verdict rests on two independent numbers).

    The metric is one of:

    - ``micro_f1`` / ``sample_f1`` — the decision view. The decision matrices
      come from ``decision`` (a mapping with ``rule`` and the rule's
      parameters, as in the ``eval.decision`` config); without it the scores
      are used as they are (a threshold-free decision is not implied — the
      caller must pass a rule).
    - ``ndcg`` — the ranking view at ``k`` (the score ranking, no decision).

    Args:
        left_scores: The left scorer's ``(n_texts, n_labels)`` score matrix.
        right_scores: The right scorer's score matrix (the same shape).
        gold: The bool gold matrix of the same shape.
        metric: The metric to compare (see above).
        k: The k of the nDCG (the ranking view).
        decision: The decision rule (the F1 metrics need one).
        resamples: The number of bootstrap resamples.
        seed: The seed of the generator (the statistic is reproducible).
        confidence: The interval's coverage (0.95 → the 2.5 / 97.5
            percentiles).

    Returns:
        A dict with the two point estimates, their difference, the interval
        and the resampling's provenance.

    Raises:
        ValueError: If the metric is unknown, the shapes disagree, the
            decision rule is missing for an F1 metric, or the resample count
            is not positive.
    """
    if metric not in PAIRED_METRICS:
        raise ValueError(
            f"Unknown paired metric {metric!r}; expected one of {list(PAIRED_METRICS)}"
        )
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")
    left = _as_matrix(left_scores)
    right = _as_matrix(right_scores)
    if left.shape != right.shape:
        raise ValueError(
            f"the two score matrices must share one shape, got {left.shape} and {right.shape}"
        )
    truth = _as_gold_matrix(gold, left.shape)
    n_rows = left.shape[0]
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if metric == "ndcg":
        left_values = _per_row_ndcg(left, truth, k)
        right_values = _per_row_ndcg(right, truth, k)
    else:
        if decision is None:
            raise ValueError(
                f"the paired {metric} needs a decision rule (the mapping with 'rule' "
                "and the rule's parameters)"
            )
        left_decision = apply_decision_rule(
            left,
            rule=decision["rule"],
            k=decision.get("k"),
            threshold=decision.get("threshold"),
            min_score=decision.get("min_score"),
        )
        right_decision = apply_decision_rule(
            right,
            rule=decision["rule"],
            k=decision.get("k"),
            threshold=decision.get("threshold"),
            min_score=decision.get("min_score"),
        )
        left_values = _per_row_f1_counts(left_decision, truth)
        right_values = _per_row_f1_counts(right_decision, truth)

    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        rows = generator.integers(0, n_rows, size=n_rows)
        if metric == "ndcg":
            left_value = float(np.nanmean(left_values[rows]))
            right_value = float(np.nanmean(right_values[rows]))
        elif metric == "micro_f1":
            left_value = float(_micro_f1_from_counts(*(value[rows] for value in left_values)))
            right_value = float(_micro_f1_from_counts(*(value[rows] for value in right_values)))
        else:
            left_value = float(
                np.mean(_sample_f1_from_counts(*(value[rows] for value in left_values)))
            )
            right_value = float(
                np.mean(_sample_f1_from_counts(*(value[rows] for value in right_values)))
            )
        differences[index] = left_value - right_value

    if metric == "ndcg":
        left_point = float(np.nanmean(left_values))
        right_point = float(np.nanmean(right_values))
    elif metric == "micro_f1":
        left_point = float(_micro_f1_from_counts(*left_values))
        right_point = float(_micro_f1_from_counts(*right_values))
    else:
        left_point = float(np.mean(_sample_f1_from_counts(*left_values)))
        right_point = float(np.mean(_sample_f1_from_counts(*right_values)))

    tail = (1.0 - confidence) / 2.0
    quantiles = np.quantile(differences, [tail, 1.0 - tail])
    return {
        "metric": metric,
        "k": k if metric == "ndcg" else None,
        "left": left_point,
        "right": right_point,
        "delta": left_point - right_point,
        "ci_low": float(quantiles[0]),
        "ci_high": float(quantiles[1]),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "n_texts": n_rows,
        "share_positive": float((differences > 0).mean()),
        "note": (
            "the resampling unit is the text: both scorers are judged on the same "
            "resampled texts, so the interval is paired"
        ),
    }
