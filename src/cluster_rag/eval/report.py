"""Assemble the Phase 3 label-evaluation report: overall + per-bucket sections.

Pure assembly on top of :mod:`cluster_rag.eval.multilabel`: given the score
matrix, the gold matrix and a decision matrix (the predictor's operating
point), produce one JSON-serializable dict per bucket (restricted view: the
bucket's columns, rows with >= 1 gold in view, k clipped to the view size)
plus the overall section (no restriction — same code path). Every number
carries its counts. No file I/O, no config knowledge — the caller owns the
decision rule, thresholds and bucket mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from cluster_rag.eval.buckets import BUCKET_NAMES
from cluster_rag.eval.multilabel import (
    decision_f1_metrics,
    label_coverage,
    ranking_metrics,
    shortlist_recall,
)

_RANK_COUNTS = ("n_rows", "n_rows_with_gold", "n_rows_without_gold")
_F1_SUBDICT = (
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_f1",
    "sample_f1",
    "n_labels_in_macro",
    "n_labels_without_gold",
    "n_predicted_cells",
)


def _validate_inputs(
    scores: np.ndarray,
    gold: np.ndarray,
    labels: Sequence[str],
    label_buckets: Mapping[str, str],
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check shapes, label uniqueness and bucket coverage; return coerced matrices."""
    score_matrix = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(gold, dtype=bool)
    predictions = np.asarray(predicted, dtype=bool)
    if score_matrix.shape != truth.shape or predictions.shape != truth.shape:
        raise ValueError(
            f"scores/gold/predicted must share one (n_rows, n_labels) shape, got "
            f"{score_matrix.shape} / {truth.shape} / {predictions.shape}"
        )
    if len(labels) != truth.shape[1]:
        raise ValueError(f"labels ({len(labels)}) must match the gold columns ({truth.shape[1]})")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique (the column order convention)")
    unknown = [label for label in labels if label not in label_buckets]
    if unknown:
        raise ValueError(
            f"{len(unknown)} label(s) missing from label_buckets, e.g. {unknown[:3]!r}; "
            "every label needs a bucket"
        )
    bad_buckets = sorted(
        {bucket for bucket in label_buckets.values() if bucket not in BUCKET_NAMES}
    )
    if bad_buckets:
        raise ValueError(f"unknown bucket name(s) {bad_buckets}; expected {BUCKET_NAMES}")
    return score_matrix, truth, predictions


def _mask_of(labels: Sequence[str], label_buckets: Mapping[str, str], bucket: str) -> np.ndarray:
    """Column mask selecting one bucket's labels."""
    return np.asarray([label_buckets[label] == bucket for label in labels], dtype=bool)


def _section(
    bucket: str,
    scores: np.ndarray,
    gold: np.ndarray,
    predicted: np.ndarray,
    *,
    column_mask: np.ndarray | None,
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
) -> dict[str, Any]:
    """One report section: restricted-view metrics plus their counts."""
    ranking = ranking_metrics(scores, gold, ks, column_mask=column_mask)
    f1 = decision_f1_metrics(predicted, gold, column_mask=column_mask)
    shortlist = shortlist_recall(scores, gold, shortlist_ms, cell_mask=column_mask)
    coverage = label_coverage(predicted, gold, column_mask=column_mask)

    view_labels = int(f1["n_labels"])  # ranking/f1/coverage share the same view size
    return {
        "bucket": bucket,
        "n_labels": view_labels,
        "n_labels_with_gold": int(coverage["n_labels_with_gold"]),
        "n_records": int(ranking["n_rows"]),
        "n_rows_with_gold": int(ranking["n_rows_with_gold"]),
        "n_rows_without_gold": int(ranking["n_rows_without_gold"]),
        "n_gold_cells": int(f1["n_gold_cells"]),
        "ranking": {key: value for key, value in ranking.items() if key not in _RANK_COUNTS},
        "f1": {key: f1[key] for key in _F1_SUBDICT},
        "coverage": {
            "coverage": coverage["coverage"],
            "n_labels_covered": int(coverage["n_labels_covered"]),
            "n_labels_suggested": int(coverage["n_labels_suggested"]),
        },
        "shortlist": {key: value for key, value in shortlist.items() if key.startswith("recall@")},
    }


def build_label_report(
    scores: np.ndarray,
    gold: np.ndarray,
    labels: Sequence[str],
    label_buckets: Mapping[str, str],
    predicted: np.ndarray,
    *,
    ks: Sequence[int],
    shortlist_ms: Sequence[int],
) -> dict[str, Any]:
    """Build the full per-bucket label-evaluation report from one prediction.

    Args:
        scores: ``(n_records, n_labels)`` score matrix (higher = more relevant).
        gold: bool matrix of the same shape; one True per gold label.
        labels: Ordered label names (the column convention of the run).
        label_buckets: ``label -> bucket`` from
            :func:`cluster_rag.eval.buckets.assign_label_buckets` (every label
            must be mapped, names must be valid).
        predicted: bool decision matrix (e.g. from
            :func:`cluster_rag.eval.multilabel.apply_decision_rule`); feeds
            the F1 and coverage metrics.
        ks: k values for precision/recall/nDCG (e.g. 1, 3, 5, 10).
        shortlist_ms: m values for the full-view gold-cell shortlist recall
            (e.g. 20, 50, 100).

    Returns:
        Dict with an ``overall`` section (no restriction) and one section per
        bucket under ``per_bucket`` (head / torso / tail / unseen), each with
        its ranking, F1, coverage and shortlist metrics plus counts.
    """
    score_matrix, truth, predictions = _validate_inputs(
        scores, gold, labels, label_buckets, predicted
    )
    report = {
        "overall": _section(
            "overall",
            score_matrix,
            truth,
            predictions,
            column_mask=None,
            ks=ks,
            shortlist_ms=shortlist_ms,
        ),
        "per_bucket": {},
    }
    for bucket in BUCKET_NAMES:
        mask = _mask_of(labels, label_buckets, bucket)
        if not mask.any():
            continue  # a bucket with no labels simply has no section
        report["per_bucket"][bucket] = _section(
            bucket,
            score_matrix,
            truth,
            predictions,
            column_mask=mask,
            ks=ks,
            shortlist_ms=shortlist_ms,
        )
    return report
