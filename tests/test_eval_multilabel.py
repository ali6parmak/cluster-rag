"""Hand-computed tests for the Phase 3 multi-label metrics (eval/multilabel.py)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.eval.multilabel import (
    apply_decision_rule,
    decision_f1_metrics,
    label_coverage,
    ranking_metrics,
    shortlist_recall,
)


def test_ranking_metrics_hand_computed() -> None:
    """P@k/R@k/nDCG@k on a 3x4 case with one goldless row."""
    scores = np.array(
        [
            [0.9, 0.8, 0.5, 0.1],  # gold {0,1}: perfect top-2 ranking
            [0.4, 0.6, 0.3, 0.2],  # gold {1}: col1 ranks first
            [1.0, 0.0, 0.0, 0.0],  # no gold -> excluded from averages, counted
        ]
    )
    gold = np.array(
        [
            [True, True, False, False],
            [False, True, False, False],
            [False, False, False, False],
        ]
    )
    metrics = ranking_metrics(scores, gold, ks=(1, 2, 3))
    assert metrics["n_rows"] == 3
    assert metrics["n_rows_with_gold"] == 2
    assert metrics["n_rows_without_gold"] == 1
    assert metrics["precision@1"] == pytest.approx(1.0)
    assert metrics["recall@1"] == pytest.approx(0.75)  # row0 1/2, row1 1/1
    assert metrics["ndcg@1"] == pytest.approx(1.0)
    assert metrics["precision@2"] == pytest.approx(0.75)  # row0 2/2, row1 1/2
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(1.0)
    assert metrics["precision@3"] == pytest.approx(0.5)  # row0 2/3, row1 1/3
    assert metrics["recall@3"] == pytest.approx(1.0)


def test_ranking_metrics_ndcg_partial_credit() -> None:
    """A gold label at rank 2 gets 1/log2(3) of the ideal DCG."""
    scores = np.array([[0.9, 0.8, 0.5]])
    gold = np.array([[False, True, False]])
    metrics = ranking_metrics(scores, gold, ks=(1, 2))
    assert metrics["precision@2"] == pytest.approx(0.5)
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(1.0 / np.log2(3.0))


def test_ranking_metrics_column_mask_restricted_view() -> None:
    """The mask restricts ranking and gold; k clips to the view size."""
    scores = np.array([[0.9, 0.1, 0.8]])
    gold = np.array([[True, False, False]])
    mask = np.array([True, False, True])
    metrics = ranking_metrics(scores, gold, ks=(1, 5), column_mask=mask)
    assert metrics["precision@1"] == pytest.approx(1.0)
    assert metrics["recall@1"] == pytest.approx(1.0)
    # k=5 clips to the 2-column view: 1 of 2 predicted slots is gold
    assert metrics["precision@5"] == pytest.approx(0.5)
    assert metrics["recall@5"] == pytest.approx(1.0)


def test_ranking_metrics_ties_stable_earlier_column() -> None:
    """Tied scores rank the earlier column first (same convention as Phase 2)."""
    scores = np.array([[0.5, 0.5, 0.5, 0.9]])
    gold = np.array([[False, False, False, True]])
    metrics = ranking_metrics(scores, gold, ks=(1,))
    assert metrics["precision@1"] == pytest.approx(1.0)  # col3 is unambiguously first

    tie_first = ranking_metrics(np.array([[0.5, 0.5]]), np.array([[False, True]]), ks=(1,))
    assert tie_first["precision@1"] == pytest.approx(0.0)  # col0 wins the tie
    tie_second = ranking_metrics(np.array([[0.5, 0.5]]), np.array([[True, False]]), ks=(1,))
    assert tie_second["precision@1"] == pytest.approx(1.0)


def test_ranking_metrics_all_goldless_rows() -> None:
    """No gold anywhere: metrics are 0.0 and the rows are counted."""
    metrics = ranking_metrics(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool), ks=(1, 5))
    assert metrics["n_rows_without_gold"] == 2 and metrics["n_rows_with_gold"] == 0
    assert metrics["precision@1"] == 0.0 and metrics["recall@5"] == 0.0


def test_ranking_metrics_rejects_bad_k_and_shapes() -> None:
    """k must be positive; gold must match the score matrix shape."""
    scores = np.array([[0.9, 0.8]])
    gold = np.array([[True, False]])
    with pytest.raises(ValueError, match="k values must be positive"):
        ranking_metrics(scores, gold, ks=(0, 1))
    with pytest.raises(ValueError, match="shape"):
        ranking_metrics(scores, np.array([[True, False, False]]), ks=(1,))


def test_apply_decision_rule_top_k() -> None:
    """top-k predicts exactly the k best columns; ties prefer the earlier one."""
    scores = np.array([[0.5, 0.5, 0.7, 0.1]])
    decision = apply_decision_rule(scores, rule="top_k", k=2)
    assert decision.tolist() == [[True, False, True, False]]  # ranked [2, 0, 1, 3]

    assert apply_decision_rule(scores, rule="top_k", k=10).all()  # k clips to n_cols

    with pytest.raises(ValueError, match="requires a positive k"):
        apply_decision_rule(scores, rule="top_k", k=0)
    with pytest.raises(ValueError, match="requires a positive k"):
        apply_decision_rule(scores, rule="top_k")


def test_apply_decision_rule_score_threshold_inclusive() -> None:
    """``score >= threshold`` decides; the boundary itself predicts."""
    scores = np.array([[0.5, 0.4, 0.9]])
    decision = apply_decision_rule(scores, rule="score_threshold", threshold=0.5)
    assert decision.tolist() == [[True, False, True]]
    with pytest.raises(ValueError, match="requires a threshold"):
        apply_decision_rule(scores, rule="score_threshold")


def test_apply_decision_rule_top_k_min_score() -> None:
    """At most k labels, and only those passing the minimum score."""
    scores = np.array([[0.9, 0.6, 0.2, 0.1]])
    decision = apply_decision_rule(scores, rule="top_k_min_score", k=3, min_score=0.5)
    assert decision.tolist() == [[True, True, False, False]]
    with pytest.raises(ValueError, match="requires min_score"):
        apply_decision_rule(scores, rule="top_k_min_score", k=3)


def test_apply_decision_rule_rejects_unknown_rule_and_nonfinite_scores() -> None:
    """Unknown rule names and NaN/inf scores fail loudly."""
    scores = np.array([[0.9, 0.8]])
    with pytest.raises(ValueError, match="Unknown decision rule"):
        apply_decision_rule(scores, rule="magic")
    with pytest.raises(ValueError, match="finite"):
        apply_decision_rule(np.array([[np.nan, 0.8]]), rule="score_threshold", threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        apply_decision_rule(np.array([[np.inf, 0.8]]), rule="top_k", k=1)


def test_decision_f1_metrics_symmetric_case() -> None:
    """micro/macro/sample all equal 2/3 on a hand-built 2x3 case."""
    predicted = np.array([[True, False, True], [False, True, False]])
    gold = np.array([[True, False, False], [False, True, True]])
    metrics = decision_f1_metrics(predicted, gold)
    assert metrics["micro_precision"] == pytest.approx(2 / 3)  # TP=2, FP=1
    assert metrics["micro_recall"] == pytest.approx(2 / 3)  # TP=2, FN=1
    assert metrics["micro_f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)  # per-label F1s: 1, 1, 0
    assert metrics["sample_f1"] == pytest.approx(2 / 3)  # both rows F1 = 2/3
    assert metrics["n_gold_cells"] == 3 and metrics["n_predicted_cells"] == 3
    assert metrics["n_labels_in_macro"] == 3 and metrics["n_labels_without_gold"] == 0


def test_decision_f1_metrics_zero_division_and_exclusions() -> None:
    """Never-gold labels are excluded from macro (counted); F1 = 0 on zero division."""
    # label1 is predicted but has no gold anywhere -> excluded; label2 is a hit.
    predicted = np.array([[False, True, True]])
    gold = np.array([[False, False, True]])
    metrics = decision_f1_metrics(predicted, gold)
    assert metrics["micro_precision"] == pytest.approx(0.5)  # TP=1, FP=1
    assert metrics["micro_recall"] == pytest.approx(1.0)
    assert metrics["micro_f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(1.0)  # only label2 is evaluable
    assert metrics["n_labels_in_macro"] == 1
    assert metrics["n_labels_without_gold"] == 2
    assert metrics["sample_f1"] == pytest.approx(2 / 3)

    # Never-predicted gold label: recall 0 -> F1 0, still counted.
    metrics2 = decision_f1_metrics(np.array([[True, False]]), np.array([[False, True]]))
    assert metrics2["micro_recall"] == 0.0 and metrics2["micro_f1"] == 0.0
    assert metrics2["macro_f1"] == 0.0 and metrics2["n_labels_in_macro"] == 1


def test_decision_f1_metrics_column_mask_counts_cross_view_predictions() -> None:
    """A prediction whose gold lies outside the view is an in-view false positive."""
    predicted = np.array([[False, True, False], [False, False, False]])
    gold = np.array([[True, False, False], [False, False, False]])
    mask = np.array([False, True, False])
    metrics = decision_f1_metrics(predicted, gold, column_mask=mask)
    assert metrics["micro_precision"] == 0.0 and metrics["micro_f1"] == 0.0
    assert metrics["macro_f1"] == 0.0 and metrics["n_labels_in_macro"] == 0
    assert metrics["n_labels_without_gold"] == 1
    assert metrics["n_rows_with_gold"] == 0 and metrics["n_rows_without_gold"] == 2


def test_shortlist_recall_full_view_gold_cells() -> None:
    """Gold-cell recall over the global ranking, goldless rows counted only."""
    scores = np.array(
        [
            [0.3, 0.9, 0.6, 0.1],  # ranked [1, 2, 0, 3]; gold {0, 2}
            [0.5, 0.5, 0.5, 0.5],  # no gold
        ]
    )
    gold = np.array(
        [
            [True, False, True, False],
            [False, False, False, False],
        ]
    )
    metrics = shortlist_recall(scores, gold, ms=(1, 2, 4))
    assert metrics["n_rows"] == 2
    assert metrics["n_rows_with_gold"] == 1 and metrics["n_rows_without_gold"] == 1
    assert metrics["n_gold_cells"] == 2
    assert metrics["recall@1"] == pytest.approx(0.0)  # top-1 = col1, not gold
    assert metrics["recall@2"] == pytest.approx(0.5)  # col2 appears at rank 2
    assert metrics["recall@4"] == pytest.approx(1.0)  # full ranking contains both golds


def test_shortlist_recall_cell_mask_keeps_global_ranking() -> None:
    """cell_mask restricts counted cells, never the ranking (full-view semantics)."""
    scores = np.array([[0.3, 0.9, 0.6, 0.1]])
    gold = np.array([[True, False, True, False]])
    mask = np.array([True, False, False, False])  # count only col0's gold cell
    metrics = shortlist_recall(scores, gold, ms=(1, 2, 3), cell_mask=mask)
    assert metrics["n_gold_cells"] == 1
    assert metrics["recall@2"] == pytest.approx(0.0)  # col0 ranks 3rd globally
    assert metrics["recall@3"] == pytest.approx(1.0)


def test_shortlist_recall_goldless_rows_and_bad_m() -> None:
    """No gold cells anywhere -> 0.0 recalls; m must be positive."""
    metrics = shortlist_recall(np.ones((2, 2)), np.zeros((2, 2), dtype=bool), ms=(5,))
    assert metrics["n_gold_cells"] == 0 and metrics["recall@5"] == 0.0
    with pytest.raises(ValueError, match="m values must be positive"):
        shortlist_recall(np.ones((1, 2)), np.array([[True, False]]), ms=(0,))


def test_label_coverage_numerator_and_denominator() -> None:
    """Coverage = distinct suggested labels that have gold / distinct gold labels."""
    predicted = np.array([[True, False, True, False], [False, False, True, False]])
    gold = np.array([[True, False, False, True], [False, False, True, False]])
    metrics = label_coverage(predicted, gold)
    # suggested = {0, 2}; gold-present = {0, 2, 3}; covered = {0, 2}
    assert metrics["coverage"] == pytest.approx(2 / 3)
    assert metrics["n_labels_covered"] == 2
    assert metrics["n_labels_with_gold"] == 3
    assert metrics["n_labels_suggested"] == 2
    assert metrics["n_labels_in_view"] == 4

    # No gold at all (e.g. a bucket-less view): coverage 0.0, not a crash.
    empty = label_coverage(predicted, np.zeros((2, 4), dtype=bool))
    assert empty["coverage"] == 0.0 and empty["n_labels_with_gold"] == 0
    assert empty["n_labels_suggested"] == 2  # suggestions are still counted
