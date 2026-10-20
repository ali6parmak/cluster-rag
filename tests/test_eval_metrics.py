"""Hand-computed tests for the pairwise metric functions (Phase 2/3 subset)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.eval.metrics import pairwise_retrieval_metrics, roc_auc, subset_pair_metrics


def test_roc_auc_perfect_separation() -> None:
    """All positives above all negatives -> AUC 1; inverted ordering -> 0."""
    assert roc_auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_roc_auc_random_case_and_ties() -> None:
    """Hand-computed midrank cases, including ties contributing 0.5."""
    # ranks: .9->4, .1->1 (positives), U = 5 - 3 = 2, AUC = 2/4 = 0.5
    assert roc_auc([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2]) == pytest.approx(0.5)
    # one tie between a positive and a negative: the tie contributes .5
    # (p.5 vs n.5 = .5, p.5 vs n.0 = 1, p.4 vs n.5 = 0, p.4 vs n.0 = 1 -> 2.5/4)
    assert roc_auc([1, 1, 0, 0], [0.5, 0.4, 0.5, 0.0]) == pytest.approx(0.625)
    # heavy ties everywhere -> 0.5
    assert roc_auc([1, 1, 1, 0, 0, 0], [0.5] * 6) == pytest.approx(0.5)


def test_roc_auc_requires_both_classes() -> None:
    """AUC with one class only is undefined (clear error)."""
    with pytest.raises(ValueError, match="both label classes"):
        roc_auc([1, 1], [0.1, 0.2])


def test_pairwise_retrieval_metrics_hand_computed() -> None:
    """2 questions per paragraph; known rankings."""
    # sim matrix: row0 ranks q1 > q0; row1 ranks q0 > q1
    sims = np.array([[0.2, 0.9], [0.8, 0.3]])
    gold = [{1}, {0}]
    metrics = pairwise_retrieval_metrics(sims, gold, ks=[1, 2])
    assert metrics["n_rows"] == 2 == metrics["n_rows_with_gold"]
    assert metrics["n_rows_without_gold"] == 0
    assert abs(metrics["hit@1"] - 1.0) < 1e-12  # both gold questions rank first
    assert abs(metrics["recall@1"] - 1.0) < 1e-12
    assert abs(metrics["recall@2"] - 1.0) < 1e-12

    # one row has two golds with only one inside the top-1
    sims = np.array([[0.9, 0.8], [0.7, 0.9]])
    gold = [{0}, {0, 1}]
    metrics = pairwise_retrieval_metrics(sims, gold, ks=[1, 2])
    assert abs(metrics["hit@1"] - 1.0) < 1e-12  # row1's top-1 (q1) is in its gold set
    assert abs(metrics["recall@1"] - 0.75) < 1e-12  # row1 keeps 1 of its 2 golds
    assert abs(metrics["hit@2"] - 1.0) < 1e-12
    assert abs(metrics["recall@2"] - 1.0) < 1e-12


def test_pairwise_retrieval_metrics_excludes_goldless_rows() -> None:
    """Rows without a judged-positive question are counted but not averaged."""
    sims = np.array([[0.1, 0.9], [0.2, 0.9]])
    gold = [set(), {1}]
    metrics = pairwise_retrieval_metrics(sims, gold, ks=[1])
    assert metrics["n_rows_without_gold"] == 1 and metrics["n_rows_with_gold"] == 1
    assert metrics["hit@1"] == 1.0  # only the gold row is averaged
    assert metrics["recall@1"] == 1.0


def test_pairwise_retrieval_metrics_ties_stable_ordering() -> None:
    """Ties prefer earlier columns (deterministic for reproducibility)."""
    sims = np.array([[0.9, 0.5, 0.5, 0.2]])
    metrics = pairwise_retrieval_metrics(sims, [{1}], ks=[2])
    assert metrics["hit@2"] == 1.0  # tied q1 beats q2 (stable order)
    metrics = pairwise_retrieval_metrics(sims, [{2}], ks=[2])
    assert metrics["hit@2"] == 0.0  # the symmetric case loses the tie


def test_pairwise_retrieval_metrics_k_clipped() -> None:
    """k larger than the candidate count considers the full ranking."""
    sims = np.array([[0.2, 0.9]])
    metrics = pairwise_retrieval_metrics(sims, [{1}], ks=[10])
    # top-10 of 2 candidates = all columns; the gold is inside
    assert metrics["hit@10"] == 1.0 and metrics["recall@10"] == 1.0


def test_pairwise_retrieval_metrics_candidate_columns_scope() -> None:
    """A per-row candidate list restricts the ranking (omct probe same-doc scope)."""
    sims = np.array([[0.2, 0.9, 0.8], [0.9, 0.5, 0.4]])
    gold = [{2}, {0}]
    # row 0 may only rank columns [0, 1] (its gold q2 is out of scope);
    # row 1 must rank all three (its gold q0 is its top-1)
    metrics = pairwise_retrieval_metrics(sims, gold, ks=[1], candidate_columns=[[0, 1], [0, 1, 2]])
    assert abs(metrics["hit@1"] - 0.5) < 1e-12  # row0 misses, row1 hits
    assert abs(metrics["recall@1"] - 0.5) < 1e-12


def test_subset_pair_metrics_respects_mask() -> None:
    """The masked variant recomputes metrics on the selected rows only."""
    sims = np.array([[0.2, 0.9], [0.9, 0.1], [0.1, 0.9]])
    gold = [{1}, {1}, {0}]
    metrics = pairwise_retrieval_metrics(sims, gold, ks=[1])
    assert abs(metrics["hit@1"] - 1 / 3) < 1e-12  # only row0 ranks its gold first

    over = subset_pair_metrics(sims, gold, [1], [True, True, False], bucket="over")
    assert over["bucket"] == "over" and over["n_rows"] == 2
    # rows 0-1 ranked like the full-set rows 0-1
    assert abs(over["hit@1"] - 0.5) < 1e-12

    under = subset_pair_metrics(sims, gold, [1], [False, False, True], bucket="within")
    assert under["hit@1"] == 0.0 and under["n_rows_with_gold"] == 1


def test_subset_pair_metrics_restricted_candidates() -> None:
    """candidate_columns reranks the subset rows within their given columns only."""
    sims = np.array([[0.9, 0.8, 0.7], [0.2, 0.5, 1.0]])
    gold = [{0}, {2}]
    columns = [[1, 2], [0, 1]]  # each row loses its top-ranked column

    full = subset_pair_metrics(sims, gold, [1], [True, True], bucket="b")
    assert full["hit@1"] == 1.0  # both rows rank their gold first (all columns)
    restricted = subset_pair_metrics(
        sims, gold, [1], [True, True], bucket="b", candidate_columns=columns
    )
    assert restricted["hit@1"] == 0.0  # golds fall outside each row's candidates
    only_row0 = subset_pair_metrics(
        sims, gold, [1], [True, False], bucket="b", candidate_columns=columns
    )
    assert only_row0["n_rows"] == 1 and only_row0["hit@1"] == 0.0
    only_row1 = subset_pair_metrics(
        sims, gold, [1], [False, True], bucket="b", candidate_columns=columns
    )
    assert only_row1["n_rows"] == 1 and only_row1["hit@1"] == 0.0


def test_pairwise_retrieval_metrics_empty_candidates_rank_nothing() -> None:
    """An empty candidate list ranks nothing; only None means 'all columns'."""
    sims = np.array([[0.9, 0.8, 0.7]])
    assert pairwise_retrieval_metrics(sims, [{0}], ks=[1])["hit@1"] == 1.0
    assert (
        pairwise_retrieval_metrics(sims, [{0}], ks=[1], candidate_columns=[[1, 2, 0]])["hit@1"]
        == 1.0
    )
    assert pairwise_retrieval_metrics(sims, [{0}], ks=[1], candidate_columns=[[]])["hit@1"] == 0.0
