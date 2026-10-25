"""Hand-computed tests for the pairwise metric functions (Phase 2/3 subset)."""

from __future__ import annotations

import numpy as np
import pytest

from cluster_rag.eval.metrics import (
    paired_auc_bootstrap,
    pairwise_retrieval_metrics,
    roc_auc,
    subset_pair_metrics,
)


def _reference_roc_auc(labels, scores) -> float:
    """A naive O(n^2) AUC (the tie-aware pairwise count), for cross-checking."""
    y = list(labels)
    s = list(scores)
    positives = [index for index, label in enumerate(y) if label == 1]
    negatives = [index for index, label in enumerate(y) if label == 0]
    total = 0.0
    for positive in positives:
        for negative in negatives:
            if s[positive] > s[negative]:
                total += 1.0
            elif s[positive] == s[negative]:
                total += 0.5
    return total / (len(positives) * len(negatives))


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


def test_roc_auc_matches_a_naive_reference_on_random_data() -> None:
    """The vectorized implementation agrees with the pairwise count (ties included)."""
    rng = np.random.default_rng(7)
    labels = [1] * 30 + [0] * 40
    scores = np.round(rng.uniform(0.0, 1.0, size=70), 1)  # rounded -> many ties
    assert roc_auc(labels, scores.tolist()) == pytest.approx(
        _reference_roc_auc(labels, scores.tolist())
    )


def test_paired_auc_bootstrap_hand_computed() -> None:
    """A strictly better arm: a positive difference and P(delta > 0) = 1."""
    labels = [1, 1, 0, 0]
    reference = [0.1, 0.2, 0.8, 0.9]
    arm = [0.9, 0.8, 0.2, 0.1]
    result = paired_auc_bootstrap(labels, reference, arm, n_resamples=200, seed=42)
    assert result["auc_reference"] == 0.0
    assert result["auc_arm"] == 1.0
    assert result["delta_auc"] == 1.0
    assert result["ci_low"] == 1.0 and result["ci_high"] == 1.0
    assert result["p_delta_positive"] == 1.0
    assert result["n_resamples"] == 200
    assert result["seed"] == 42


def test_paired_auc_bootstrap_of_identical_scorers_is_zero() -> None:
    """The same scorer on both sides: no difference, no positive resample."""
    labels = [1, 0, 1, 0, 1, 0]
    scores = [0.4, 0.2, 0.9, 0.7, 0.1, 0.8]
    result = paired_auc_bootstrap(labels, scores, scores, n_resamples=100, seed=1)
    assert result["delta_auc"] == 0.0
    assert result["ci_low"] == 0.0 and result["ci_high"] == 0.0
    assert result["p_delta_positive"] == 0.0


def test_paired_auc_bootstrap_is_reproducible_and_paired() -> None:
    """The same seed gives the same interval; the pairing keeps the labels together."""
    labels = [1, 1, 1, 0, 0, 0]
    reference = [0.5, 0.4, 0.3, 0.6, 0.7, 0.8]
    arm = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    first = paired_auc_bootstrap(labels, reference, arm, n_resamples=300, seed=5)
    second = paired_auc_bootstrap(labels, reference, arm, n_resamples=300, seed=5)
    assert first == second
    other = paired_auc_bootstrap(labels, reference, arm, n_resamples=300, seed=6)
    assert other["delta_auc"] == first["delta_auc"]  # the point estimate is seed-free
    assert other["ci_low"] <= first["ci_low"] or other["ci_high"] != first["ci_high"]


def test_paired_auc_bootstrap_requires_both_classes() -> None:
    """A single class makes the AUC undefined (clear error)."""
    with pytest.raises(ValueError, match="both label classes"):
        paired_auc_bootstrap([1, 1], [0.1, 0.2], [0.3, 0.4], n_resamples=10)
    with pytest.raises(ValueError, match="same length"):
        paired_auc_bootstrap([1, 0], [0.1, 0.2], [0.3], n_resamples=10)


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
